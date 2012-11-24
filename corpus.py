#!/usr/bin/env python
#
# Copyright 2012 Michael N. Gagnon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obCONFIG_INSTALLED_BACKUPtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Tests crm114 against a labelled corpus. For each model, X, computes:

    precision = tp / (tp + fp)
    recall =    tp / (tp + fn)

where:
    tp = the number of times an X-item was correctly classified as X
    fp = the number of times a non-X-item was misclassified as X
    tn = the number of times a non-X-item was correctly classified as not X
    fn = the number of times an X-item was misclassified as not X
"""

import crm114

import argparse
import logging
import os
import random
import sys

import json

class LabeledItem:

    def __init__(self, data, actualModel, classification = None):
        self.data = data
        self.actualModel = actualModel

        # holds the classification according to CRM114
        self.classification = classification

def limitItems(items, limit):
    if limit != None:
        items = items[:]
        random.shuffle(items)
        items = items[:limit]
    return items


def lineitems(path, model, limit = None):
    """
    creates a list of LabeledItem objects, by reading one data item per line
    from path.
    """

    with open(path, "r") as f:
        lines = f.readlines()

    lines = limitItems(lines, limit)

    return [LabeledItem(line, model) for line in lines]

class Accuracy:

    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.precision = float(tp) / (tp + fp) if tp + fp > 0 else 0
        self.recall = float(tp) / (tp + fn) if tp + fn > 0 else 0

def delmodels(models):
    """
    deletes models if they exist
    """
    for model in models:
        if os.path.exists(model):
            os.remove(model)

def learnClassify(crm, learnItems, classifyItems, logger, logHeader):
    """
    learnItems and classifyItems are a lists of LabeledItem objects
    for each item in classifyItems, sets item.classification
    """

    delmodels(crm.models)

    for i, item in enumerate(learnItems):
        crm.learn(item.data, item.actualModel)
        logger.debug("%slearned %d/%d, %s", logHeader, i + 1, len(learnItems),
            item.actualModel)

    for i, item in enumerate(classifyItems):
        item.classification = crm.classify(item.data)
        classifiedAs = item.classification.bestMatch.model
        if item.actualModel == classifiedAs:
            logger.debug("%sclassified %d/%d, correctly classified %s", logHeader,
                i + 1, len(classifyItems), item.actualModel)
        else:
            logger.debug("%sclassified %d/%d, misclassified %s as %s", logHeader,
                i + 1, len(classifyItems), item.actualModel,
                item.classification.bestMatch.model)

def partition(items, folds):
    """
    items is a list; divide items into approximately equal folds
    """
    small_fold_size = len(items) / folds
    big_fold_size = small_fold_size + 1

    num_big_folds = len(items) % folds
    num_small_folds = folds - num_big_folds

    result = []
    for i in xrange(0, num_big_folds):
        subresult = items[0 : big_fold_size]
        result.append(subresult)
        items = items[big_fold_size:]
    for i in xrange(0, num_small_folds):
        subresult = items[0 : small_fold_size]
        result.append(subresult)
        items = items[small_fold_size:]

    assert(len(items) == 0)

    return result

def genCrossValidate(items, folds):
    """
    generates a series of (learnItems, classifyItems) pairs
    """

    parts = partition(items, folds)

    for fold in xrange(0, folds):
        learnParts = parts[:]
        del(learnParts[fold])
        learn = [item for part in learnParts for item in part]
        classify = parts[fold]
        yield (fold + 1, learn, classify)

def crossValidate(crm, items, folds, logger):
    """
    classififies every item using N-fold cross validation
    """
    logger.info("crossValidate, folds = %d", folds)

    items = items[:]
    random.shuffle(items)

    for fold, learn, classify in genCrossValidate(items, folds):
        logger.info("beginning fold %d", fold)
        logHeader = "fold %d/%d, " % (fold, folds)
        learnClassify(crm, learn, classify, logger, logHeader)

def holdoutValidate(crm, items, holdout, logger):
    """
    trains on (1 - holdout)-proportion of items, classifies the rest.
    Returns the items that were classified
    """

    logger.info("holdoutValidate, holdout = %f", holdout)

    items = items[:]
    random.shuffle(items)

    if holdout <= 0.0 or holdout >= 1.0:
            raise ValueError("holdout must be in range (0, 1)")

    splitIndex = int(len(items) * holdout)
    classifyItems = items[:splitIndex]
    learnItems = items[splitIndex:]
    learnClassify(crm, learnItems, classifyItems, logger, "")

    return classifyItems

def accuracy(crm, items, threshold):
    """
    computes the accuracy metrics for each model
    returns a dict that maps the model name to its Accuracy object
    """

    accuracy = {}

    # post process all classified items
    for item in items:
        crm.postProcess(item.classification, threshold)

    # bookmark
    for m in crm.models:

        # assuming m is the positive-model
        actualPositives = filter(lambda i: i.actualModel == m, items)
        actualNegatives = filter(lambda i: i.actualModel != m, items)

        tp = sum(i.classification.bestMatch.model == m for i in
            actualPositives)
        fn = sum(i.classification.bestMatch.model != m for i in
            actualPositives)
        
        fp = sum(i.classification.bestMatch.model == m for i in
            actualNegatives)
        tn = sum(i.classification.bestMatch.model != m for i in
            actualNegatives)

        accuracy[m] = Accuracy(tp, fp, tn, fn)

    return accuracy


def pathToModel(path, modelDir):
    """
    converts path == "foo/bar/modelname.txt" to "modelDir/modelname.css"
    """
    oldBasename = os.path.basename(path)
    newBasename = os.path.splitext(oldBasename)[0] + ".css"
    return os.path.join(modelDir, newBasename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tests CRM114 against a " +
            "corpus of lablled data")
    parser.add_argument("--classifier", default=crm114.defaultClassifier,
        help="a string a describing a valid CRM114 classifer. See Section " +
             "'Current Classifiers in CRM114' in the CRM114 book for valid " +
             "values. Default: '%(default)s'")
    parser.add_argument("-o", "--output_dir", default="model/", type=str,
        help="save final models in OUTPUT_DIR. Default: %(default)s")
    parser.add_argument("-f", "--fold", type=int,
        help="perform FOLD-fold cross validation")
    parser.add_argument("--holdout", type=float,
        help="use HOLDOUT proportion of the data as the classification set. " +
             "Use the rest as the classification set. If defined, then " +
             "overrides --fold.") 
    parser.add_argument("-l", "--linedata", nargs="+",
        help="for each line LINEDATA file, read line of data an label it " + 
             "after LINEDATA")
    parser.add_argument("--limit", type=int,
        help="limit each dataset to LIMIT items")
    parser.add_argument("-t", "--toe", action='store_true',
        help="set this flag to only 'train on error.'")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error",
        "critical"], default='info',
        help="logging level. Default: %(default)s")
    
    args = parser.parse_args()
    args.log = args.log.upper()

    logger = logging.getLogger(os.path.basename(__file__))
    handler = logging.StreamHandler()
    handler.setLevel(args.log)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(args.log)

    if args.linedata == None or len(args.linedata) < 2:
        sys.stderr.write("You must specify at least two datasets\n")
        parser.print_help()
        sys.exit(1)

    logger.info("classifier = '%s'", args.classifier)
    logger.info("linedata = %s", args.linedata)
    logger.info("limit = %s", args.limit)
    logger.info("output_dir = %s", args.output_dir)
    logger.info("toe = %s", args.toe)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    models = [pathToModel(linedata, args.output_dir) for linedata in
        args.linedata]

    items = []

    for (path, model) in zip(args.linedata, models):
        newItems = lineitems(path, model, args.limit)
        logger.info("loaded %d %s items", len(newItems), model)
        items += newItems

    crm = crm114.Crm114(models, args.classifier, None, args.toe)

    classifyItems = None

    if args.holdout != None:
        classifyItems = holdoutValidate(crm, items, args.holdout, logger)
    elif args.fold != None:
        crossValidate(crm, items, args.fold, logger)
        classifyItems = items

    logger.info("Building final model")
    for i, item in enumerate(items):
        logger.debug("final model learning %d/%d %s", i + 1, len(items),
            item.actualModel)
        crm.learn(item.data, item.actualModel)

    if classifyItems != None:
        print json.dumps(accuracy(crm, classifyItems, threshold = None),
            indent = 4, sort_keys = True, default = lambda x: x.__dict__ )


