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
# TODO: iterate over classifiers, output ROC data
# TODO: map/reduce learning via cssmerge and multiprocess module

"""
Tests and/or train crm114 against a labeled corpus. For each model, X, computes

    precision = tp / (tp + fp)
    recall =    tp / (tp + fn)

where:
    tp = the number of times an X-item was correctly classified as X
    fp = the number of times a non-X-item was misclassified as X
    tn = the number of times a non-X-item was correctly classified as not X
    fn = the number of times an X-item was misclassified as not X
"""

import crm114
import normalize

import argparse
import json
import logging
import os
import random
import sys


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

def learn(crm, learnItems, logger, logHeader = ""):
    """
    learnItems is a list of LabeledItem objects
    runs crm.learn on each item in learnItems
    """

    for i, item in enumerate(learnItems):
        crm.learn(item.data, item.actualModel)
        logger.debug("%slearned %d/%d, %s", logHeader, i + 1, len(learnItems),
            item.actualModel)

def classify(crm, classifyItems, logger, logHeader = ""):
    """
    classifyItems is a list of LabeledItem objects
    runs crm.classify on each item in learnItems; sets item.classification
    returns items that were classified, which is classifyItems
    """

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
    return classifyItems


def learnClassify(crm, learnItems, classifyItems, logger, logHeader = ""):
    """
    learnItems and classifyItems are a lists of LabeledItem objects
    learns and classified the items, setting item.classification for each item
    in classifyItems
    returns items that were classified, which is classifyItems
    """

    delmodels(crm.models)
    learn(crm, learnItems, logger, logHeader)
    return classify(crm, classifyItems, logger, logHeader)

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
    classififies every item using N-fold cross validation.
    returns items that were classified, which is all items
    """
    logger.info("crossValidate, folds = %d", folds)

    items = items[:]
    random.shuffle(items)

    for fold, learn, classify in genCrossValidate(items, folds):
        logger.info("beginning fold %d", fold)
        logHeader = "fold %d/%d, " % (fold, folds)
        learnClassify(crm, learn, classify, logger, logHeader)

    return items

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
        crm.postprocess(item.classification, threshold)

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

def minMaxPr(items):
    """
    returns (min, max) where min the minimum pr score in items, and max is the
    maximum
    """
    # determine lower- and upper-bounds for all pr scores
    prScores = [model.pr for item in items for model in
                    item.classification.model.values()]

    return (min(prScores), max(prScores))

def varyThreshold(crm, items, dataPoints = 100):
    """
    items is a list of classified LabeledItem objects.
    Explores N different values for threshold, where N = dataPoints.
    Returns a dict where model maps to a list of (threshold, precision, recall)
    triples
    """

    if len(crm.models) != 2:
        raise ValueError("varyThreshold only makes sense when there are more" +
            " than two models")

    result = dict((m, []) for m in crm.models)

    low, high = minMaxPr(items)
    increment = (high - low) / (dataPoints + 1)

    threshold = low + increment
    for i in xrange(dataPoints):
        a = accuracy(crm, items, threshold)
        for m in crm.models:
            result[m].append((threshold, a[m].precision, a[m].recall))
        threshold += increment

    return result

def pathToModel(path, modelDir):
    """
    converts path == "foo/bar/modelname.txt" to "modelDir/modelname.css"
    """
    oldBasename = os.path.basename(path)
    newBasename = os.path.splitext(oldBasename)[0] + ".css"
    return os.path.join(modelDir, newBasename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tests CRM114 against a " +
            "corpus of labeled data")
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
    parser.add_argument("-l", "--learn", action='store_true',
        help="learn the labeled data into fresh models.")
    parser.add_argument("-c", "--classify", action='store_true',
        help="classify the labeled data according to existing models") 
    parser.add_argument("--linedata", nargs="+",
        help="for each line LINEDATA file, read line of data an label it " + 
             "after LINEDATA")
    parser.add_argument("--limit", type=int,
        help="limit each dataset to LIMIT items")
    parser.add_argument("-t", "--toe", action='store_true',
        help="set this flag to only 'train on error.'")
    parser.add_argument("-n", "--normalize", nargs="+",
        help="A list of normalize functions, e.g. 'lower startEnd'; see " +
             "normalize.py. Before learning or classifying, the input string" +
             "will be passed through each normalize function, in order.")
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

    modeFlags = [
        ('--learn', args.learn),
        ('--classify', args.classify),
        ('--holdout', args.holdout),
        ('--fold', args.fold)]
    modeFlagStrs = [m[0] for m in modeFlags]
    # the set of modeFlags that are activated
    activatedMode = [m[0] for m in modeFlags if m[1]]
    if len(activatedMode) == 0 or len(activatedMode) > 1:
        sys.stderr.write(("You must specify exactly one of the following " +
            "modes: %s\n") % ", ".join(modeFlagStrs))
        sys.exit()

    logger.info("classifier = '%s'", args.classifier)
    logger.info("linedata = %s", args.linedata)
    logger.info("limit = %s", args.limit)
    logger.info("output_dir = %s", args.output_dir)
    logger.info("toe = %s", args.toe)
    logger.info("normalize = %s", args.normalize)

    normalizeFunction = normalize.makeNormalizeFunction(args.normalize)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    models = [pathToModel(linedata, args.output_dir) for linedata in
        args.linedata]

    items = []

    for (path, model) in zip(args.linedata, models):
        newItems = lineitems(path, model, args.limit)
        logger.info("loaded %d %s items", len(newItems), model)
        items += newItems

    crm = crm114.Crm114(models, args.classifier, None, args.toe,
        normalizeFunction)

    classifyItems = None

    if args.classify:
        classifyItems = classify(crm, items, logger)
    elif args.holdout != None:
        classifyItems = holdoutValidate(crm, items, args.holdout, logger)
    elif args.fold != None:
        classifyItems = crossValidate(crm, items, args.fold, logger)

    if args.learn or args.holdout != None or args.fold != None:
        logger.info("Building final model")
        learn(crm, items, logger, "final model")

    print 1
    if classifyItems != None:
        print 2
        print json.dumps(accuracy(crm, classifyItems, threshold = None),
            indent = 4, sort_keys = True, default = lambda x: x.__dict__ )


