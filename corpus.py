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

def lineitems(path, model):
    """
    creates a list of LabeledItem objects, by reading one data item per line
    from path.
    """

    with open(path, "r") as f:
        lines = f.readlines()

    return [LabeledItem(line, model) for line in lines]

class Accuracy:

    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.precision = float(tp) / (tp + fp)
        self.recall = float(tp) / (tp + fn)

def delmodels(models):
    """
    deletes models if they exist
    """
    for model in models:
        if os.path.exists(model):
            os.remove(model)

def learnClassify(crm, learnItems, classifyItems):
    """
    learnItems and classifyItems are a lists of LabeledItem objects
    for each item in classifyItems, sets item.classification
    """

    delmodels(crm.models)

    i = 0
    for item in learnItems:
        i += 1
        print "learn %d/%d" % (i, len(learnItems))
        crm.learn(item.data, item.actualModel)

    i = 0
    for item in classifyItems:
        i += 1
        print "classify %d/%d" % (i, len(classifyItems))
        item.classification = crm.classify(item.data)        

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
        yield (learn, classify)

def crossValidateFold(crm, items, folds = 10):
    """
    classififies every item using N-fold cross validation
    """
    items = items[:]
    random.shuffle(items)

    i = 0
    for learn, classify in genCrossValidate(items, folds):
        i += 1
        print "testing fold %d" % i
        learnClassify(crm, learn, classify)

def crossValidate(crm, items, train_size = 0.8):
    """
    trains on training_size-proportion of items, classifies the rest.
    Returns the items that were classified
    """

    items = items[:]
    random.shuffle(items)

    if train_size <= 0.0 or train_size >= 1.0:
            raise ValueError("train_size must be in range (0, 1)")

    splitIndex = int(len(items) * train_size)
    learnItems = items[:splitIndex]
    classifyItems = items[splitIndex:]
    learnClassify(crm, learnItems, classifyItems)

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
    parser.add_argument("-f", "--fold", default=10, type=int,
        help="perform FOLD-fold cross validation")
    parser.add_argument("--train_size", default=None, type=float,
        help="use TRAIN_SIZE proportion of the data as the training set. " +
             "If defined, then overrides --fold.") 
    parser.add_argument("-l", "--linedata", nargs="+",
        help="for each line LINEDATA file, read line of data an label it " + 
             "after LINEDATA")
    parser.add_argument("--limit", type=int,
        help="limit each set model data to LIMIT items")
    parser.add_argument("-t", "--toe", action='store_true',
        help="set this flag to only 'train on error.'")
    args = parser.parse_args()

    if args.linedata == None or len(args.linedata) < 2:
        sys.stderr.write("You must specify at least two datasets\n")
        parser.print_help()
        sys.exit(1)

    model_dir = "temp_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    models = [pathToModel(linedata, model_dir) for linedata in args.linedata]

    items = []

    for (path, model) in zip(args.linedata, models):
        newItems = lineitems(path, model)
        if args.limit != None:
            newItems = newItems[:args.limit]
        print "loaded %d %s items" %(len(newItems), model)
        items += newItems

    crm = crm114.Crm114(models, args.classifier, None, args.toe)

    if args.train_size != None:
        classifyItems = crossValidate(crm, items, args.train_size)

    else:
        crossValidateFold(crm, items, args.fold)
        classifyItems = items

    print json.dumps(accuracy(crm, classifyItems, threshold = None),
        indent = 4, sort_keys = True, default = lambda x: x.__dict__ )

