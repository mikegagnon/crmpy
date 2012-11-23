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

#from crm114 import *

class LabeledItem:

    def __init__(self, data, actualModel, classification = None):
        self.data = data
        self.actualModel = actualModel

        # holds the classification according to CRM114
        self.classification = classification

class Accuracy:

    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.precision = float(tp) / (tp + fp)
        self.recall = float(tp) / (tp + fn)

DEFAULT_MODEL_PATH = "temp_models"

def freshDir(models, path):
    """
    creates a fresh directory for cross validation, if it doesn't already
    exist
    """
    if not os.path.exists(path):
        os.mkdir(path)
    for model in models:
        filename = os.path.join(path, model)
        if os.path.exists(filename):
            os.remove(filename)

def learnClassify(crm, learnItems, classifyItems, path = DEFAULT_MODEL_PATH):
    """
    learnItems and classifyItems are a lists of LabeledItem objects
    for each item in classifyItems, sets item.classification
    """
    
    freshDir(crm.models, path)

    for item in self.learnItems:
        crm.learn(item.data, item.actualModel)

    for item in self.classifyItems:
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

def crossValidate(crm, items, folds = 10, path = DEFAULT_MODEL_PATH):
    """
    classififies every item using cross validation
    """

    for learn, classify in genCrossValidate(items, folds):
        learnClassify(crm, learn, classify, path):

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




