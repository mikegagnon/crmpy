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

class Corpus:

    def __init__(self, crm, learnItems, classifyItems):
        """
        crm is a Crm114 object
        learnItems and classifyItems are a lists of LabeledItem objects
        """

        self.crm = crm
        self.learnItems = learnItems
        self.classifyItems = classifyItems

        for item in self.learnItems:
            self.crm.learn(item.data, self.actualModel)

        for item in self.classifyItems:
            item.classification = self.crm.classify(item.data)        

    def accuracy(self, threshold):
        """
        computes the accuracy metrics for each model, assuming threshold
        returns a dict that maps the model name to its Accuracy object
        """

        accuracy = {}

        # post process all classified items
        for item in self.classifyItems:
            self.crm.postProcess(item.classification, threshold)

        # bookmark
        for m in self.crm.models:

            # assuming m is the positive-model
            actualPositives = filter(lambda i: i.actualModel == m,
                self.classifyItems)
            actualNegatives = filter(lambda i: i.actualModel != m,
                self.classifyItems)

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

            

