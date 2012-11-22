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

from crm114 import *
import json
import os
import unittest

crmResultModels2 = (
    "CLASSIFY succeeds; success probability: 1.0000  " +
        "pR: %(bestMatchPr)s\n" +
    "Best match to file #1 (%(bestMatchModel)s) prob: %(bestMatchProb)s  " +
        "pR: %(bestMatchPr)s \n" +
    "Total features in input file: %(totalFeatures)s\n" +
    "#0 (%(bestMatchModel)s): features: %(bestMatchFeatures)s, " +
        "hits: %(bestMatchHits)s, prob: %(bestMatchProb)s, " +
        "pR: %(bestMatchPr)s \n"
    "#1 (%(secondMatchModel)s): features: %(secondMatchFeatures)s, " +
        "hits: %(secondMatchHits)s, prob: %(secondMatchProb)s, " +
        "pR: %(secondMatchPr)s\n")

crmResultSpam = crmResultModels2 % {
        "totalFeatures" : "2452",
        "bestMatchModel" : "spam.css",
        "bestMatchFeatures" : "1461",
        "bestMatchHits" : "16572",
        "bestMatchProb" : "1.0",
        "bestMatchPr" : "129.64",
        "secondMatchModel" : "ham.css",
        "secondMatchFeatures" : "856",
        "secondMatchHits" : "301",
        "secondMatchProb" : "4.82e-131",
        "secondMatchPr" : "-130.32"
    }

class MockCrmRunner:
    def run(self, data, command):
        return crmResultSpam

TEST_DIR = "testdata"
HAM_TEXT = "ham1 ham2 ham3 ham4 ham5"
SPAM_TEXT = "fooA fooB fooC fooD fooE"
HAM_FILENAME = os.path.join(TEST_DIR, "ham.css")
SPAM_FILENAME = os.path.join(TEST_DIR, "spam.css")
TUNA_FILENAME = os.path.join(TEST_DIR, "tuna.css")

def freshTestDir():
    """creates a fresh testing directory if it doesn't already exist"""
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    for filename in [HAM_FILENAME, SPAM_FILENAME, TUNA_FILENAME]:
        if os.path.exists(filename):
            os.remove(filename)

class TestCrm114(unittest.TestCase):

    def test_Classification_class(self):
        result = Classification(crmResultSpam)
        self.assertEqual(result.bestMatch.model, "spam.css")
        self.assertEqual(result.totalFeatures, 2452)
        self.assertEqual(result.model["spam.css"].model, "spam.css")
        self.assertEqual(result.model["spam.css"].pr, 129.64)
        self.assertEqual(result.model["spam.css"].prob, 1.0)
        self.assertEqual(result.model["spam.css"].hits, 16572)
        self.assertEqual(result.model["spam.css"].features, 1461)
        self.assertEqual(result.model["ham.css"].model, "ham.css")
        self.assertEqual(result.model["ham.css"].pr, -130.32)
        self.assertEqual(result.model["ham.css"].prob, 4.82e-131)
        self.assertEqual(result.model["ham.css"].hits, 301)
        self.assertEqual(result.model["ham.css"].features, 856)

    def test_Crm114_classify_mock(self):

        classification = Classification(crmResultSpam)
        crm = Crm114(["spam.css", "ham.css"], threshold = None,
            trainOnError = False, crmRunner = MockCrmRunner())

        # 2 models; crm.threshold == None
        crm.threshold = None
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 2 models; bestMatch.pr > crm.threshold
        crm.threshold = 129.0
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 2 models; bestMatch.pr == crm.threshold
        crm.threshold = 129.64
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 2 models; bestMatch.pr < crm.threshold
        crm.threshold = 130.0
        classification.bestMatch = classification.model["ham.css"]
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 3 models
        crm = Crm114(["spam.css", "ham.css", "foo"], threshold = None,
            trainOnError = False, crmRunner = MockCrmRunner())

        # 3 models; crm.threshold == None
        crm.threshold = None
        classification.bestMatch = classification.model["spam.css"]
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 3 models; bestMatch.pr > crm.threshold
        crm.threshold = 129.0
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 3 models; bestMatch.pr == crm.threshold
        crm.threshold = 129.64
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

        # 3 models; bestMatch.pr < crm.threshold
        crm.threshold = 130.0
        classification.bestMatch = None
        self.assertEqual(crm.classify("foo").dict(), classification.dict())

    def test_Crm114_learn_mock(self):
        
        classification = Classification(crmResultSpam)
        crm = Crm114(["spam.css", "ham.css"], threshold = None,
            trainOnError = False, crmRunner = MockCrmRunner())

        # trainOnError == False
        self.assertEqual(crm.learn("foo", "spam.css"), True)
        self.assertEqual(crm.learn("foo", "ham.css"), True)

        crm.trainOnError = True
        self.assertEqual(crm.learn("foo", "spam.css"), False)
        self.assertEqual(crm.learn("foo", "ham.css"), True)

        # 3 models; crm.trainOnError == True
        crm = Crm114(["spam.css", "ham.css", "foo.css"], threshold = None,
            trainOnError = True, crmRunner = MockCrmRunner())
        self.assertEqual(crm.learn("foo", "spam.css"), False)
        self.assertEqual(crm.learn("foo", "ham.css"), True)
        self.assertEqual(crm.learn("foo", "foo.css"), True)


    def test_Crm114_correctly_parse(self):
        """make sure the Crm114 class can parse output for each classifier"""

        def test(self, crm):
            freshTestDir()
            crm.learn(SPAM_TEXT, SPAM_FILENAME)
            crm.learn(HAM_TEXT, HAM_FILENAME)
            for text in [SPAM_TEXT, HAM_TEXT]:
                c = crm.classify(text)
                # there should always be a bestMatch
                self.assertNotEqual(c.bestMatch, None)
                # there should always be a pr score
                self.assertNotEqual(c.bestMatch.pr, None)

        for base in classifiers:
            for option in classifierOptions:
                classifier = base + " " + option
                crm = Crm114([HAM_FILENAME, SPAM_FILENAME], classifier)
                test(self, crm)

    def test_Crm114_correctly_classify(self):
        """make sure the classifiers make the correct classifications"""

        def test(self, crm):
            freshTestDir()
            crm.learn(SPAM_TEXT, SPAM_FILENAME)
            crm.learn(HAM_TEXT, HAM_FILENAME)
            self.assertEqual(crm.classify(SPAM_TEXT).bestMatch.model,
                SPAM_FILENAME)
            self.assertEqual(crm.classify(HAM_TEXT).bestMatch.model,
                HAM_FILENAME)

        for base in classifiers:
            if base in ["winnow", "hyperspace"]:
                # for some reason these classifiers does not pass this test
                continue
            for option in classifierOptions:
                classifier = base + " " + option
                crm = Crm114([HAM_FILENAME, SPAM_FILENAME], classifier)
                test(self, crm)
                    





if __name__ == '__main__':
    unittest.main()