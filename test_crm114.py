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
import unittest

crmResultModels2 = """CLASSIFY succeeds; success probability: 1.0000  pR: %(bestMatchPr)s
Best match to file #1 (%(bestMatchModel)s) prob: %(bestMatchProbability)s  pR: %(bestMatchPr)s 
Total features in input file: %(totalFeatures)s
#0 (%(bestMatchModel)s): features: %(bestMatchFeatures)s, hits: %(bestMatchHits)s, prob: %(bestMatchProb)s, pR: %(bestMatchPr)s 
#1 (%(secondMatchModel)s): features: %(secondMatchFeatures)s, hits: %(secondMatchHits)s, prob: %(secondMatchProb)s, pR: %(secondMatchPr)s
"""

class MockCrmRunner:

    def run(self, data, command):
        if command[1].startswith("-{ isolate (:stats:); classify"):
            if data == "foo":
                return crmResultModels2 % {}


class TestCrm114(unittest.TestCase):

    def test_ClassifyResult(self):
        resultStr = crmResultModels2 % {
                "bestMatchModel" : "spam.css",
                "bestMatchFeatures" : "1461",
                "bestMatchHits" : "16572",
                "bestMatchProb" : "1.0"
                "bestMatchPr" : "129.64",
                "secondMatchModel" : "ham.css",
                "secondMatchFeatures" : "856",
                "secondMatchHits" : "301",
                "secondMatchProb" : "4.82e-131"
                "secondMatchPr" : "-130.32",
            }

        result = ClassifyResult(resultStr)

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


if __name__ == '__main__':
    unittest.main()