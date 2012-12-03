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

from corpus import *
from crm114 import *
import mock

import pprint
import unittest

class TestCorpus(unittest.TestCase):

    def test_accuracy(self):
        crm = Crm114(["ham.css", "spam.css"])

        classifyItems = [
            # actual ham
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=30.2),
                 mock.model("spam.css", pr=-16.0)]
                )),
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=16.2),
                 mock.model("spam.css", pr=-16.0)]
                )),
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=16.2),
                 mock.model("spam.css", pr=-16.0)]
                )),
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=-10.0),
                 mock.model("spam.css", pr=99.0)]
                )),
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=-40.0),
                 mock.model("spam.css", pr=80.0)]
                )),
            # actual spam
            LabeledItem(None, "spam.css", mock.classification(
                [mock.model("ham.css", pr=-5.0),
                 mock.model("spam.css", pr=80.0)]
                )),
            LabeledItem(None, "spam.css", mock.classification(
                [mock.model("ham.css", pr=-45.0),
                 mock.model("spam.css", pr=89.0)]
                )),
            LabeledItem(None, "spam.css", mock.classification(
                [mock.model("ham.css", pr=85.0),
                 mock.model("spam.css", pr=-25.0)]
                ))
            ]

        result = accuracy(crm, classifyItems, threshold = None)
        self.assertEquals(result["ham.css"].tp, 3)
        self.assertEquals(result["ham.css"].fp, 1)
        self.assertEquals(result["ham.css"].tn, 2)
        self.assertEquals(result["ham.css"].fn, 2)
        self.assertEquals(result["ham.css"].precision, 3.0 / 4.0)
        self.assertEquals(result["ham.css"].recall, 3.0 / 5.0)
        self.assertEquals(result["spam.css"].tp, 2)
        self.assertEquals(result["spam.css"].fp, 2)
        self.assertEquals(result["spam.css"].tn, 3)
        self.assertEquals(result["spam.css"].fn, 1)
        self.assertEquals(result["spam.css"].precision, 2.0 / 4.0)
        self.assertEquals(result["spam.css"].recall, 2.0 / 3.0)

        # if ham.pr >= -20.0, then ham is considered best match
        result = accuracy(crm, classifyItems, threshold = -20.0)
        self.assertEquals(result["ham.css"].tp, 4)
        self.assertEquals(result["ham.css"].fp, 2)
        self.assertEquals(result["ham.css"].tn, 1)
        self.assertEquals(result["ham.css"].fn, 1)
        self.assertEquals(result["ham.css"].precision, 4.0 / 6.0)
        self.assertEquals(result["ham.css"].recall, 4.0 / 5.0)
        self.assertEquals(result["spam.css"].tp, 1)
        self.assertEquals(result["spam.css"].fp, 1)
        self.assertEquals(result["spam.css"].tn, 4)
        self.assertEquals(result["spam.css"].fn, 2)
        self.assertEquals(result["spam.css"].precision, 1.0 / 2.0)
        self.assertEquals(result["spam.css"].recall, 1.0 / 3.0)

    def test_partition(self):

        self.assertEqual(partition([1,2,3], 1), [[1,2,3]])
        self.assertEqual(partition([1,2,3], 2), [[1,2],[3]])
        self.assertEqual(partition([1,2,3], 3), [[1],[2],[3]])
        self.assertEqual(partition([1,2,3], 4), [[1],[2],[3], []])

        self.assertEqual(partition([1,2,3,4,5,6,7], 1), [[1,2,3,4,5,6,7]])
        self.assertEqual(partition([1,2,3,4,5,6,7], 2), [[1,2,3,4],[5,6,7]])
        self.assertEqual(partition([1,2,3,4,5,6,7], 3), [[1,2,3],[4,5],[6,7]])
        self.assertEqual(partition([1,2,3,4,5,6,7], 4),
                [[1,2],[3,4],[5,6],[7]])
        self.assertEqual(partition([1,2,3,4,5,6,7], 5),
                [[1,2],[3,4],[5],[6],[7]])

    def test_genCrossValidate(self):

        # [1,2,3,4], [5,6,7], [8,9,10]
        items = [1,2,3,4,5,6,7,8,9,10]
        learnClassifyPairs = list(genCrossValidate(items, 3))

        self.assertEqual(learnClassifyPairs, [
                (1, [5,6,7, 8,9,10], [1,2,3,4]),
                (2, [1,2,3,4, 8,9,10], [5,6,7]),
                (3, [1,2,3,4, 5,6,7], [8, 9, 10])
            ])

    def test_minMaxPr(self):

        classifyItems = [
            # actual ham
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=30.2),
                 mock.model("spam.css", pr=-18.0)]
                )),
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=16.2),
                 mock.model("spam.css", pr=-21.0)]
                )),
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=57.2),
                 mock.model("spam.css", pr=-16.0)]
                ))
            ]

        self.assertEquals(minMaxPr(classifyItems), (-21.0, 57.2))

    def test_varyThreshold(self):

        crm = Crm114(["ham.css", "spam.css"])

        items = [
            # actual ham
            # threshold -60 -> ham -> correct
            # threshold -20 -> ham -> correct
            # threshold  20 -> ham -> correct
            # threshold  60 -> ham -> correct
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=100.0),
                 mock.model("spam.css", pr=-100.0)]
                )),
            # actual ham
            # threshold -60 -> ham -> correct
            # threshold -20 -> ham -> correct
            # threshold  20 -> ham -> correct
            # threshold  60 -> spam -> mistake
            LabeledItem(None, "ham.css", mock.classification(
                [mock.model("ham.css", pr=20.0),
                 mock.model("spam.css", pr=-20.0)]
                )),
            # actual spam
            # threshold -60 -> ham -> mistake
            # threshold -20 -> spam -> correct
            # threshold  20 -> spam -> correct
            # threshold  60 -> spam -> correct
            LabeledItem(None, "spam.css", mock.classification(
                [mock.model("ham.css", pr=-30.0),
                 mock.model("spam.css", pr=30.0)]
                )),
            # actual spam
            # threshold -60 -> spam -> correct
            # threshold -20 -> spam -> correct
            # threshold  20 -> spam -> correct
            # threshold  60 -> spam -> correct
            LabeledItem(None, "spam.css", mock.classification(
                [mock.model("ham.css", pr=-100.0),
                 mock.model("spam.css", pr=100.0)]
                ))
            ]

        result = varyThreshold(crm, items, 4)
        # min = -100
        # max = 100
        # increment = 40
        # precision = tp / (tp + fp)
        # recall =    tp / (tp + fn)
        expected = {'ham.css': [
                        (-60.0, 2.0 / 3.0, 1.0),
                        (-20.0, 1.0, 1.0),
                        (20.0, 1.0, 1.0),
                        (60.0, 1.0, 0.5)],
                    'spam.css': [
                        (-60.0, 1.0, 0.5),
                        (-20.0, 1.0, 1.0),
                        (20.0, 1.0, 1.0),
                        (60.0, 2.0/3.0, 1.0)]}

        self.assertEquals(expected, result)


if __name__ == '__main__':
    unittest.main()
