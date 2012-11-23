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

import unittest

class TestCorpus(unittest.TestCase):

    def test_accuracy(self):
        crm = Crm114(["ham.css", "spam.css"])
        corpus = Corpus(crm, [], [])

        corpus.classifyItems = [
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

        accuracy = corpus.accuracy(threshold = None)
        self.assertEquals(accuracy["ham.css"].tp, 3)
        self.assertEquals(accuracy["ham.css"].fp, 1)
        self.assertEquals(accuracy["ham.css"].tn, 2)
        self.assertEquals(accuracy["ham.css"].fn, 2)
        self.assertEquals(accuracy["ham.css"].precision, 3.0 / 4.0)
        self.assertEquals(accuracy["ham.css"].recall, 3.0 / 5.0)
        self.assertEquals(accuracy["spam.css"].tp, 2)
        self.assertEquals(accuracy["spam.css"].fp, 2)
        self.assertEquals(accuracy["spam.css"].tn, 3)
        self.assertEquals(accuracy["spam.css"].fn, 1)
        self.assertEquals(accuracy["spam.css"].precision, 2.0 / 4.0)
        self.assertEquals(accuracy["spam.css"].recall, 2.0 / 3.0)

        # if ham.pr >= -20.0, then ham is considered best match
        accuracy = corpus.accuracy(threshold = -20.0)
        self.assertEquals(accuracy["ham.css"].tp, 4)
        self.assertEquals(accuracy["ham.css"].fp, 2)
        self.assertEquals(accuracy["ham.css"].tn, 1)
        self.assertEquals(accuracy["ham.css"].fn, 1)
        self.assertEquals(accuracy["ham.css"].precision, 4.0 / 6.0)
        self.assertEquals(accuracy["ham.css"].recall, 4.0 / 5.0)
        self.assertEquals(accuracy["spam.css"].tp, 1)
        self.assertEquals(accuracy["spam.css"].fp, 1)
        self.assertEquals(accuracy["spam.css"].tn, 4)
        self.assertEquals(accuracy["spam.css"].fn, 2)
        self.assertEquals(accuracy["spam.css"].precision, 1.0 / 2.0)
        self.assertEquals(accuracy["spam.css"].recall, 1.0 / 3.0)

if __name__ == '__main__':
    unittest.main()
