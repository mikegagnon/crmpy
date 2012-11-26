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

from normalize import *

import unittest

class TestPreprocess(unittest.TestCase):

    def test_makeNormalizeFunction(self):

        # identity function
        f = makeNormalizeFunction(None)
        self.assertEqual(f("foo"), "foo")

        # list of one function object
        f = makeNormalizeFunction([lower])
        self.assertEqual(f("FoO"), "foo")

        # list of multiple function objects
        f = makeNormalizeFunction([lower, rmPunctuation, startEnd])
        self.assertEqual(echen("Foo-BaR"), "START foobar END")

        # list of names of functions objects
        f = makeNormalizeFunction(["lower", "rmPunctuation", "startEnd"])
        self.assertEqual(echen("Foo-BaR"), "START foobar END")

        self.assertRaises(ValueError, makeNormalizeFunction, [None])
        self.assertRaises(ValueError, makeNormalizeFunction, ["nonexistant"])

    def test_echen(self):
        self.assertEqual(echen("foo"), "START foo END")
        self.assertEqual(echen("Foo!"), "START foo END")
        self.assertEqual(echen("Foo! BaR"), "START foo bar END")
        self.assertEqual(echen("Foo-BaR"), "START foobar END")

if __name__ == '__main__':
    unittest.main()
