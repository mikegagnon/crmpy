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
# TODO: Use http://nltk.org/

"""
Normalization funcions. Each normalization function pre-processes a string
before learning and classification.
"""

import re
import string

def functionObjects(functions):
    """
    returns True iff every item in functions is a function object
    """
    return all(hasattr(f, '__call__') for f in functions)

def makeNormalizeFunction(functions):
    """
    functions is either None, or a list of functions, or a list of function
    names.

    return a "normalize function"; a function that accepts a string,
    "normalizes" it according to functions, and returns a string.

    if functions == None, then returns the identity function
    if functions is a list of functions (or function names), yield a function
        that calls each function in succession.
    """

    if functions == None:
        return identity
    else:

        if not functionObjects(functions):
            # convert function names to function objects
            try:
                functions = [globals()[f] for f in functions]
            except KeyError, k:
                raise ValueError(("item %s in functions does not refer to " +
                    "an in-scope function object") % k)

        if not functionObjects(functions):
            raise ValueError("functions contains a non-function object")

        def normalize(string):
            return reduce(lambda string, f: f(string), functions, string)
        return normalize    

# Normalize functions
###############################################################################

def identity(str):
    return str

def lower(string):
    return string.lower()

puncationRe = re.compile("[%s]" % re.escape(string.punctuation))
def rmPunctuation(string):
    return puncationRe.sub('', string)

def startEnd(string):
    return "START " + string + " END"

def echen(string):
    return startEnd(rmPunctuation(lower(string)))

