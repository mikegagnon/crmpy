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
crm114.py: a Python wrapper for CRM114. Copyright 2012 Michael N. Gagnon. Apache License.

CRM114 is a great machine learning tool. Unfortunatley, it doesn't have the best interface. So here's a simple Python
wrapper that uses some (hopefully) decent default settings. Assumes you have installed crm114.

CRM114:
   http://crm114.sourceforge.net/
   http://crm114.sourceforge.net/wiki/doku.php?id=download
   The CRM114 book: http://crm114.sourceforge.net/wiki/doku.php?id=books

Tested with CRM114, version 20100106-BlameMichelson (TRE 0.8.0 (BSD)). Because crm114.py parses crm114's output, and
because crm114's output format is not backwards / forwards compatible, minor version changes in CRM114 could break
the output parsers here.
""" 

import argparse
import re
import subprocess
import sys

import json

classifyTemplate = "-{ isolate (:stats:); classify <%(learnMethod)s> (%(models)s) (:stats:); output /:*:stats:/ }"
learnTemplate = "-{ learn <%(learnMethod)s> ( %(model)s ) }"
crmBinary = "crm"

# See Section "Current Classifiers in CRM114" in the CRM114 book for explanations
defaultLearnMethod = "osb unique microgroom"
learnMethods = [
    "",             # Markovian
    "osb",          # Orthogonal Sparse Bigram
    "osbf",         # similar to OSB, but uses some heuristic improvements that sometimes improves accuracy
    "winnow",       # Lightstone's WINNOW algorithm
    "correlate",    # uses letters instead words as features
    "hyperspace",   # calculates distance between unkown text and learned texts
    "entropy crosslink"] # the model that leads to lowest entropy for unknown text is the winner
learnOptions = [
    "unigram",      # only valid for OSB, Winnow, and Hyperspace. only considers unigrams, not ngrams.
    "microgroom",   # automaticall manages size of model files
    "unique"]       # treat features as sets, not multisets. I.e. repeated features have no effect

# regex to match the floating point values, as produced by Crm114
flotingPointReStr = r"(\+|-)?\d+\.?\d*(e(\+|-))?\d*"

# uh oh, just peeked at Crm's source and it looks like each learnMethod has its own output format.
resultReStr = r"""CLASSIFY succeeds; success probability:\s+(?P<successProbability>%(float)s)\s+pR:\s+(?P<successPr>%(float)s)
Best match to file #\d+\s+\((?P<bestMatch>.*)\)\s+prob:\s+(?P<matchProbability>%(float)s)\s+pR:\s+(?P<matchPr>%(float)s)  
Total features in input file:\s+(?P<totalFeatures>\d+)""" % { 'float' : flotingPointReStr }
resultRe = re.compile(resultReStr) 

#0 (python.css): features: 2922, hits: 0, prob: 5.00e-01, pR:   0.00 
subresultReStr = r"#\d+\s+\((?P<model>.*)\):\s+features:\s+(?P<features>\d+),\s+hits:\s+(?P<hits>\d+),\s+prob:\s+" + \
    r"(?P<prob>%(float)s),\s+pR:\s+(?P<pr>%(float)s)" % { 'float' : flotingPointReStr }
subresultRe = re.compile(subresultReStr)

class ClassifyResult:
    """
    Holds the result of a CRM114 classification.

    Fields:
    bestMatch: the name of the model file that represents the best match
    totalFeatures: the number of features extracted from in the input data
    model: a dict that maps model filenames to SubResult objects
    dict: a dictionary representation of the CRM114 classification
    """

    class SubResult:

        def __init__(self, resultLine):
            """
            For a particular CRM114 classification, there is one SubResult object for each model used in the
            classification. Each SubResult object stores information about how closely the input data matches the model.

            Fields:
            model: which model this subresult is for
            features: the number of features that have been learned into this model
            hits: the number of features in input that that hit the model-
            pr: the pR score that represents the likelihood that the input data matches this model. Typically a value in
                the range [-320.0, 320.0]. This value is intended to be more human readable than prob. See subsection
                "Why pR?" on page 171 of the CRM114 book.
            prob: the "probability" that the input data matches this model. pr scores are better.
            """
            match = subresultRe.match(resultLine)
            if not match:
                raise ValueError("Could not parse resultLine: %s" % resultLine)
            self.model = match.group('model')
            self.features = int(match.group('features'))
            self.hits = int(match.group('hits'))
            self.pr = float(match.group('pr'))
            self.prob = float(match.group('prob'))

    def __init__(self, resultString):
        match = resultRe.match(resultString)
        if not match:
            raise ValueError("Could not parse resultString: %s" % resultString)
        self.bestMatch = match.group('bestMatch')
        self.totalFeatures = int(match.group('totalFeatures'))

        lines = resultString.split("\n")
        subresultLines = filter(lambda line: line.startswith("#"), lines)

        subresults = [ClassifyResult.SubResult(line) for line in subresultLines]
        pairs = [(result.model, result) for result in subresults]

        self.model = dict(pairs)

        # just a dict representation of object; for debugging and testing
        self.dict = {"bestMatch" : self.bestMatch, "totalFeatures" : self.totalFeatures, \
            "model" : dict([(pair[0], pair[1].__dict__ ) for pair in self.model.iteritems()]) }

    def __str__(self):
        return json.dumps(self.dict, indent=4)

# Indicates an error in the execution of the crm114 binary
class Crm114Error(Exception):
    pass

def runcrm(data, command):
    p = subprocess.Popen(command, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    p.stdin.write(data)
    (stdout, stderr) = p.communicate()
    if stderr != "" or p.returncode != 0:
        raise Crm114Error(stdout + stderr)
    return stdout

class Crm114:

    def __init__(self, models, learnMethod = defaultLearnMethod):
        """
        models -- list of all model filenames
        learnMethod -- a string a describing a valid CRM114 classifer. See CRM114 documentation for valid values.
        """
        self.learnMethod = learnMethod
        self.models = models
        self.modelsStr = " ".join(models)

    def classify(self, data):
        command = [crmBinary,  classifyTemplate % {"learnMethod" : self.learnMethod, "models" : self.modelsStr} ]
        return ClassifyResult(runcrm(data, command))

    def learn(self, data, model):
        if model not in self.models:
            raise ValueError("Invalid model file: %s" % model)
        command = [crmBinary,  learnTemplate % {"learnMethod" : self.learnMethod, "model" : model} ]
        runcrm(data, command)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple CRM114 wrapper")
    parser.add_argument("-m", "--method", default=defaultLearnMethod, help="a string a describing a valid " +
        "CRM114 classifer. See Section 'Current Classifiers in CRM114' in the CRM114 book for valid values. " + 
        "Default: '%(default)s'")
    parser.add_argument("-l", "--learn", help="learn the text from stdin into the LEARN model file.")
    parser.add_argument("-c", "--classify", nargs="+", help="learn the text from stdin into the LEARN model file.")
    args = parser.parse_args()


    if args.learn == None and args.classify == None :
        parser.print_help()
        sys.exit(1)
    elif args.learn != None and args.classify != None:
        sys.stderr.write("Cannot learn and classify at the same time\n")
        parser.print_help()
        sys.exit(1)
    elif args.learn != None:
        crm = Crm114(args.learn, args.method)
        crm.learn(sys.stdin.read(), args.learn)
    else:
        assert(args.classify != None)
        crm = Crm114(args.classify, args.method)
        crm.classify(sys.stdin.read())



