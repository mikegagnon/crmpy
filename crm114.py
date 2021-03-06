#!/usr/bin/env python
#
# Copyright 2012 Michael N. Gagnon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Tip for compiling CRM114 on OS X. Edit the make and turn off the -static flag
#

"""
crm114.py: a Python wrapper for CRM114. Copyright 2012 Michael N. Gagnon.
Apache License.

CRM114 is a great machine learning tool. Unfortunatley, it doesn't have the
best interface. So here's a simple Python wrapper. Assumes you have already
installed the crm binary.

CRM114:
   http://crm114.sourceforge.net/
   http://crm114.sourceforge.net/wiki/doku.php?id=download
   The CRM114 book: http://crm114.sourceforge.net/wiki/doku.php?id=books

Tested with CRM114, version 20100106-BlameMichelson (TRE 0.8.0 (BSD)). Because
crm114.py parses crm114's output, and because crm114's output format is not
backwards / forwards compatible, minor version changes in CRM114 could break
the output parsers here.

Performance note: Every call to learn() or classify() invokes the Crm114 binary
as a separate process, whose performance tends to be dominated by disk io.
To improve performance, store your model files in a ramdisk.
""" 

import normalize

import argparse
import re
import os
import subprocess
import sys
import json

classifyTemplate = "-{ isolate (:stats:); classify <%(classifier)s> " + \
    "(%(models)s) (:stats:); output /:*:stats:/ }"
learnTemplate = "-{ learn <%(classifier)s> ( %(model)s ) }"
crmBinary = "crm"

# See "Current Classifiers in CRM114" in the CRM114 book for explanations
defaultClassifier = "osb unique microgroom"
classifiers = [
    "",             # Markovian
    "osb",          # Orthogonal Sparse Bigram
    "osbf",         # similar to OSB, but uses some heuristic improvements
                    # that sometimes improves accuracy
    "winnow",       # Lightstone's WINNOW algorithm
    "correlate",    # uses letters instead words as features
    "hyperspace",   # calculates distance between unkown text and learned texts
    "entropy crosslink"] # the model that leads to lowest entropy for unknown
                         # text is the winner
classifierOptions = [
    "unigram",      # only valid for OSB, Winnow, and Hyperspace. only
                    # considers unigrams, not ngrams.
    "microgroom",   # automaticall manages size of model files
    "unique"]       # treat features as sets, not multisets. I.e. repeated
                    # features have no effect

# regex to match the floating point values, as produced by Crm114
flotingPointReStr = r"(\+|-)?\d+\.?\d*(e(\+|-))?\d*"

# uh oh, just peeked at Crm's source and it looks like each classifier has its
# own output format.
classificationReStr = (r"CLASSIFY succeeds;\s"
    r"success probability:\s+(?P<successProbability>%(float)s)\s+" +
    r"pR:\s+(?P<successPr>%(float)s)\s*" +
    r"Best match to file #\d+\s+\((?P<bestMatch>.*)\)\s+"
    r"(weight:\s+%(float)s\s+)?"
    r"(prob:\s+(?P<matchProbability>%(float)s)\s+)?"
    r"pR:\s+(?P<matchPr>%(float)s)\s*" +
    r"Total features in input file:\s+(?P<totalFeatures>\d+)""") % \
        { 'float' : flotingPointReStr }
classificationRe = re.compile(classificationReStr) 

subClassificationReStr = (r"#\d+\s+\((?P<model>.*)\):\s+" +
    r"(features:\s+(?P<features>%(float)s)(\s+\(%(float)s%%\))?,\s+)?" +
    r"(unseen:\s+(?P<unseen>%(float)s),\s+)?" +
    r"(weight:\s+(?P<weight>%(float)s),\s+)?" +
    r"(hits:\s+(?P<hits>\d+),\s+)?" +
    r"(entropy:\s+(?P<entropy>%(float)s),\s+)?" +
    r"(jumps:\s+(?P<jumps>\d+),\s+)?" +
    r"(radiance:\s+(?P<radiance>%(float)s),\s+)?" +
    r"(ufeats: \d+,\s+)?" +
    r"(L1: \d+ L2: \d+ L3: \d+, l4: \d+\s+)?" +
    r"(prob:\s+(?P<prob>%(float)s),\s+)?" +
    r"pR:\s+(?P<pr>%(float)s)") % \
        { 'float' : flotingPointReStr }
subClassificationRe = re.compile(subClassificationReStr)

class Classification:
    """
    Holds the result of a CRM114 classification.

    Fields:
    bestMatch: the ModelMatch that represents the best match
    totalFeatures: the number of features extracted from in the input data
    model: a dict that maps model filenames to ModelMatch objects
    """

    class ModelMatch:

        def __init__(self, modelLine):
            """
            For a particular CRM114 classification, there is one ModelMatch
            object for each model used in the classification. Each ModelMatch
            object stores information about how closely the input data matches
            the model. Note, some of the fields may be None because different
            classifiers produce values for different fields. There will always
            be a pr field.

            Fields:
            model: which model this ModelMatch is for
            features: the number of features that have been learned into this
                model
            hits: the number of features in input that that hit the model-
            pr: the pR score that represents the likelihood that the input data
                matches this model. Typically a value in the range [-320.0,
                320.0]. This value is intended to be more human readable than
                prob. See section "Why pR?" on page 171 of the CRM114 book.
            prob: the "probability" that the input data matches this model. pr
                scores are better.
            """
            match = subClassificationRe.match(modelLine)
            if not match:
                raise ValueError("Could not parse modelLine: %s" % modelLine)
            self.model = match.group('model')

            self.pr = float(match.group('pr'))

            featuresStr = match.group('features')
            self.features = float(featuresStr) if featuresStr else None

            hitsStr = match.group('hits')
            self.hits = int(hitsStr) if hitsStr else None

            probStr = match.group('prob')
            self.prob = float(probStr) if probStr else None

    def __init__(self, classificationString):
        match = classificationRe.match(classificationString)
        if not match:
            raise ValueError("Could not parse classificationString: %s" %
                classificationString)
        self.totalFeatures = int(match.group('totalFeatures'))

        lines = classificationString.split("\n")
        modelLines = filter(lambda line: line.startswith("#"), lines)

        modelMatches = [Classification.ModelMatch(line) for line in modelLines]
        pairs = [(modelMatch.model, modelMatch) for modelMatch in modelMatches]

        self.model = dict(pairs)

        self.bestMatch = self.model[match.group('bestMatch')]

    def dict(self):
        """returns a dict representation of object; for debugging and
        testing"""
        if self.bestMatch:
            bestMatch = self.bestMatch.__dict__
        else:
            bestMatch = None
        model = [(m[0], m[1].__dict__ ) for m in self.model.iteritems()]

        return {"bestMatch" : bestMatch, "totalFeatures" : self.totalFeatures,
            "model" : dict(model) }

    def __str__(self):
        return json.dumps(self.dict(), indent=4, sort_keys = True)

# Indicates an error in the execution of the crm114 binary
class Crm114Error(Exception):
    pass

# implemented as a class for mockability
class CrmRunner:

    def run(self, data, command):
        p = subprocess.Popen(command, stdin = subprocess.PIPE, stdout =
            subprocess.PIPE, stderr = subprocess.PIPE)
        p.stdin.write(data)
        (stdout, stderr) = p.communicate()
        if stderr != "" or p.returncode != 0:
            raise Crm114Error("commond = " + str(command) + "\n" + stdout + stderr)
        return stdout

class Crm114:
    """CRM114 wrapper. Provides learn and classify methods."""

    def __init__(self, models, classifier = defaultClassifier,
            threshold = None, trainOnError = False, normalizeFunction = None,
            crmRunner=None):
        """
        models -- list of all model filenames
        classifer -- a string a describing a valid CRM114 classifer. See CRM114
            documentation for valid values.
        threshold -- affects how classify() chooses bestMatch, but threshold
            only makes sense when there are two models.
            if len(models) > 2 then threshold must be None
            if len(models) == 2 and threshold == None, then classify() just
                yields the bestMatch given by the CRM114 binary.
            if len(models) == 2 and threshold != None, then classify() sets
                bestMatch to firstModel, iff firstmodel's pr score >= threshold
        if not trainOnError (default), then learn() always learns the data
        if trainOnError, then learn() only learns the data when the classifer
            makes a mistake when classifying the data.
        normalizeFunction: a function that "normalizes" a string before
            learning or classifying
        """

        if len(models) < 2:
            raise ValueError("models must contain at least 2 model filenames")
        if len(models) > 2 and threshold != None:
            raise ValueError("threshold only makes sense when len(models)==2")

        self.models = models

        self.classifier = classifier
        self.threshold = threshold
        self.trainOnError = trainOnError
        self.normalize = normalizeFunction

        self.classifyCommand = [crmBinary, classifyTemplate %
            { "classifier" : self.classifier, "models" : " ".join(models) }]

        if crmRunner == None:
            self.crmRunner = CrmRunner()
        else:
            self.crmRunner = crmRunner

    def postprocess(self, classification, threshold):
        """
        post-process classification according to threshold
        """
        if threshold == None:
            newModel = classification.bestMatch.model
        elif classification.model[self.models[0]].pr >= threshold:
            newModel = self.models[0]
        else:
            newModel = self.models[1]

        classification.bestMatch = classification.model[newModel]

    def preprocess(self, data):
        """
        override this to pre-process strings before classifying or learning
        """
        return data

    def classify(self, data):
        """return the Classification from running crm114 on data"""
        
        data = self.normalize(data)
        c = Classification(self.crmRunner.run(data, self.classifyCommand))

        self.postprocess(c, self.threshold)

        return c

    def learn(self, data, model):
        """
        returns True if learned; returns False otherwise
        """

        data = self.normalize(data)

        # true iff every model file exists
        allAvailable = all(os.path.exists(model) for model in self.models)

        if (self.trainOnError and allAvailable and
            self.classify(data).bestMatch != None and
            self.classify(data).bestMatch.model == model):
            # no need to learn because the classifier already knows how to
            # correctly classify data
            return False
        else:
            if model not in self.models:
                raise ValueError("Invalid model file: %s" % model)
            command = [ crmBinary,
                        learnTemplate % { "classifier" : self.classifier,
                                          "model" : model} ]
            self.crmRunner.run(data, command)
            return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple CRM114 wrapper")
    parser.add_argument("--classifier", default=defaultClassifier,
        help="a string a describing a valid CRM114 classifer. See Section " +
             "'Current Classifiers in CRM114' in the CRM114 book for valid " +
             "values. Default: '%(default)s'")
    parser.add_argument("-l", "--learn",
        help="learn the text from stdin into the LEARN model file.")
    parser.add_argument("-c", "--classify", nargs="+",
        help="learn the text from stdin into the LEARN model file.")
    parser.add_argument("-t", "--toe", action='store_true',
        help="set this flag to with --learn, to only 'train on error.'")
    parser.add_argument("-r", "--threshold", type=float,
        help="threshold for pr score (see documentation in source)'")
    parser.add_argument("-n", "--normalize", nargs="+",
        help="A list of normalize functions, e.g. 'lower startEnd'; see " +
             "normalize.py. Before learning or classifying, the input string" +
             "will be passed through each normalize function, in order.")
    args = parser.parse_args()

    if args.learn == None and args.classify == None :
        parser.print_help()
        sys.exit(1)
    elif args.learn != None and args.classify != None:
        sys.stderr.write("Cannot learn and classify at the same time\n")
        parser.print_help()
        sys.exit(1)

    if args.learn != None:
        models = list(args.learn)
    else:
        assert(args.classify != None)
        models = args.classify

    f = normalize.makeNormalizeFunction(args.normalize)
    crm = Crm114(models, args.classifier, args.threshold, args.toe, f)

    if args.learn != None:
        crm.learn(sys.stdin.read(), args.learn)
    else:
        assert(args.classify != None)
        print crm.classify(sys.stdin.read())
