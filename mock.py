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

def model(name, features=7, hits=3, prob=0.1, pr = 7.0):
    return {
        'model' : name,
        'features' : features,
        'hits' : hits,
        'prob' : prob,
        'pr' : pr}

def classificationString(models, totalFeatures = 17):

    bestMatch = models[0]
    for model in models:
        if float(model['pr']) > float(bestMatch['pr']):
            bestMatch = model

    body = ("CLASSIFY succeeds; success probability: 1.0000  pR: %s\n" +
            "Best match to file #1 (%s) prob: %s  pR: %s \n" +
            "Total features in input file: %d\n") % (
            bestMatch['pr'],
            bestMatch['model'],
            bestMatch['prob'],
            bestMatch['pr'],
            totalFeatures)

    for i in xrange(0, len(models)):
        model = models[i]
        body += "#%d (%s): features: %s, hits: %s, prob: %s, pR: %s \n" % (
            i, model['model'], model['features'], model['hits'], model['prob'],
            model['pr'])

    return body

def classification(models, totalFeatures = 17):
    return Classification(classificationString(models, totalFeatures))

