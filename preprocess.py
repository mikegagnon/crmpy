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
# TODO: make negative thresholds work
#

"""
A few subclasses of Crm114 that provided different preprocessing steps
"""

import crm114

import re
import string

puncationRe = re.compile("[%s]" % re.escape(string.punctuation))

class EchenCrm(crm114.Crm114):

    def preprocess(self, data):
        return "START " + puncationRe.sub('', data.lower()) + " END"
