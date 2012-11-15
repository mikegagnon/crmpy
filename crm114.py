#!/usr/bin/env python
#
# Copyright 2012 Mike Gagnon
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
# Python wrapper for CRM114
#
# CRM114 is great machine learning tool. Unfortunatley, it doesn't have the best interface. So here's a simplec Python
# wrapper that uses some (hopefully) decent default settings. Assumes you have installed crm114
#
# Inspiration from https://github.com/bendiken/crm114
# CRM114:
#   http://crm114.sourceforge.net/
#   http://crm114.sourceforge.net/wiki/doku.php?id=download
#   http://crm114.sourceforge.net/wiki/doku.php?id=books
# 

import argparse
import subprocess
import sys

class Crm114:

    crmBinary = "crm"
    defaultClassifier = "osb unique microgroom"

    def __init__(self, classifier = defaultClassifier):
        self.classifier = classifier

    def learn(self, data, modelFile):
        command = [Crm114.crmBinary, """-{ learn %s ( %s ) }""" % (self.classifier, modelFile)]
        p = subprocess.Popen(command, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        p.stdin.write(data)
        (stdout, stderr) = p.communicate()
        print stdout
        sys.stderr.write(stderr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple CRM114 wrapper")
    parser.add_argument("-l", "--learn", help="learn the text from stdin into the LEARN model file.")
    parser.add_argument("-c", "--classify", nargs="*", help="learn the text from stdin into the LEARN model file.")
    args = parser.parse_args()

    crm114 = Crm114()

    if args.learn == None and args.classify == None :
        parser.print_help()
    elif args.learn != None and args.classify != None:
        sys.stderr.write("Cannot learn and classify at the same time\n")
        parser.print_help()
    elif args.learn != None:
        crm114.learn(sys.stdin.read(), args.learn)

