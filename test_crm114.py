
from crm114 import *
import json
import unittest

class TestCrm114(unittest.TestCase):

    def test_ClassifyResult(self):
        resultStr = """CLASSIFY succeeds; success probability: 1.0000  pR: 304.6527
Best match to file #1 (spam.css) prob: 1.0000  pR: 129.64  
Total features in input file: 2452
#0 (ham.css): features: 856, hits: 301, prob: 4.82e-131, pR: -130.32 
#1 (spam.css): features: 1461, hits: 16572, prob: 1.00e+00, pR: 129.64"""

        result = ClassifyResult(resultStr)

        self.assertEqual(result.bestMatch, "spam.css")
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