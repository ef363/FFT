## Unit tests for FFT!

import fft
import unittest
import cmath

class TestFFT(unittest.TestCase):
    def testAllZerosSmall(self):
        self.assertEqual(fft.FFT([0]*16), fft.DFT([0]*16))

    def testAllOnesSmall(self):
        self.almostEqualArrays(fft.FFT([1]*16), fft.DFT([1]*16))

    def testAlternatingSmall(self):
        x = [(-1)**k for k in range(16)]
        self.almostEqualArrays(fft.FFT(x), fft.DFT(x))

    def testExpSmall(self):
        e = [cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)]
        self.almostEqualArrays(fft.FFT(e), fft.DFT(e))

    def testAllZerosBig(self):
        self.assertEqual(fft.FFT([0]*(2**10)), fft.DFT([0]*(2**10)))

    def testAllOnesBig(self):
        self.almostEqualArrays(fft.FFT([1]*(2**10)), fft.DFT([1]*(2**10)))

    def testAlternatingBig(self):
        x = [(-1)**k for k in range((2**10))]
        self.almostEqualArrays(fft.FFT(x), fft.DFT(x))

    def testExpBig(self):
        e = [cmath.exp(8j*2*cmath.pi*k/(2**10)) for k in range((2**10))]
        self.almostEqualArrays(fft.FFT(e), fft.DFT(e))



    def almostEqualArrays(self, v1, v2):
        self.assertEqual(len(v1), len(v2))
        for i in range(len(v1)):
            self.assertAlmostEqual(v1[i],v2[i])



def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFFT('testAllZerosSmall'))
    suite.addTest(TestFFT('testAllOnesSmall'))
    suite.addTest(TestFFT('testAlternatingSmall'))
    suite.addTest(TestFFT('testExpSmall'))
    suite.addTest(TestFFT('testAllZerosBig'))
    suite.addTest(TestFFT('testAllOnesBig'))
    suite.addTest(TestFFT('testAlternatingBig'))
    suite.addTest(TestFFT('testExpBig'))
    return suite

unittest.TextTestRunner(verbosity=2).run(suite())
