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


class TestDFT(unittest.TestCase):
    def testAllZeros(self):
        self.almostEqualArrays([0]*8, fft.DFT([0]*8))

    def testAllOnes(self):
        self.almostEqualArrays([8]+[0]*7, fft.DFT([1]*8))

    def testAlternating(self):
        self.almostEqualArrays([0]*4+[8]+[0]*3, fft.DFT([(-1)**k for k in range(8)]))

    def testExp(self):
        e = [cmath.exp(2j*2*cmath.pi*k/(8)) for k in range(8)]
        self.almostEqualArrays([0]*2+[8]+[0]*5, fft.DFT(e))
    
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

    suite.addTest(TestDFT('testAllZeros'))
    suite.addTest(TestDFT('testAllOnes'))
    suite.addTest(TestDFT('testAlternating'))
    suite.addTest(TestDFT('testExp'))
    return suite

unittest.TextTestRunner(verbosity=2).run(suite())
