## Unit tests for FFT!

import fft
import unittest
import cmath
import numpy as np

class TestFFT(unittest.TestCase):
    def testAllZerosSmall(self):
        np.allclose(fft.FFT([0]*16), np.fft.fft([0]*16))

    def testAllOnesSmall(self):
        np.allclose(fft.FFT([1]*16), np.fft.fft([1]*16))

    def testAlternatingSmall(self):
        x = [(-1)**k for k in range(16)]
        np.allclose(fft.FFT(x), np.fft.fft(x))

    def testExpSmall(self):
        e = [cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)]
        np.allclose(fft.FFT(e), np.fft.fft(e))

    def testAllZerosBig(self):
        np.allclose(fft.FFT([0]*(2**10)), np.fft.fft([0]*(2**10)))

    def testAllOnesBig(self):
        np.allclose(fft.FFT([1]*(2**10)), np.fft.fft([1]*(2**10)))

    def testAlternatingBig(self):
        x = [(-1)**k for k in range((2**10))]
        np.allclose(fft.FFT(x), np.fft.fft(x))

    def testExpBig(self):
        e = [cmath.exp(8j*2*cmath.pi*k/(2**10)) for k in range((2**10))]
        np.allclose(fft.FFT(e), np.fft.fft(e))


class TestDFT(unittest.TestCase):
    def testAllZeros(self):
        np.allclose([0]*8, fft.DFT([0]*8))

    def testAllOnes(self):
        np.allclose([8]+[0]*7, fft.DFT([1]*8))

    def testAlternating(self):
        np.allclose([0]*4+[8]+[0]*3, fft.DFT([(-1)**k for k in range(8)]))

    def testExp(self):
        e = [cmath.exp(2j*2*cmath.pi*k/(8)) for k in range(8)]
        np.allclose([0]*2+[8]+[0]*5, fft.DFT(e))
    
    
suiteFFT = unittest.TestLoader().loadTestsFromTestCase(TestFFT)
suiteDFT = unittest.TestLoader().loadTestsFromTestCase(TestDFT)
allTests = unittest.TestSuite([suiteFFT,suiteDFT])

unittest.TextTestRunner(verbosity=2).run(allTests)
