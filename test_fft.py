## Unit tests for FFT!

import fft
import unittest
import cmath
import numpy as np

class TestFFT(unittest.TestCase):
    def testAllZerosSmall(self):
        np.testing.assert_allclose(fft.FFT(np.zeros(16)), np.fft.fft(np.zeros(16)), rtol=0, atol=1e-10)

    def testAllOnesSmall(self):
        np.testing.assert_allclose(fft.FFT(np.ones(16)), np.fft.fft(np.ones(16)), rtol=0, atol=1e-10)

    def testAlternatingSmall(self):
        x = np.array([(-1)**k for k in range(16)])
        np.testing.assert_allclose(fft.FFT(x), np.fft.fft(x), rtol=0, atol=1e-10)

    def testExpSmall(self):
        e = np.array([cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)])
        np.testing.assert_allclose(fft.FFT(e), np.fft.fft(e), rtol=0, atol=1e-10)

    def testAllZerosBig(self):
        np.testing.assert_allclose(fft.FFT(np.zeros(2**10)), np.fft.fft(np.zeros(2**10)), rtol=0, atol=1e-10)

    def testAllOnesBig(self):
        np.testing.assert_allclose(fft.FFT(np.ones(2**10)), np.fft.fft(np.ones(2**10)), rtol=0, atol=1e-10)

    def testAlternatingBig(self):
        x = np.array([(-1)**k for k in range((2**10))])
        np.testing.assert_allclose(fft.FFT(x), np.fft.fft(x), rtol=0, atol=1e-10)

    def testExpBig(self):
        e = np.array([cmath.exp(8j*2*cmath.pi*k/(2**10)) for k in range((2**10))])
        np.testing.assert_allclose(fft.FFT(e), np.fft.fft(e), rtol=0, atol=1e-10)

#testing fails here...fixing
    def testCatastrophicAnnihilationFFT(self):
	y=[2**52,1,-2**52,1,2**52,1,-2**52,1]
	np.testing.assert_allclose(fft.FFT(y), np.fft.fft(y))


class TestDFT(unittest.TestCase):
    def testAllZeros(self):
        np.testing.assert_allclose(np.array([0]*8), fft.DFT(np.array([0]*8)), rtol=0, atol=1e-10)

    def testAllOnes(self):
        np.testing.assert_allclose(np.array([8]+[0]*7), fft.DFT(np.array([1]*8)), rtol=0, atol=1e-10)

    def testAlternating(self):
        np.testing.assert_allclose(np.array([0]*4+[8]+[0]*3), fft.DFT(np.array([(-1)**k for k in range(8)])), rtol=0, atol=1e-10)

    def testExp(self):
        e = np.array([cmath.exp(2j*2*cmath.pi*k/(8)) for k in range(8)])
        np.testing.assert_allclose(np.array([0]*2+[8]+[0]*5), fft.DFT(e), rtol=0, atol=1e-10)
    
    def testCatastrophicAnnihilationDFT(self):
	   y=np.array([2**55,1,-2**55])
	   np.testing.assert_allclose(np.fft.fft(y), fft.DFT(y))

    
suiteFFT = unittest.TestLoader().loadTestsFromTestCase(TestFFT)
suiteDFT = unittest.TestLoader().loadTestsFromTestCase(TestDFT)
allTests = unittest.TestSuite([suiteFFT,suiteDFT])

unittest.TextTestRunner(verbosity=2).run(allTests)
