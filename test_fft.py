## Unit tests for FFT!

import fft
import unittest
import cmath
import numpy as np

class TestFFT(unittest.TestCase):
    def testAllZerosSmall(self):
        np.allclose(fft.FFT([0]*16), np.fft.fft([0]*16), rtol=0, atol=1e-10)

    def testAllOnesSmall(self):
        np.allclose(fft.FFT([1]*16), np.fft.fft([1]*16), rtol=0, atol=1e-10)

    def testAlternatingSmall(self):
        x = [(-1)**k for k in range(16)]
        np.allclose(fft.FFT(x), np.fft.fft(x), rtol=0, atol=1e-10)

    def testExpSmall(self):
        e = [cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)]
        np.allclose(fft.FFT(e), np.fft.fft(e), rtol=0, atol=1e-10)

    def testAllZerosBig(self):
        np.allclose(fft.FFT([0]*(2**10)), np.fft.fft([0]*(2**10)), rtol=0, atol=1e-10)

    def testAllOnesBig(self):
        np.allclose(fft.FFT([1]*(2**10)), np.fft.fft([1]*(2**10)), rtol=0, atol=1e-10)

    def testAlternatingBig(self):
        x = [(-1)**k for k in range((2**10))]
        np.allclose(fft.FFT(x), np.fft.fft(x), rtol=0, atol=1e-10)

    def testExpBig(self):
        e = [cmath.exp(8j*2*cmath.pi*k/(2**10)) for k in range((2**10))]
        np.allclose(fft.FFT(e), np.fft.fft(e), rtol=0, atol=1e-10)

    def testCatastrophicAnnihilationFFT(self):
	y=[2**55,1,-2**55,1,2**55,1,-2**55,1]
	np.allclose(fft.FFT(y), np.zeros(8), rtol=0, atol=1e-10)


class TestDFT(unittest.TestCase):
    def testAllZeros(self):
        np.allclose([0]*8, fft.DFT([0]*8), rtol=0, atol=1e-10)

    def testAllOnes(self):
        np.allclose([8]+[0]*7, fft.DFT([1]*8), rtol=0, atol=1e-10)

    def testAlternating(self):
        np.allclose([0]*4+[8]+[0]*3, fft.DFT([(-1)**k for k in range(8)]), rtol=0, atol=1e-10)

    def testExp(self):
        e = [cmath.exp(2j*2*cmath.pi*k/(8)) for k in range(8)]
        np.allclose([0]*2+[8]+[0]*5, fft.DFT(e), rtol=0, atol=1e-10)
    
    def testCatastrophicAnnihilationDFT(self):
	np.allclose(np.fft.fft([2**55, 1, -2**55]), fft.DFT([2**55, 1, -2**55]), rtol=0, atol=1e-10)

    
suiteFFT = unittest.TestLoader().loadTestsFromTestCase(TestFFT)
suiteDFT = unittest.TestLoader().loadTestsFromTestCase(TestDFT)
allTests = unittest.TestSuite([suiteFFT,suiteDFT])

unittest.TextTestRunner(verbosity=2).run(allTests)
