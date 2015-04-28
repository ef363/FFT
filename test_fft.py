## Unit tests for FFT!

import fft
import unittest
import cmath
import numpy as np

class TestFFT(unittest.TestCase):
    def testAllZerosSmall(self):
        np.testing.assert_allclose(fft.FFT(np.zeros(16)), np.fft.fft(np.zeros(16)), atol=1e-10)

    def testAllOnesSmall(self):
        np.testing.assert_allclose(fft.FFT(np.ones(16)), np.fft.fft(np.ones(16)), atol=1e-10)

    def testAlternatingSmall(self):
        x = np.array([(-1)**k for k in range(16)])
        np.testing.assert_allclose(fft.FFT(x), np.fft.fft(x),  atol=1e-10)

    def testExpSmall(self):
        e = np.array([cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)])
        np.testing.assert_allclose(fft.FFT(e), np.fft.fft(e),  atol=1e-10)

    def testAllZerosBig(self):
        np.testing.assert_allclose(fft.FFT(np.zeros(2**10)), np.fft.fft(np.zeros(2**10)), atol=1e-10)

    def testAllOnesBig(self):
        np.testing.assert_allclose(fft.FFT(np.ones(2**10)), np.fft.fft(np.ones(2**10)),  atol=1e-10)

    def testAlternatingBig(self):
        x = np.array([(-1)**k for k in range((2**10))])
        np.testing.assert_allclose(fft.FFT(x), np.fft.fft(x), atol=1e-10)

    def testExpBig(self):
        e = np.array([cmath.exp(8j*2*cmath.pi*k/(2**10)) for k in range((2**10))])
        np.testing.assert_allclose(fft.FFT(e), np.fft.fft(e), atol=1e-10)

    def testCatastrophicAnnihilationFFT(self):
	y=[2**52,1,-2**52,1,2**52,1,-2**52,1]
	np.testing.assert_allclose(fft.FFT(y), np.fft.fft(y), atol=1e-10)


class TestDFT(unittest.TestCase):
    def testAllZeros(self):
        np.testing.assert_allclose(np.array([0]*8), fft.DFT(np.array([0]*8)), atol=1e-10)

    def testAllOnes(self):
        np.testing.assert_allclose(np.array([8]+[0]*7), fft.DFT(np.array([1]*8)),  atol=1e-10)

    def testAlternating(self):
        np.testing.assert_allclose(np.array([0]*4+[8]+[0]*3), fft.DFT(np.array([(-1)**k for k in range(8)])),  atol=1e-10)

    def testExp(self):
        e = np.array([cmath.exp(2j*2*cmath.pi*k/(8)) for k in range(8)])
        np.testing.assert_allclose(np.array([0]*2+[8]+[0]*5), fft.DFT(e),  atol=1e-10)
    
    def testCatastrophicAnnihilationDFT(self):
	   y=np.array([2**55,1,-2**55])
	   np.testing.assert_allclose(np.fft.fft(y), fft.DFT(y), atol=1e-10)

class TestIFFT(unittest.TestCase):
    def testEmpty(self):
        self.assertEqual(fft.IFFT(np.array([])), np.array([]))

    def testSingle(self):
        self.assertEqual(fft.IFFT(np.array([1])), 1) 

    def testAllZerosSmall(self):
        np.testing.assert_allclose(fft.IFFT(np.zeros(16)), np.fft.ifft(np.zeros(16)), atol=1e-10)

    def testAllOnesSmall(self):
        np.testing.assert_allclose(fft.IFFT(np.ones(16)), np.fft.ifft(np.ones(16)), atol=1e-10)

    def testAlternatingSmall(self):
        x = np.array([(-1)**k for k in range(16)])
        np.testing.assert_allclose(fft.IFFT(x), np.fft.ifft(x),  atol=1e-10)

    def testExpSmall(self):
        e = np.array([cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)])
        np.testing.assert_allclose(fft.IFFT(e), np.fft.ifft(e),  atol=1e-10)

    def testAllZerosBig(self):
        np.testing.assert_allclose(fft.IFFT(np.zeros(2**10)), np.fft.ifft(np.zeros(2**10)), atol=1e-10)

    def testAllOnesBig(self):
        np.testing.assert_allclose(fft.IFFT(np.ones(2**10)), np.fft.ifft(np.ones(2**10)),  atol=1e-10)

    def testAlternatingBig(self):
        x = np.array([(-1)**k for k in range((2**10))])
        np.testing.assert_allclose(fft.IFFT(x), np.fft.ifft(x), atol=1e-10)

    def testExpBig(self):
        e = np.array([cmath.exp(8j*2*cmath.pi*k/(2**10)) for k in range((2**10))])
        np.testing.assert_allclose(fft.IFFT(e), np.fft.ifft(e), atol=1e-10)


class TestKahan(unittest.TestCase):
    def testEmpty(self):
	   self.assertEqual(fft.Kahan(np.array([])), 0)

    def testSingle(self):
	   self.assertEqual(fft.Kahan(np.array([cmath.pi])), cmath.pi) 

    def testLarge(self):
	   self.assertAlmostEqual(fft.Kahan(np.array([10000, cmath.pi, cmath.exp(1)])), 10005.8598744820488384738229) 


suiteFFT = unittest.TestLoader().loadTestsFromTestCase(TestFFT)
suiteDFT = unittest.TestLoader().loadTestsFromTestCase(TestDFT)
suiteIFFT = unittest.TestLoader().loadTestsFromTestCase(TestIFFT)
suiteKahan = unittest.TestLoader().loadTestsFromTestCase(TestKahan)
allTests = unittest.TestSuite([suiteFFT,suiteDFT, suiteIFFT, suiteKahan])

unittest.TextTestRunner(verbosity=2).run(allTests)
