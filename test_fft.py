## Unit tests for FFT!

import fft
import unittest
import cmath

class TestFFT(unittest.TestCase):
     def testAllZeros(self):
	  self.assertEqual(fft.FFT([0]*16), fft.DFT([0]*16))


#### The following fail, I think because there is no "assertAlmostEqual" for vectors?

#     def testAllOnes(self):
#          self.assertEqual(fft.FFT([1]*16), fft.DFT([1]*16))

#     def testAlternating(self):
#          x = [(-1)**k for k in range(16)]
#          self.assertEqual(fft.FFT(x), fft.DFT(x))

#     def testExp(self):
#          e = [cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)]
#          self.assertEqual(fft.FFT(e), fft.DFT(e))




def suite():
     suite = unittest.TestSuite()
     suite.addTest(TestFFT('testAllZeros'))
#     suite.addTest(TestFFT('testAllOnes'))
#     suite.addTest(TestFFT('testAlternating'))
#     suite.addTest(TestFFT('testExp'))
     return suite

unittest.TextTestRunner(verbosity=2).run(suite())
