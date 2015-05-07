## Unit tests for ILT!

import ilt
import unittest
import cmath
import numpy as np
#import scipy # Need scipy to test Bessel function

delta = 1
M = 16

class TestILT(unittest.TestCase):
    def testConstantZero(self):
        fhat = lambda s: 0
        fx = np.zeros(M)
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)
    
    # def testBesselZero(self):
    #     fhat = lambda s: (s^2+1)**(-0.5)
    #     fx = np.array([scipy.special.jv(0,x) for x in range(M)]) # Bessel function of order 0: J_0(x)
    #     np.testing.assert_allclose(fft.ILT(fhat, delta, M), fx, atol=1e-2)

    def testExpX(self):
        fhat = lambda s: 1./(s+0.5)
        fx = np.array([np.exp(-0.5*x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

    def testExpXSinX(self):
        fhat = lambda s: 1./((s+0.2)**2+1)
        fx = np.array([np.exp(-0.2*x)*np.sin(x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

    def testConstantOne(self):
        fhat = lambda s: 1./s
        fx = np.ones(M)
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

    def testX(self):
        fhat = lambda s: 1./(s**2)
        fx = np.array(range(M))
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

    def testXExpX(self):
        fhat = lambda s: (s+1)**(-2)
        fx = np.array([x*np.exp(-x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

    def testSinX(self):
        fhat = lambda s: 1./(s**2+1)
        fx = np.array([np.sin(x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

    def testXCosX(self):
        fhat = lambda s: (s**2-1)*((s**2+1)**(-2))
        fx = np.array([x*np.cos(x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT(fhat, delta, M), fx, atol=1e-2)

suiteILT = unittest.TestLoader().loadTestsFromTestCase(TestILT)
allTests = unittest.TestSuite([suiteILT])

unittest.TextTestRunner(verbosity=2).run(allTests)
