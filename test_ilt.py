## Unit tests for ILT!

import ilt
import unittest
import cmath
import numpy as np
#import scipy # Need scipy to test Bessel function

delta = 1
M = 16
M2 = 8*M
n = 16

class TestMakeInputs(unittest.TestCase):
	def testConstantZero(self):
		fhat = lambda s: 0
		fhat_mat = ilt.make_ILT_input(fhat, delta, M)
		np.testing.assert_allclose(fhat_mat, np.zeros((n/2,M2+1)))

	def testExpX(self):
		fhat = lambda s: 1./(s+0.5)
		fhat_mat = ilt.make_ILT_input(fhat, delta, 1)
		np.testing.assert_allclose(fhat_mat, np.array([[0.16666, 0.16385, 0.15597, 0.14439, 0.13080, 0.11668, 0.10308, 0.09059, 0.07949], [0.07949, 0.06979, 0.06142, 0.05423, 0.04806, 0.04278, 0.03824, 0.03432, 0.03094], [0.03094, 0.02800, 0.02543, 0.02319, 0.02122, 0.01947, 0.01793, 0.01656, 0.01533], [0.01533, 0.01423, 0.01324, 0.01235, 0.01154, 0.01081, 0.01014, 0.00954, 0.00898], [0.00888, 0.00838, 0.00792, 0.00749, 0.007107, 0.006745, 0.006410, 0.0060993, 0.005810], [0.004949, 0.004736, 0.004536, 0.004349, 0.004173, 0.004007, 0.003851, 0.003704, 0.003565], [0.00188, 0.001829, 0.001780, 0.001733, 0.001688, 0.001644, 0.001603, 0.001562, 0.001524], [0.00020606, 0.00020418, 0.00020232, 0.00020049, 0.00019868, 0.0001969, 0.00019514, 0.00019341, 0.00019169]]), atol=1e-5)


class TestILTDiscrete(unittest.TestCase):
    def testConstantZero(self):
        fhat = lambda s: 0
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.zeros(M)
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)
    
    def testExpX(self):
        fhat = lambda s: 1./(s+0.5)
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.array([np.exp(-0.5*x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)

    def testExpXSinX(self):
        fhat = lambda s: 1./((s+0.2)**2+1)
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.array([np.exp(-0.2*x)*np.sin(x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)

    def testConstantOne(self):
        fhat = lambda s: 1./s
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.ones(M)
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)

    def testX(self):
        fhat = lambda s: 1./(s**2)
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.array(range(M))
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)

    def testXExpX(self):
        fhat = lambda s: (s+1)**(-2)
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.array([x*np.exp(-x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)

    def testSinX(self):
        fhat = lambda s: 1./(s**2+1)
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.array([np.sin(x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)

    def testXCosX(self):
        fhat = lambda s: (s**2-1)*((s**2+1)**(-2))
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
        fx = np.array([x*np.cos(x) for x in range(M)])
        np.testing.assert_allclose(ilt.ILT_from_discrete(fhat_mat, delta, M), fx, atol=1e-2)






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
suiteILTDiscrete = unittest.TestLoader().loadTestsFromTestCase(TestILTDiscrete)
suiteMakeInputs = unittest.TestLoader().loadTestsFromTestCase(TestMakeInputs)

allTests = unittest.TestSuite([suiteILT, suiteILTDiscrete, suiteMakeInputs])

unittest.TextTestRunner(verbosity=2).run(allTests)
