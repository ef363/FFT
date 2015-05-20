# Inverse Laplace Transform!

# Reference:
# Den Iseger, P. (2006), "Numerical transform inversion using Gaussian quadrature", Probability in the Engineering and Informational Sciences, 20, 1-44.

# Inputs: fhat - output of laplace transform on function f (fhat is a lambda function)
#	  delta - step size
#	  M - a power of 2, length of vector
#	  lambda - sequence of numbers defined in paper
#	  beta - same thing as above
# 	  Inverse FFT
# Output: f(i*delta), i = 0, 1, ..., M-1
import numpy as np
import fft

# Constants (taken from paper)
lamb = np.array([4.44089209850063e-016, 6.28318530717958, 12.5663706962589, 18.8502914166954, 25.2872172156717, 34.296971663526, 56.1725527716607, 170.533131190126])
n=16

# From a lambda function, make matrix of evaluated points
def make_ILT_input(function, delta, M):
	M2 = 8*M
	a = 44./M2
	
	#The term inside function(.) is the value of s	
	fhat_mat=np.array([[np.real(function((a+1j*lamb[i]+2*1j*np.pi*k/M2)/delta)) for k in range(M2+1)] for i  in range(n/2)])

	return fhat_mat

# Performs ILT from the discretized values above (delta and M need to be the same as in above)
def ILT(fhat_mat, delta, M):
	# More constants, taken from paper, first two relating to mesh size
	M2 = 8*M
	a = 44./M2
	beta = np.array([1, 1.00000000000004, 1.00000015116847, 1.00081841700481, 1.09580332705189, 2.00687652338724, 5.94277512934943, 54.9537264520382])

	# STEP 1 ################
		
	fhat_vect=np.array([(2./delta)*np.dot(beta, fhat_mat[:,k]) for k in range(1, M2+1)])
	fhat_vect=np.append(np.array([(1./delta)*np.dot(beta, np.add(fhat_mat[:,0], fhat_mat[:,M2]))]), fhat_vect)

		
	# STEP 2 ################
	# This currently only works efficiently when fl has length equal to a power of two.
	fl = np.real(fft.IFFT(fhat_vect[:M2]))


	# STEP 3 ################
	# The paper is off by a factor of 2, and we correct it here.
	f=np.array([2*np.exp(a*l)*fl[l] for l in range(M)])

	return(f)

