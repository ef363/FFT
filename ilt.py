# Inverse Laplace Transform!

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
	fhat_mat = np.zeros((n/2, M2+1))
	for k in range(M2+1):
		for i in range(n/2):
			s = (a+1j*lamb[i]+2*1j*np.pi*k/M2)/delta
			fhat_mat[i,k] = np.real(function(s))
	return fhat_mat

# Performs ILT from the discretized values above (delta and M need to be the same as in above)
def ILT(fhat_mat, delta, M):
	# More constants, taken from paper, first two relating to mesh size
	M2 = 8*M
	a = 44./M2
	beta = np.array([1, 1.00000000000004, 1.00000015116847, 1.00081841700481, 1.09580332705189, 2.00687652338724, 5.94277512934943, 54.9537264520382])

	# STEP 1 ################
	fhat_vect = np.zeros(M2+1)
	for k in range(M2+1):
		col = fhat_mat[:,k]
		if k==0: col0 = col
		else: fhat_vect[k] = (2./delta)*np.dot(beta, col)
	fhat_vect[0] = (1./delta)*np.dot(beta, np.add(col0, col))

	# STEP 2 ################
	# This currently only works efficiently when fl has length equal to a power of two.
	fl = np.real(fft.IFFT(fhat_vect[:M2]))

	# STEP 3 ################
	f = np.zeros(M)
	for l in range(M):
		# not sure why we're off by a factor of 2
		f[l] = 2*np.exp(a*l)*fl[l] # f[l] here is supposed to be f[l*delta] in the function

	return(f)

