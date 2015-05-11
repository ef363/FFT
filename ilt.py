# Inverse Laplace Transform!

# Inputs: fhat - output of laplace transform on function f (fhat is a lambda function)
#	  delta - step size
#	  M - a power of 2, length of vector
#	  lambda - sequence of numbers defined in paper
#	  beta - same thing as above
# 	  "backwards" (or inverse) FFT
# Output: f(i*delta), i = 0, 1, ..., M-1
import numpy as np
import fft

# Constants (taken from paper)
lamb = np.array([4.44089209850063e-016, 6.28318530717958, 12.5663706962589, 18.8502914166954, 25.2872172156717, 34.296971663526, 56.1725527716607, 170.533131190126])
n=16

# From a lambda function, make discrete matrix
def make_ILT_input(function, delta, M):
	M2 = 8*M
	a = 44./M2
	n = 16 # n must be even!
	fhat_mat = np.zeros((n/2, M2+1))
	for k in range(M2+1):
		for i in range(n/2):
			s = (a+1j*lamb[i]+2*1j*np.pi*k/M2)/delta
			fhat_mat[i,k] = np.real(function(s))
	return fhat_mat

# Performs ILT from the discretized values above (delta and M need to be the same as in above)
def ILT_from_discrete(fhat_mat, delta, M):
	M2 = 8*M
	a = 44./M2
	beta = np.array([1, 1.00000000000004, 1.00000015116847, 1.00081841700481, 1.09580332705189, 2.00687652338724, 5.94277512934943, 54.9537264520382])

	col_sum0 = 0
	fhat_vect = np.zeros(M2+1)
	for k in range(M2+1):
		col_sum = 0
		for i in range(n/2):
			col_sum += beta[i]*fhat_mat[i,k]
		if k==0: col_sum0 = col_sum
		else: fhat_vect[k] = (2./delta)*col_sum
	fhat_vect[0] = (1./delta)*(col_sum + col_sum0)
	
	# This isn't working!
	#fl = np.real(fft.IFFT(fhat_vect))
	
	# this is the hard-coded version (instead of using IFFT)
	fl = np.zeros(M2)
	for l in range(M2):
		for k in range(M2):
			fl[l] += fhat_vect[k]*np.cos(2*np.pi*l*k/M2)
		fl[l] = fl[l]/M2	

	# STEP 3
	f = np.zeros(M)
	for l in range(M):
		# not sure why we're off by a factor of 2
		f[l] = 2*np.exp(a*l)*fl[l] # f[l] here is supposed to be f[l*delta] in the function

	return(f)


# Need lambda function as input. See above if have discrete inputs.
def ILT(fhat, delta, M):
	# These are their magic numbers
	M2 = 8*M
	a = 44./M2
	n = 16
#	lamb = np.array([4.44089209850063e-016, 6.28318530717958, 12.5663706962589, 18.8502914166954, 25.2872172156717, 34.296971663526, 56.1725527716607, 170.533131190126])
	beta = np.array([1, 1.00000000000004, 1.00000015116847, 1.00081841700481, 1.09580332705189, 2.00687652338724, 5.94277512934943, 54.9537264520382])

	# STEP 1
	fhat_mat = np.zeros((n/2,M2+1)) 
	fhat_vect = np.zeros(M2+1) 
	beta_sum0 = 0

	for k in range(M2+1):
		beta_sum = 0
		for i in range(n/2):
			# May have floating point division error in next line
			s = (a+1j*lamb[i]+2*1j*np.pi*k/M2)/delta
			fhat_mat[i,k] = np.real(fhat(s)) #np.real(1./s) # Hard coding fhat(s) = 1/s for now
			beta_sum += beta[i]*fhat_mat[i,k]
		fhat_vect[k] = (2./delta)*beta_sum
		if k==0: beta_sum0 = beta_sum
	# at this point, beta_sum is sum with k=M2
	fhat_vect[0] = (1./delta)*(beta_sum + beta_sum0)

	# STEP 2 ## This currently only works efficiently when fl has length equal to a power of two.

	#fl = np.real(fft.IFFT(fhat_vect))
	fl = np.zeros(M2)
	for l in range(M2):
		for k in range(M2):
			fl[l] += fhat_vect[k]*np.cos(2*np.pi*l*k/M2)
		fl[l] = fl[l]/M2	

	# STEP 3
	f = np.zeros(M)
	for l in range(M):
		# not sure why we're off by a factor of 2
		f[l] = 2*np.exp(a*l)*fl[l] # f[l] here is supposed to be f[l*delta] in the function

	return(f)
# So we just get f(l*delta) out of this, but we can find f(k) at other values of k using quadrature.
