# Inverse Laplace Transform!

# Inputs: fhat - output of laplace transform on function f (that is a function?)
#	  delta - step size
#	  M - a power of 2, length of vector
#	  lambda - sequence of numbers defined in paper
#	  beta - same thing as above
# 	  "backwards" FFT (I'm hoping that means inverse FFT)
# Output: f(i*delta), i = 0, 1, ..., M-1
import numpy as np
import fft

def ILT(fhat, delta, M): # Right now, the function is hardcoded (not really an input)
	# These are their magic numbers
	M2 = 8*M
	a = 44./M2
	n = 16
	lamb = np.array([4.44089209850063e-016, 6.28318530717958, 12.5663706962589, 18.8502914166954, 25.2872172156717, 34.296971663526, 56.1725527716607, 170.533131190126])
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
			fhat_mat[i,k] = np.real(1./s) # Hard coding fhat(s) = 1/s for now
			beta_sum += beta[i]*fhat_mat[i,k]
		fhat_vect[k] = (2./delta)*beta_sum
		if k==0: beta_sum0 = beta_sum
	# at this point, beta_sum is sum with k=M2
	fhat_vect[0] = (1./delta)*(beta_sum + beta_sum0)

	# STEP 2 ## This currently only works efficiently when fl has length equal to a power of two.
	fl = np.real(fft.IFFT(fhat_vect))

	# STEP 3
	f = np.zeros(M)
	for l in range(M):
		f[l] = np.exp(a*l)*fl[l] # f[l] here is supposed to be f[l*delta] in the function

	return(f)
# So we just get f(l*delta) out of this, but we can find f(k) at other values of k using quadrature.
