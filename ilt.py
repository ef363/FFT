# Inverse Laplace Transform!

# Inputs: fhat - output of laplace transform on function f (fhat is a function?)
#	  delta - step size
#	  M - a power of 2, length of vector
#	  lambda - ?
#	  beta - ?
# 	  "backwards" FFT (I'm hoping that means inverse FFT)
# Output: f(i*delta), i = 0, 1, ..., M-1
import numpy as np

def ILT(fhat, delta, M):
	# These are their magic numbers
	M2 = 8*M
	a = 44/M2
	n = 16

	# STEP 1
	fhat_mat = np.zeros(n/2,M2+1) 
	fhat_vect = np.zeros(M2+1) 
	beta_sum0 = 0
	# THIS DOESN'T WORK - don't know what lambda or beta are
	for k in range(M2+1):
		beta_sum = 0
		for j in range(n/2+1)[1:]:
			# This next line gives an error. Trying to extract real part.
			fhat_mat(j,k)=fhat((a+1j*lambda(j)+2*1j*np.pi*k/M2)/delta).real
			beta_sum += beta(j)*fhat_mat(j,k)
		fhat_vect(k) = (2/delta)*beta_sum
		if k=0: beta_sum0 = beta_sum
	# at this point, beta_sum is sum with k=M2
	fhat_vect(0) = (beta_sum + beta_sum0)/delta

	# STEP 2 ## THIS PART IS SUPPOSED TO BE SAME AS BACKWARDS FFT!!!!!
	fl = np.zeros(M2)
	for l in range(M2):
		sum  = 0
		for k in range(M2):
			sum += fhat_vect(k)*np.cos(2*np.pi*l*k/M2)
		fl(l) = sum/M2

	# STEP 3
	f = np.zeros(M)
	for l in range(M):
		f(l) = np.exp(a*l)*fl(l) # f(l) here is supposed to be f(l*delta) in the function

	return(f)
# So we just get f(l*delta) out of this, but we can find f(k) at other values of k using quadrature.
