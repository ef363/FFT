#FFT takes in an input X.  It will use the Cooley-Tukey algorithm to 
#compute the DFT of X.  When it enounters a list of odd length, it will
#use the naive DFT to compute its DFT.  For example, if L has length 24,
#it will only use the naive DFT for lists of length 3.

#In addition, if the length is <= 8, it uses the naive DFT since the 
#running time is better for small lengths.

##need to fix floating point

import cmath
import numpy as np

RECURSION_LIMIT = 1 # Cut-off point for using DFT (found empirically)

def FFT(X):
	N = len(X)
	if N%2 != 0 or N <= RECURSION_LIMIT:
		return DFT(X)
	evens = FFT(X[0::2])
	odds = FFT(X[1::2])
	Y = np.empty(N, dtype = complex)
	w = cmath.exp(-2*cmath.pi*1j/N)
	Twiddle = 1
	# Twiddle should equal cmath.exp(-2*cmath.pi*k*1j/N) in run k
	for k in range(N/2):
		Y[k] = evens[k] + Twiddle * odds[k]
		Y[k+N/2] = evens[k] - Twiddle * odds[k]
		Twiddle *= w
	return Y


def DFT(X):
	# This is a naive implementation of the DFT for testing
	N = len(X)
	Y = np.empty(N, dtype = complex)
	# Passes through Y, defining each term
	for k in range(N):
		sum = 0
		# Passes through X
		w = cmath.exp(-2*cmath.pi*1j*k/N)
		Twiddle = 1
		# Twiddle should equal cmath.exp(-2*cmath.pi*kn*1j/N) in run n
		for n in range(N):
			sum += X[n]*Twiddle
		
			Twiddle *= w
		Y[k] = sum
	return Y
