#FFT takes in an input X that we assume to have length equal to a power of 2

import cmath

def FFT(X):
	N = len(X)
	if N==1:
		return X
	evens = FFT(X[0::2])
	odds = FFT(X[1::2])
	Y = range(N)
	w = cmath.exp(-2*cmath.pi*1j/N)
	Twiddle = 1
	# Twiddle should equal cmath.exp(-2*cmath.pi*k*1j/N) in run k
	for k in range(N/2):
		Y[k]=evens[k] + Twiddle * odds[k]
		Y[k+N/2] = evens[k] - Twiddle * odds[k]
		Twiddle *= w
	return Y


def DFT(X):
	#This is a naive implementation of the DFT for testing
	Y = []
	N = len(X)
	for k in range(N):
		YNew = 0
		for n in range(N):
			YNew += X[n]*cmath.exp(-2*cmath.pi*1j*k*n/N)
		Y+=[YNew]
	return Y
