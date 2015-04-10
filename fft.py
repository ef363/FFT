#FFT takes in an input X that we assume to have length equal to a power of 2

import cmath

def FFT(X):
	N = len(X)
	if N==1:
		return X
	evens = FFT(X[0::2])
	odds = FFT(X[1::2])
	Y = range(N)
	for k in range(N/2):
		Twiddle = cmath.exp(-2*cmath.pi*k*1j/N)
		Y[k]=evens[k] + Twiddle * odds[k]
		Y[k+N/2] = evens[k] - Twiddle * odds[k]
	return Y

def test():
	print "All zeroes ", FFT([0]*16)
	print "All ones ", FFT([1]*16)
	print "Alternate -1 and 1 ", FFT([(-1)**k for k in range(16)])
	print "Exp ", FFT([cmath.exp(8j*2*cmath.pi*k/16) for k in range(16)])


test()