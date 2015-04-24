#FFT takes in an input X.  It will use the Cooley-Tukey algorithm to 
#compute the DFT of X.  When it enounters a list of odd length, it will
#use the naive DFT to compute its DFT.  For example, if L has length 24,
#it will only use the naive DFT for lists of length 3.

#In addition, if the length is <= 8, it uses the naive DFT since the 
#running time is better for small lengths.

##need to fix floating point

import cmath
import numpy as np

RECURSION_LIMIT = 8 # Cut-off point for using DFT (found empirically)

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

	#To account for catastrophic annihilation when input
	#is within [-2**55, 2**55].  To update later for other cases.
	m=2**26
		
	N = len(X)
	Y = np.empty(N, dtype = complex)
	# Passes through Y, defining each term
	for k in range(N):
	
		#Track the large numbers and small numbers separately
		#to avoid losing floating points
		sum = np.zeros(2, dtype=complex)

		# Passes through X
		w = cmath.exp(-2*cmath.pi*1j*k/N)
		Twiddle = 1
		# Twiddle should equal cmath.exp(-2*cmath.pi*kn*1j/N) in run n
		for n in range(N):

			val=X[n]*Twiddle
			if abs(val) >= m:
				sum[0]+=val
			else:
				sum[1]+=val
			
			
			#Update Twiddle for next iteration: check whether we can
			#simply make it +-1 or +-i and avoid floating point error  
			if (k*(n+1))%N ==0:
				Twiddle = 1
			elif (2*k*(n+1))%N ==0:
				Twiddle = -1
			elif (4*k*(n+1)-3*N)%(4*N)==0:
				Twiddle = 1j
			elif (4*k*(n+1)-N)%(4*N)==0:
				Twiddle= -1j
#			u = (4*k*n)%(4*N)
#			if u == N:
#				Twiddle = 1j
#			elif u == 2*N:
#				Twiddle = -1
#			elif u == 3*N:
#				Twiddle = -1j
#			elif u == 0:
#				Twiddle = 1
			else:	
				Twiddle *= w
#		print sum
		Y[k] = sum[0]+sum[1]
	return Y
