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
	powerOfTwo = 2**(int.bit_length(N-1))
	difference = powerOfTwo - N
	X=np.append(X, np.zeros(difference))
	return FFT_powerOfTwo(X)


# IFFT expects X to be a power of two (for it to use the fast FFT algorithm).
# N is the desired lenth (so it will remove excess digits)
def IFFT(X, N = -1):
	if N == -1:
		N = len(X)
	naive_IFFT = IFFT_powerOfTwo(X)
	return naive_IFFT[:N]

#   HELPER CODE
def FFT_powerOfTwo(X):
	N = len(X)
	if N <= RECURSION_LIMIT:
		return DFT(X)
	elif N%4 == 0: # Attempt split-radix FFT
		evens = FFT_powerOfTwo(X[0::2])
		odds_mod1 = FFT_powerOfTwo(X[1::4])
		odds_mod3 = FFT_powerOfTwo(X[3::4])
		Y = np.empty(N, dtype = complex)
		w_mod1 = cmath.exp(-2*cmath.pi*1j/N)
		Twiddle_mod1 = 1
		w_mod3 = cmath.exp(-6*cmath.pi*1j/N)
		Twiddle_mod3 = 1
		# Twiddle_mod1 should equal cmath.exp(-2*cmath.pi*k*1j/N) in run k
		# Twiddle_mod3 should equal cmath.exp(-2*cmath.pi*3*k*1j/N) in run k
		for k in range(N/4):
			Y[k] = evens[k] + Twiddle_mod1*odds_mod1[k] + Twiddle_mod3*odds_mod3[k]
			Y[k + N/2] = evens[k] - (Twiddle_mod1*odds_mod1[k] + Twiddle_mod3*odds_mod3[k])
			Y[k + N/4] = evens[k + N/4] - 1j*(Twiddle_mod1*odds_mod1[k] - Twiddle_mod3*odds_mod3[k])
			Y[k + 3*N/4] = evens[k + N/4] + 1j*(Twiddle_mod1*odds_mod1[k] - Twiddle_mod3*odds_mod3[k])
			Twiddle_mod1 *= w_mod1
			Twiddle_mod3 *= w_mod3
			#Twiddle_mod1 *= UpdateTwiddle(Twiddle_mod1,k,N,1)
			#Twiddle_mod3 *= UpdateTwiddle(Twiddle_mod3,k,N,3)
		return Y
	elif N%2 == 0: # Attempt 2-radix FFT
		evens = FFT_powerOfTwo(X[0::2])
		odds = FFT_powerOfTwo(X[1::2])
		Y = np.empty(N, dtype = complex)
		#w = cmath.exp(-2*cmath.pi*1j/N)
		Twiddle = 1
		# Twiddle should equal cmath.exp(-2*cmath.pi*k*1j/N) in run k
		for k in range(N/2):
			Y[k] = evens[k] + Twiddle * odds[k]
			Y[k + N/2] = evens[k] - Twiddle * odds[k]
			#Twiddle *= w
			Twiddle = UpdateTwiddle(Twiddle,k,N,1)
		return Y
	else:
		return DFT(X)
	


def DFT(X):
	# This is a naive implementation of the DFT for testing

	#To account for catastrophic annihilation when input
	#is within [-2**55, 2**55].  To update later for other cases.
	m=2**26
		
	N = len(X)
	Y = np.empty(N, dtype = complex)
	# Passes through Y, defining each term
	for k in range(N):
	
		#Track the large real/imaginary numbers and small 
		#real/imaginary numbers separately to avoid losing floating points
		big_real_sum = np.zeros(N, dtype=float)
		small_real_sum = np.zeros(N, dtype=float)
		big_imag_sum = np.zeros(N, dtype=float)
		small_imag_sum = np.zeros(N, dtype=float)

		#w = cmath.exp(-2*cmath.pi*1j*k/N)
		Twiddle = 1


		# Twiddle should equal cmath.exp(-2*cmath.pi*kn*1j/N) in run n
		for n in range(N):

			val = X[n]*Twiddle
			
			if abs(val.real) >= m:
				big_real_sum[n] = val.real
			else:
				small_real_sum[n] = val.real
			

			if abs(val.imag) >= m:
				big_imag_sum[n] = val.imag
			else:
				small_imag_sum[n] = val.imag
			
			#Update Twiddle for next iteration: check whether we can
			#simply make it +/-1 or +/-i and avoid floating point error  
			# u = (4*k*(n+1))%(4*N)
			# if u == 0:
			# 	Twiddle = 1
			# elif u == N:
			# 	Twiddle = -1j
			# elif u == 2*N:
			# 	Twiddle = -1
			# elif u == 3*N:
			# 	Twiddle = 1j
			# else:	
			# 	Twiddle *= w
			Twiddle = UpdateTwiddle(Twiddle,n,N,k)

		Y[k] = Kahan(big_real_sum) + Kahan(small_real_sum) + (Kahan(big_imag_sum) + Kahan(small_imag_sum))*1j
	return Y

def IFFT_powerOfTwo(Y):
	N = max(1, len(Y)) # length zero arrays should still be possible
	X_conj = (1./N)* FFT_powerOfTwo(np.conjugate(Y))
	X = np.conjugate(X_conj)
	return X

def UpdateTwiddle(Twiddle, index, N, multiplier):
	# Want to set Twiddle = cmath.exp(-2*cmath.pi*(index+1)*1j/N)
	# Note: cmath.exp(-1j*cmath.pi/2) = -1j

	w = cmath.exp(-2*cmath.pi*1j*multiplier/N)
	# Check cases for easy update of Twiddle to +/-1 or +/-1j
	check_mod = (4*multiplier*(index+1))%(4*N)
	if check_mod == 0: 
		Twiddle = 1 
	elif check_mod == N: 
		Twiddle = -1j 
	elif check_mod == 2*N: 
		Twiddle = -1 
	elif check_mod == 3*N: 
		Twiddle = 1j 
	else:	 
		Twiddle *= w
	return Twiddle


#Added Kahan to sum up terms in DFT in order to reduce floating point error
def Kahan(V):	
	s = 0
	c = 0
	for i in range(len(V)):
		y = V[i] - c
		t = s + y
		c = t - s - y
		s = t
	return s
