import fft 
import cmath 
import numpy as np

print 'fft divide and conquer'
a= fft.FFT([2**55, 1, -2**55])
print a

print 'dft'
print fft.DFT([2**55, 1, -2**55])

print 'actual'
b= np.fft.fft([2**55, 1, -2**55])
print b

print 'difference'
print (np.linalg.norm(a-b))

print 'fft good order'
print fft.FFT([2**55, -2**55, 1])

print 'dft good order'
print fft.DFT([2**55, -2**55, 1])

print 'actual good order'
print np.fft.fft([2**55, -2**55, 1])

