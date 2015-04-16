import math
import numpy

# Script for generating a corrupted signal
N = 2**6
x = numpy.array([(4*math.cos(math.pi*i) + 3*math.sin((1./2)*math.pi*i + (1./3)*math.pi) + 6*math.cos((1./3)*math.pi*i + (1./5)*math.pi) + math.sin((1./6)*math.pi*i + 0.5*math.pi)) for i in range(N)])
x = x + numpy.random.normal(0,1,N); # Corrupt signal

# Write test data to a file
f = open('testdata.txt','w')
f.write(str(x))
f.close()