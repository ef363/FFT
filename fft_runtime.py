import matplotlib.pyplot as plt
import time
import fft
import math
import numpy as np

np.random.seed(17)

max_case=15

runtime=np.empty(max_case)

for i in range(max_case):
	X=np.random.uniform(-100, 100, 2**i+1)
	t=time.clock()
	fft.FFT(X)
	runtime[i]=time.clock() - t
	

x_len =np.array( [2**(i+1) for i in range(max_case)])
log_runtime = [math.log(x, 2) for x in runtime]


fig=plt.figure()
plt.scatter(x_len, runtime)
fig.suptitle('FFT Actual runtime')
plt.xlabel('number of data points')
plt.ylabel('actual runtime, secs')
fig.savefig('fft_runtime_plot.png')

fig=plt.figure()
plt.scatter(x_len, log_runtime)
fig.suptitle('FFT Log runtime')
plt.xlabel('number of data points')
plt.ylabel('log runtime')
fig.savefig('fft_log_runtime_plot.png')
