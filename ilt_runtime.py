import matplotlib.pyplot as plt
import time
import ilt
import math
import numpy as np

max_case=15
runtime=np.empty(max_case)

for i in range(max_case):
	delta = .1
	M = 2**i
	fhat = lambda s: 1./(s+0.5)
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
	t=time.clock()
	ilt.ILT(fhat_mat, delta, M)
	runtime[i]=time.clock() - t

x_len =np.array( [2**(i+1) for i in range(max_case)])
log_runtime = [math.log(x, 2) for x in runtime]

fig=plt.figure()
plt.scatter(x_len, runtime)
fig.suptitle('ILT Actual runtime')
plt.xlabel('number of data points')
plt.ylabel('actual runtime, secs')
fig.savefig('ilt_runtime_plot.png')

fig=plt.figure()
plt.scatter(x_len, log_runtime)
fig.suptitle('ILT Log runtime')
plt.xlabel('number of data points')
plt.ylabel('log runtime')
fig.savefig('ilt_log_runtime_plot.png')
