import ilt

delta = 1
M=16
	
for i in range (1000):
	fhat=lambda s:0
	fhat_mat = ilt.make_ILT_input(fhat, delta, M)
	ilt.ILT(fhat_mat, delta, M)
