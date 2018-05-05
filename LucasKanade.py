import cv2
import numpy as np
import matplotlib.pyplot as plt

def LucasKanade (Ixx, Ixy, Iyy, Itx, Ity, point_array, kernel_size):
	assert Ixx.shape == Ixy.shape
	assert Ixx.shape == Iyy.shape
	assert Ixx.shape == Itx.shape
	assert Ixx.shape == Ity.shape
	height, width = Ixx.shape

	mid_kernel = kernel_size // 2

	A = np.zeros(shape = (2, 2), dtype = 'int16')
	B = np.zeros(shape = (2, 1), dtype = 'int16')
	vector_list = []

	#for each point in the point array, apply the Lucas Kanade algorithm
	for k in range (len(point_array)):
		#loop the kernel, set up the matrix
		axx = axy = ayy = btx = bty = 0
		for i in range (-mid_kernel, mid_kernel):
			for j in range (-mid_kernel, mid_kernel):
				if (point_array[k][0] + i >= 0 and point_array[k][0] + i < height) and (point_array[k][1] + j >= 0 and point_array[k][1] + j < width):
					axx += Ixx[point_array[k][0] + i][point_array[k][1] + j]
					axy += Ixy[point_array[k][0] + i][point_array[k][1] + j]
					ayy += Iyy[point_array[k][0] + i][point_array[k][1] + j]
					btx += Itx[point_array[k][0] + i][point_array[k][1] + j]
					bty += Ity[point_array[k][0] + i][point_array[k][1] + j]
				else:
					axx = axy = ayy = btx = bty = 0

		#put the value into matrix A and B
		A[0][0] = axx
		A[0][1] = axy
		A[1][0] = axy
		A[1][1] = ayy
		B[0][0] = -btx
		B[1][0] = -bty
		det = axx * ayy - axy * axy

		#solve the matrix and get a vector v
		temp = np.matmul(A, B)
		if det == 0:
			v = [[0.0], [0.0]]
		else:
			v = np.true_divide(temp, det)
		
		#stor the vector for each input point
		vector_list.append(v)

	return vector_list