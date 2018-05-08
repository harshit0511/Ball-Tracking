import cv2
import numpy as np
import matplotlib.pyplot as plt

def LucasKanade (Ix, Iy, It, point_array, kernel_size):

	height, width = Ix.shape

	mid_kernel = kernel_size // 2

	S = np.zeros(shape = (9, 2), dtype = 'int16')
	t = np.zeros(shape = (9, 1), dtype = 'int16')
	vector_list = []

	#for each point in the point array, apply the Lucas Kanade algorithm
	for k in range (len(point_array)):
		for i in range (-mid_kernel, mid_kernel):
		      n = 0
		      for j in range (-mid_kernel, mid_kernel):
				if (0 == 0 or (point_array[k][0] + i >= 0 and point_array[k][0] + i < height) and (point_array[k][1] + j >= 0 and point_array[k][1] + j < width)):
				        
				    S[n][0] = Ix[point_array[k][0] + i][point_array[k][1] + j]
				    S[n][1] = Iy[point_array[k][0] + i][point_array[k][1] + j]
				    t[n][0] = -It[point_array[k][0] + i][point_array[k][1] + j]
				    
				    #axx += Ixx[point_array[k][0] + i][point_array[k][1] + j]
				    #axy += Ixy[point_array[k][0] + i][point_array[k][1] + j]
				    #ayy += Iyy[point_array[k][0] + i][point_array[k][1] + j]
				    #btx += Itx[point_array[k][0] + i][point_array[k][1] + j]
				    #bty += Ity[point_array[k][0] + i][point_array[k][1] + j]
				
				n += 1


		#solve the matrix and get a vector v
		v = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(S), S)), np.transpose(S)), t)
		
		#stor the vector for each input point
		vector_list.append(v)

	return vector_list