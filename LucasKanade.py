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
    cWeight = 1.0 / 9.0
    for k in range (len(point_array)):
        #loop the kernel, set up the matrix
        axx = axy = ayy = btx = bty = 0
        for i in range (-mid_kernel, mid_kernel + 1):
            for j in range (-mid_kernel, mid_kernel + 1):
                    if i == 0 and j == 0:
                        w = cWeight
                    else:
                        w = (1.0 - cWeight) / (kernel_size * kernel_size - 1)
                    if (point_array[k][0] + i >= 0 and point_array[k][0] + i < height) and (point_array[k][1] + j >= 0 and point_array[k][1] + j < width):
                        axx += Ixx[point_array[k][0] + i][point_array[k][1] + j] * w
                        axy += Ixy[point_array[k][0] + i][point_array[k][1] + j] * w
                        ayy += Iyy[point_array[k][0] + i][point_array[k][1] + j] * w
                        btx += Itx[point_array[k][0] + i][point_array[k][1] + j] * w
                        bty += Ity[point_array[k][0] + i][point_array[k][1] + j] * w

        #put the value into matrix A and B
        A[0][0] = axx
        A[0][1] = axy
        A[1][0] = axy
        A[1][1] = ayy
        B[0][0] = -btx
        B[1][0] = -bty

        #print(A)
        #print(B)

        #det = axx * ayy - axy * axy

        #solve the matrix and get a vector v
        temp = np.matmul(np.linalg.inv(A), B)
        u = 3 * temp[0][0]
        v = 3.5 * temp[1][0]
        vector_list.append((u, v))
        #if det == 0:
        #	v = [[0.0], [0.0]]
        #else:
        #	v = np.true_divide(temp, det)

        #stor the vector for each input point


    return vector_list