import cv2
import numpy as np
import matplotlib.pyplot as plt

def opticalflowLK(img1, img2, kernel_size):
    fig1 = img1
    fig2 = img2
    assert img1.shape == img2.shape
    height, width = img1.shape
    x_range = width // kernel_size
    y_range = height // kernel_size

    It = np.zeros(shape = (height, width), dtype = 'int16')
    Ix = np.zeros(shape = (height, width), dtype = 'int16')
    Iy = np.zeros(shape = (height, width), dtype = 'int16')

    for i in range (0, height - 1):
        for j in range (0, width - 1):
            It[i][j] = fig1[i][j] - fig2[i][j]
            Ix[i][j] = fig1[i][j + 1] - fig1[i][j]
            Iy[i][j] = fig1[i + 1][j] - fig1[i][j]

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Itx = It * Ix
    Ity = It * Iy

    #of = [[(0.0, 0.0)] * width for i in range(height)]
    ofu = np.zeros(shape = (y_range, x_range), dtype = 'float64')
    ofv = np.zeros(shape = (y_range, x_range), dtype = 'float64')
    ofm = np.zeros(shape = (y_range, x_range), dtype = 'float64')
    A = np.zeros(shape = (2, 2), dtype = 'int16')
    B = np.zeros(shape = (2, 1), dtype = 'int16')

    for i in range (y_range):
        for j in range (x_range):
            axx = 0
            axy = 0
            ayy = 0
            btx = 0
            bty = 0
            for y in range (kernel_size):
                for x in range (kernel_size):
                    axx += Ixx[i * kernel_size + y][j * kernel_size + x]
                    axy += Ixy[i * kernel_size + y][j * kernel_size + x]
                    ayy += Iyy[i * kernel_size + y][j * kernel_size + x]
                    btx += Itx[i * kernel_size + y][j * kernel_size + x]
                    bty += Ity[i * kernel_size + y][j * kernel_size + x]
            '''
            A[0][0] = axx
            A[0][1] = axy
            A[1][0] = axy
            A[1][1] = ayy
            B[0][0] = - btx
            B[1][0] = - bty
            try:
                v = np.linalg.solve(A, B)
            except(Exception):
                v = [[0.0], [0.0]]
            '''

            A[0][0] = ayy
            A[0][1] = -axy
            A[1][0] = -axy
            A[1][1] = axx
            B[0][0] = -btx
            B[1][0] = -bty
            det = axx * ayy - axy * axy
            temp = np.matmul(A, B)
            if det == 0:
                v = [[0.0], [0.0]]
            else:
                v = np.true_divide(temp, det)

            ofu[i][j] = v[0][0]
            ofv[i][j] = v[1][0]
            ofm[i][j] = np.sqrt(v[0][0] ** 2 + v[1][0] ** 2)
         

    x = np.linspace(0, x_range, x_range)
    y = np.linspace(y_range, 0 , y_range)
    plt.figure(figsize = (12, 8))
    plt.quiver(x, y, ofu, ofv, scale = 100)
    plt.show()
    return ofu, ofv, ofm