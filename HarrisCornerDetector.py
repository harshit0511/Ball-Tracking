import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import heapq

path = os.path.dirname(os.path.realpath(__file__))

# wx, wy are corner coordinates of the window
def cornerDetector(img, wx, wy, window_size, kernel_size, N):
    fig = img
    height, width = img.shape

    Ix = np.zeros(shape = (height, width), dtype = 'int16')
    Iy = np.zeros(shape = (height, width), dtype = 'int16')

    for i in range (wx, window_size - 1):
        for j in range (wy, window_size - 1):
            Ix[i][j] = fig[i][j + 1] - fig[i][j]
            Iy[i][j] = fig[i + 1][j] - fig[i][j]

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    E = np.zeros(shape = (window_size, window_size))
    
    # loop over window with kernel and calculate the E value for each window
    lboundx = wx + kernel_size / 2
    uboundx = wx + window_size - kernel_size / 2
    lboundy = wy + kernel_size / 2
    uboundy = wy + window_size - kernel_size / 2
    for x in range(lboundx, uboundx):
        for y in range(lboundy, uboundy):
            
            # calculate matrix M
            M = np.zeros(shape = (2, 2))
            for i in range(-kernel_size / 2, kernel_size / 2):
                for j in range(-kernel_size / 2, kernel_size / 2):
                    M[0][0] += Ixx[x + i][y + j]
                    M[0][1] += Ixy[x + i][y + j]
                    M[1][0] += Ixy[x + i][y + j]
                    M[1][1] += Iyy[x + i][y + j]
            
            # use M to calculate E for a few u, v values
            for u in range(x - kernel_size / 2, x + kernel_size / 2):
                for v in range(y - kernel_size / 2, y + kernel_size / 2):
                    E[x - wx][y - wy] += np.matmul(np.matmul(np.array([[u, v]]), M), np.array([[u], [v]]))
    
    # choose top N values for feature points 
    top_points = list()
    x = 0
    while x < window_size:
        y = 0
        while y < window_size:
            
            # loop over kernel
            vals = list()
            for i in range(-kernel_size / 2, kernel_size / 2):
                for j in range(-kernel_size / 2, kernel_size / 2):
                    vals.append((E[x + i][y + j], (x + i, y + j)))
            
            # push the larget value from vals
            heapq.heappush(top_points, max(vals))
            
            # if length > N, remove the lowest value
            if len(top_points) > N:
                heapq.heappop(top_points)
                
            y += kernel_size
        x += kernel_size
            

    
    # grab points from heap
    points = list()
    for i in range(len(top_points)):
        points.append(top_points[i][1])
    
    # display heatmap of feature points
    plt.imshow(E);
    plt.colorbar()
    plt.show()

    return points

img = cv2.imread("ball/ball-1.png", 0)
fig = cv2.GaussianBlur(img, (5, 5), 0).astype('int16')
height, width = img.shape
print(cornerDetector(fig, 0, 0, height, 3, 10))