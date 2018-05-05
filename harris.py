import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import LucasKanade as lk

path = os.path.dirname(os.path.realpath(__file__))

# wx, wy are corner coordinates of the window
def cornerDetector(img1, img2, wx, wy, window_size, kernel_size):
    assert img1.shape == img2.shape
    height, width = img1.shape

    mid_kernel = kernel_size // 2

    '''
    Ix = np.zeros(shape = (height, width), dtype = 'int16')
    Iy = np.zeros(shape = (height, width), dtype = 'int16')
    
    for i in range (0, height - 1):
        for j in range (0, width - 1):
            Ix[i][j] = img1[i][j + 1] - img1[i][j]
            Iy[i][j] = img1[i + 1][j] - img1[i][j]

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    '''
    #create numpy array
    Ix = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Iy = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    It = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Ixx = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Ixy = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Iyy = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Itx = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Ity = np.zeros(shape = (window_size, window_size), dtype = 'int16')

    E = np.zeros(shape = (window_size, window_size))
    R = np.zeros(shape = (window_size, window_size))
    
    #create a dictionary
    pointlist = list()

    # loop over window with kernel and calculate the E value for each window
    lboundx = wx + mid_kernel
    uboundx = wx + window_size - mid_kernel
    lboundy = wy + mid_kernel
    uboundy = wy + window_size - mid_kernel
    for x in range(lboundx, uboundx):
        for y in range(lboundy, uboundy):

            #store the Ixx, Ixy, Iyy, Itx, Ity for the Lucas Algorithm
            Ix[x - wx][y - wy] = img1[x][y + 1] - img1[x][y]
            Iy[x - wx][y - wy] = img1[x + 1][y] - img1[x][y]
            It[x - wx][y - wy] = img1[x][y] - img2[x][y]

            Ixx[x - wx][y - wy] = Ix[x - wx][y - wy] * Ix[x - wx][y - wy]
            Ixy[x - wx][y - wy] = Ix[x - wx][y - wy] * Iy[x - wx][y - wy]
            Iyy[x - wx][y - wy] = Iy[x - wx][y - wy] * Iy[x - wx][y - wy]
            Itx[x - wx][y - wy] = It[x - wx][y - wy] * Ix[x - wx][y - wy]
            Ity[x - wx][y - wy] = It[x - wx][y - wy] * Iy[x - wx][y - wy]

            # calculate matrix M
            M = np.zeros(shape = (2, 2))
            for i in range(-mid_kernel, mid_kernel):
                for j in range(-mid_kernel, mid_kernel):
                    #M[0][0] += Ixx[x + i][y + j]
                    #M[0][1] += Ixy[x + i][y + j]
                    #M[1][0] += Ixy[x + i][y + j]
                    #M[1][1] += Iyy[x + i][y + j]
                    M[0][0] += Ixx[x - wx + i][y - wy + j]
                    M[0][1] += Ixy[x - wx + i][y - wy + j]
                    M[1][0] += Ixy[x - wx + i][y - wy + j]
                    M[1][1] += Iyy[x - wx + i][y - wy + j]                 
                    #M[0][0] += (img1[x + i][y + j + 1] - img1[x + i][y + j]) ** 2
                    #M[0][1] += (img1[x + i][y + j + 1] - img1[x + i][y + j]) * (img1[x + i + 1][y + j] - img1[x + i][y + j])
                    #M[1][0] += (img1[x + i][y + j + 1] - img1[x + i][y + j]) * (img1[x + i + 1][y + j] - img1[x + i][y + j])
                    #M[1][1] += (img1[x + i + 1][y + j] - img1[x + i][y + j]) ** 2
            
            # use M to calculate E and R for a few u, v values
            for u in range(x - mid_kernel, x + mid_kernel):
                for v in range(y - mid_kernel, y + mid_kernel):
                    E[x - wx][y - wy] += np.matmul(np.matmul(np.array([[u, v]]), M), np.array([[u], [v]]))
            
            R[x - wx][y - wy] = np.linalg.det(M) - 0.04 * (np.trace(M)) * (np.trace(M))

            if R[x - wx][y - wy] > 20:
                print(R[x - wx][y - wy], (x - wx, y - wy))
                #pointdict.update({(x - wx, y - wy): R[x - wx][y - wy]})
                pointlist.append((x - wx, y - wy))

            #store to the dictionary
            #maxpoints.update({E[x - wx][y - wy] : (x - wx, y - wy)})
    
    #sort and take top N values
    #fpoints = [value for (key, value) in sorted(maxpoints.items())][:N]

    '''
    # choose top N values for feature points 
    top_points = list()
    x = 0
    while x < window_size:
        y = 0
        while y < window_size:
            
            # loop over kernel
            vals = list()
            for i in range(-kernel_size // 2, kernel_size // 2):
                for j in range(-kernel_size // 2, kernel_size // 2):
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
    '''

    # display heatmap of feature points
    #plt.imshow(R);
    #plt.colorbar()
    #plt.show()

    return Ixx, Ixy, Iyy, Itx, Ity, pointlist


img1 = cv2.imread("ball-0.png", 0)
img2 = cv2.imread("ball-1.png", 0)
img3 = cv2.imread("ball-2.png", 0)
fig1 = cv2.GaussianBlur(img1, (5, 5), 0).astype('int16')
fig2 = cv2.GaussianBlur(img2, (5, 5), 0).astype('int16')
fig3 = cv2.GaussianBlur(img3, (5, 5), 0).astype('int16')
height, width = img1.shape
#pointlist = cornerDetector(fig1, fig2, 0, 0, height, 3)
Ixx, Ixy, Iyy, Itx, Ity, pointlist = cornerDetector(fig1, fig2, 0, 0, height, 3)
#Ixx, Ixy, Iyy, Itx, Ity, pointlist = cornerDetector(fig2, fig3, 0, 0, height, 3)
vlist = lk.LucasKanade(Ixx, Ixy, Iyy, Itx, Ity, pointlist, 3)
print(vlist)


img = cv2.imread("ball-0.png")
#plist = [(272, 78), (273, 79), (273, 80), (274, 81), (275, 82), (276, 83), (279, 86)]
#plist1 = [(234, 136), (238, 136), (243, 132), (248, 129), (247, 130), (237, 136), (244, 132), (239, 135), (243, 133), (244, 133)]
for i in range (len(pointlist)):
    #cv2.circle(img, (pointlist[i][1], pointlist[i][0]), 1, (0, 0, 255), -1)
    cv2.line(img, (pointlist[i][1], pointlist[i][0]), (pointlist[i][1] + vlist[i][1], pointlist[i][0] + vlist[i][0]), (0, 0, 255))
cv2.imwrite('test.jpg', img)

#0.04 30
#lkoutput = [array([[17.02181562], [11.70021112]]), array([[ 5.23140496], [-7.66942149]]), array([[-180.], [ 480.]])]
#1st harris = 422.4399999999986 (136, 30) 50.440000000000026 (140, 42) 12.159999999999982 (143, 24)
#2nd harris = 35.999999999999716 (112, 48) 39.36000000000004 (118, 41)


