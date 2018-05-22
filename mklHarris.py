import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import LucasKanade as lk
#reload(lk)

path = os.path.dirname(os.path.realpath(__file__))

# wx, wy are corner coordinates of the window
def cornerDetector(img1, img2, wx, wy, window_size, kernel_size):
    assert img1.shape == img2.shape
    height, width = img1.shape
    #print wx, wy, window_size

    mid_kernel = kernel_size // 2

    #create numpy array
    Ix = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Iy = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    It = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Ixx = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Ixy = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Iyy = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Itx = np.zeros(shape = (window_size, window_size), dtype = 'int16')
    Ity = np.zeros(shape = (window_size, window_size), dtype = 'int16')

    R = np.zeros(shape = (window_size, window_size))
    
    #create a dictionary
    pointdict = dict()

    # loop over window with kernel and calculate the E value for each window
    lboundx = wx + mid_kernel
    uboundx = wx + window_size - mid_kernel
    lboundy = wy + mid_kernel
    uboundy = wy + window_size - mid_kernel
    for x in range(lboundx, uboundx):
        for y in range(lboundy, uboundy):
            
            #print x, y
            
            #store the Ixx, Ixy, Iyy, Itx, Ity for the Lucas Algorithm
            Ix[x - wx][y - wy] = img1[x][y + 1] - img1[x][y - 1]
            Iy[x - wx][y - wy] = img1[x + 1][y] - img1[x - 1][y]
            It[x - wx][y - wy] = img2[x][y] - img1[x][y]

            Ixx[x - wx][y - wy] = Ix[x - wx][y - wy] * Ix[x - wx][y - wy]
            Ixy[x - wx][y - wy] = Ix[x - wx][y - wy] * Iy[x - wx][y - wy]
            Iyy[x - wx][y - wy] = Iy[x - wx][y - wy] * Iy[x - wx][y - wy]
            Itx[x - wx][y - wy] = It[x - wx][y - wy] * Ix[x - wx][y - wy]
            Ity[x - wx][y - wy] = It[x - wx][y - wy] * Iy[x - wx][y - wy]
            
    for x in range(lboundx, uboundx):
        for y in range(lboundy, uboundy):
            # calculate matrix M
            M = np.zeros(shape = (2, 2))
            for i in range(-mid_kernel, mid_kernel + 1):
                for j in range(-mid_kernel, mid_kernel + 1):

                    M[0][0] += Ixx[x - wx + i][y - wy + j]
                    M[0][1] += Ixy[x - wx + i][y - wy + j]
                    M[1][0] += Ixy[x - wx + i][y - wy + j]
                    M[1][1] += Iyy[x - wx + i][y - wy + j]                 
            
            R[x - wx][y - wy] = np.linalg.det(M) - 0.04 * (np.trace(M)) * (np.trace(M))

            if R[x - wx][y - wy] > 50000000:
                pointdict.update({(x - wx, y - wy): R[x - wx][y - wy]})

    return Ixx, Ixy, Iyy, Itx, Ity, pointdict
    

def localmaxima(pointdict, kernel_size):
    #create a list for point delete
    delpoint = list()
    #create a pointlist
    pointlist = list(pointdict.keys())

    mid_kernel = kernel_size // 2
    
    #for each point in the point list, compare it to its neighbors, neighbor is the other points in the kernel
    for i in pointlist:
        #find the neighbors of the point
        for y in range (i[0]-mid_kernel, i[0]+mid_kernel+1):
            for x in range (i[1]-mid_kernel,i[1]+mid_kernel+1):
                k = (y, x)
                if k in pointdict and k != i:
                    if pointdict[i] <= pointdict[k]:
                        pointdict[i] = 0
                        #if there exist some point with larger value in the kernel, delete this point
                        delpoint.append(i)
    
    
    #if the point in the delete list, delete it
    for i in delpoint:
        if i in pointdict:
            del pointdict[i]

    #get the remain keys in the point dictionary
    #print(pointdict)
    resultlist = list(pointdict.keys())

    return resultlist

    #return Ixx, Ixy, Iyy, Itx, Ity, resultlist


def vector_field(name, n, window_x, window_y, t_height):
    img1 = cv2.imread("{}/{}-{}.png".format(name, name, n), 0)
    img2 = cv2.imread("{}/{}-{}.png".format(name, name, n+1), 0)
    #img3 = cv2.imread("{}/{}-2.png", 0)
    for i in range(0, 10):
        fig1 = cv2.GaussianBlur(img1, (5, 5), 0).astype('int16')
        fig2 = cv2.GaussianBlur(img2, (5, 5), 0).astype('int16')
    #fig3 = cv2.GaussianBlur(img3, (5, 5), 0).astype('int16')
    height, width = img1.shape
    #pointlist = cornerDetector(fig1, fig2, 0, 0, height, 3)
    Ixx, Ixy, Iyy, Itx, Ity, pointdict = cornerDetector(fig1, fig2, 0, 0, height, 3)
    #Ixx, Ixy, Iyy, Itx, Ity, pointlist = cornerDetector(fig2, fig3, 0, 0, height, 3)
    pointlist = localmaxima(pointdict, 3)
    vlist = lk.LucasKanade(Ixx, Ixy, Iyy, Itx, Ity, pointlist, 3)
    center_vector_list = getcentervector(pointlist, vlist)
        
    '''
    img = cv2.imread("{}/{}-{}.png".format(name, name, n))
    for i in range (len(pointlist)):
        #cv2.circle(img, (pointlist[i][1], pointlist[i][0]), 1, (0, 0, 255), -1)
        cv2.arrowedLine(img, 
        (int(pointlist[i][1]), 
        int(pointlist[i][0])), 
        (int(pointlist[i][1] + vlist[i][0][0]), 
        int(pointlist[i][0] + vlist[i][1][0])), 
        (0, 0, 255))
    #for i in pointlist:
    #    img[i[0],i[1]] = [0,0,255]
    '''
    
    #cv2.imwrite('{}/test-{}.png'.format(name, n), img)
    return center_vector_list


def drawarrow(name, n, center, avgvector):
    resultimg = cv2.imread("{}/{}-{}.png".format(name, name, n), 0)
    cv2.arrowedLine(resultimg, )

#0.04 30
#lkoutput = [array([[17.02181562], [11.70021112]]), array([[ 5.23140496], [-7.66942149]]), array([[-180.], [ 480.]])]
#1st harris = 422.4399999999986 (136, 30) 50.440000000000026 (140, 42) 12.159999999999982 (143, 24)
#2nd harris = 35.999999999999716 (112, 48) 39.36000000000004 (118, 41)


def getcentervector(pointlist, vlist):
    center_vector_list = list()
        
    #find the center of the square
    x = y = 0
    n = len(pointlist)
    for i in pointlist:
        x += i[1]
        y += i[0]
    xc = round(x / n)
    yc = round(y / n)
    center = (xc, yc)


    #calculate the average
    u = v = 0
    k = len(vlist)
    #vlist = np.array(vlist)
    #for i in range(len(vlist)):
        #u += vlist[i, 0]
        #v += vlist[i, 1]
        #print(u, v)
        
    for p in vlist:
        u += p[0]
        v += p[1]
    uc = round(u / k)
    vc = round(v / k)
    avgvector = (uc, vc)

    #store into the list
    center_vector_list.append((center, avgvector))

    return center_vector_list
    #print the result image



#center_vector_list = getcentervector("square2")
#print(center_vector_list)



