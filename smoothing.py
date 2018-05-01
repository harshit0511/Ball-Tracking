import numpy as np

def smoothing(img, k):

    height = img.shape[0]
    width = img.shape[1]
    div = k + k +  1

    for i in np.arange(k, height-k):
        for j in np.arange(k, width-k):
            sum = 0
            for x in np.arange(-k, k+1):
                for y in np.arange(-k, k+1):
                    a = img.item(i+x, j+y)
                    sum = sum + a
            b = int(sum / (div*div))
            img.itemset((i,j), b)

    return img
