import numpy as np

# Convolving Image - X with Filter - F
def convolve_np(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = (F_height - 1) / 2
    W = (F_width - 1) / 2

    out = np.zeros((X_height, X_width))

    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    w = F[H+k, W+l]
                    sum += (w * a)
            out[i,j] = sum

    return out

def sobel(img):
    height = img.shape[0]
    width = img.shape[1]

    # Sobel Operator
    Hx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

    Hy = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

    # Partial derivative in x-direction
    img_x = convolve_np(img, Hx)
    # Partial derivative in y-direction
    img_y = convolve_np(img, Hy)

    # Calculating vector normal
    img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

    img_out = (img_out / np.max(img_out)) * 255
    
    # Returning partial derivatives and gradient
    return img_x, img_y, img_out
