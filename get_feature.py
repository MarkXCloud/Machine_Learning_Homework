import cv2
import numpy as np
from skimage.feature import greycomatrix
from PCA import PCA

def get_HOG(img):
    winSize = (32, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 4


    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    winStride = (8, 8)
    padding = (16, 16)
    test_hog = hog.compute(img, winStride, padding).reshape((-1,))
    return test_hog

if __name__ == '__main__':
    path = './MSRA10K_Imgs_GT/Imgs/101.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img,(27,27),interpolation=cv2.INTER_CUBIC)
    test_hog = get_HOG(img)
    print(test_hog.shape)
    cv2.waitKey(0)