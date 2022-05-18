import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

def grayHist(img, filename):
    plt.figure(filename, figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(img, 'gray')
    plt.subplot(122)
    h, w = img.shape[:2]
    pixelSequence = img.reshape(1, h * w)
    numberBins = 256

    histogram, bins, patch = plt.hist(img.ravel(), 256, [0, 255])

    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.savefig(filename)
    plt.show()