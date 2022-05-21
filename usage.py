import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from math import *


def swap(val1, val2):
    return val2, val1


def order_desc(array):
    # 列表的长度
    length = len(array)
    # 对列表进行选择排序，获得有序的列表
    for i in range(length):
        for j in range(i + 1, length):
            # 选择最大的值
            if array[j] > array[i]:
                # 交换位置
                temp = array[j]
                array[j] = array[i]
                array[i] = temp
    return array


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


def rotate(image, angle):
    height, width, channels = image.shape

    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = None
    if channels == 1:
        imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255))
    elif channels == 3:
        imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation
