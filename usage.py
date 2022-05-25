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


# 频域平滑
def ideal_low_filter(img, D0):
    """
    生成一个理想低通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.ones((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 0 if d > D0 else 1
    return filter_img


def butterworth_low_filter(img, D0, rank):
    """
        生成一个Butterworth低通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 1 / (1 + 0.414 * (d / D0) ** (2 * rank))
    return filter_img


def exp_low_filter(img, D0, rank):
    """
        生成一个指数低通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = np.exp(np.log(1 / np.sqrt(2)) * (d / D0) ** (2 * rank))
    return filter_img


def filter_use(img, filter):
    """
    将图像img与滤波器filter结合，生成对应的滤波图像
    """
    # 首先进行傅里叶变换
    f = np.fft.fft2(img)
    f_center = np.fft.fftshift(f)
    # 应用滤波器进行反变换
    S = np.multiply(f_center, filter)  # 频率相乘——l(u,v)*H(u,v)
    f_origin = np.fft.ifftshift(S)  # 将低频移动到原来的位置
    f_origin = np.fft.ifft2(f_origin)  # 使用ifft2进行傅里叶的逆变换
    f_origin = np.abs(f_origin)  # 设置区间
    return f_origin


def DFT_show(img):
    """
    对传入的图像进行傅里叶变换，生成频域图像
    """
    f = np.fft.fft2(img)  # 使用numpy进行傅里叶变换
    fshift = np.fft.fftshift(f)  # 把零频率分量移到中间
    result = np.log(1 + abs(fshift))
    return result


# 频域锐化
def ideal_high_filter(img, D0):
    """
    生成一个理想高通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 0 if d < D0 else 1
    return filter_img


def butterworth_high_filter(img, D0, rank):
    """
        生成一个Butterworth高通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 1 / (1 + (D0 / d) ** (2 * rank))
    return filter_img


def exp_high_filter(img, D0, rank):
    """
        生成一个指数高通滤波器（并返回）
    """
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = np.exp((-1) * (D0 / d) ** rank)
    return filter_img


def filter_use2(img, filter):
    """
    将图像img与滤波器filter结合，生成对应的滤波图像
    """
    # 首先进行傅里叶变换
    f = np.fft.fft2(img)
    f_center = np.fft.fftshift(f)
    # 应用滤波器进行反变换
    S = np.multiply(f_center, filter)  # 频率相乘——l(u,v)*H(u,v)
    f_origin = np.fft.ifftshift(S)  # 将低频移动到原来的位置
    f_origin = np.fft.ifft2(f_origin)  # 使用ifft2进行傅里叶的逆变换
    f_origin = np.abs(f_origin)  # 设置区间
    f_origin = f_origin / np.max(f_origin.all())
    return f_origin
