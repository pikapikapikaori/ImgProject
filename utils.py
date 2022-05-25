import os
import sys

import matplotlib.pyplot as plt
import cv2
import numpy as np

import usage


def img_repair():
    print("可选操作：\n0.返回上一步\n1.高斯噪声添加\n2.椒盐噪声添加\n3.均值滤波\n4.统计滤波\n5.频域滤波")

    try:
        repa_choi = int(input("请选择要进行的基本操作："))
    except ValueError:
        print("选择基本操作输入错误，请重新输入：")
        return

    if repa_choi == 0:

        return
    # 高斯噪声
    elif repa_choi == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        image = np.array(img / 255, dtype=float)

        noise = np.random.normal(0, 0.1, image.shape)

        out = image + noise
        out = np.clip(out, 0.0, 1.0)
        out = np.uint8(out * 255)
        cv2.imwrite("results/result.jpg", out)
        print("结果请查看根目录下的results文件夹")

        return
    # 椒盐噪声
    elif repa_choi == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)
        out = np.zeros(img.shape, np.uint8)

        try:
            ran_x1, ran_y1 = input("请输入胡椒噪声范围，无穷大则输入MAX（格式为\"x y\"）：").split()
            ran_x2, ran_y2 = input("请输入食盐噪声范围，无穷大则输入MAX（格式为\"x y\"）：").split()
        except ValueError:
            print("输入参数个数不足，请重新输入：")
            return

        try:
            if ran_y1 == "MAX":
                ran_y1 = sys.maxsize

            if ran_y2 == "MAX":
                ran_y2 = sys.maxsize

            ran_x1 = int(ran_x1)
            ran_y1 = int(ran_y1)
            ran_x2 = int(ran_x2)
            ran_y2 = int(ran_y2)
        except ValueError:
            print("范围值输入错误，请重新输入：")
            return

        if ran_x1 >= ran_y1:
            ran_x1, ran_y1 = usage.swap(ran_x1, ran_y1)

        if ran_x2 >= ran_y2:
            ran_x2, ran_y2 = usage.swap(ran_x2, ran_y2)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if ran_x1 < img[i][j] < ran_y1:
                    # 添加胡椒噪声
                    out[i][j] = 0
                elif ran_x2 < img[i][j] < ran_y2:
                    # 添加食盐噪声
                    out[i][j] = 255
                else:
                    # 不添加噪声
                    out[i][j] = img[i][j]

        cv2.imwrite("results/result.jpg", out)
        print("结果请查看根目录下的results文件夹")

        return
    # 均值滤波
    elif repa_choi == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 0)
        out = np.zeros(img.shape, np.uint8)

        print("可选滤波器：\n1.算术均值滤波\n2.几何均值滤波\n3.谐波均值滤波\n4.逆谐波均值滤波")

        try:
            filter_choi = int(input("请选择要使用的滤波器："))
        except ValueError:
            print("选择操作输入错误，请重新输入：")
            return

        antiharm_Q = 0

        if filter_choi == 4:
            try:
                antiharm_Q = int(input("请输入滤波和调整的阶数："))
            except ValueError:
                print("阶数输入错误，请重新输入：")
                return

        try:
            filter_size_p, filter_size_q = input("请输入滤波器大小p*q（奇数，格式为\"p q\"）：").split()
        except ValueError:
            print("输入参数个数不足，请重新输入：")
            return

        try:
            filter_size_p = int(filter_size_p)
            filter_size_q = int(filter_size_q)
        except ValueError:
            print("滤波器大小输入错误，请重新输入：")
            return

        if (filter_size_p <= 0) or (filter_size_q <= 0):
            print("滤波器大小输入错误，请重新输入：")
            return

        if ((filter_size_p % 2) == 0) or ((filter_size_q % 2) == 0):
            print("滤波器大小输入错误，请重新输入：")
            return

        p = int(filter_size_p / 2)
        q = int(filter_size_q / 2)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                if filter_choi == 1:
                    sum = 0

                    for m in range(-p, p + 1):
                        for n in range(-q, q + 1):
                            if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                                sum += img[i + m][j + n]

                    out[i][j] = int(sum / (filter_size_p * filter_size_q))
                elif filter_choi == 2:
                    mul = 1

                    for m in range(-p, p + 1):
                        for n in range(-q, q + 1):
                            if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                                mul *= img[i + m][j + n]

                    out[i][j] = int(pow(mul, 1 / (filter_size_p * filter_size_q)))
                elif filter_choi == 3:
                    harm = 0.0

                    for m in range(-p, p + 1):
                        for n in range(-q, q + 1):
                            if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                                harm += 1 / img[i + m][j + n]

                    out[i][j] = int((filter_size_p * filter_size_q) / harm)
                elif filter_choi == 4:
                    antiharm1 = 0
                    antiharm2 = 0

                    for m in range(-p, p + 1):
                        for n in range(-q, q + 1):
                            if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                                antiharm1 += pow(img[i + m][j + n], antiharm_Q + 1)
                                antiharm2 += pow(img[i + m][j + n], antiharm_Q)

                    out[i][j] = int(antiharm1 / antiharm2)
                else:
                    print("选择滤波器输入错误，请重新输入：")
                    return

        cv2.imwrite("results/result.jpg", out)
        print("结果请查看根目录下的results文件夹")

        return
    # 统计滤波
    elif repa_choi == 4:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 0)
        out = np.zeros(img.shape, np.uint8)

        print("可选滤波器：\n1.中值滤波\n2.最小值滤波\n3.最大值滤波")

        try:
            filter_choi = int(input("请选择要使用的滤波器："))
        except ValueError:
            print("选择操作输入错误，请重新输入：")
            return

        array = []

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                array.clear()
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                            array.append(img[i + m][j + n])
                res_array = usage.order_desc(array)
                res_array_leng = len(res_array)
                if filter_choi == 1:
                    out[i][j] = res_array[int(res_array_leng / 2)]
                elif filter_choi == 2:
                    out[i][j] = res_array[res_array_leng - 1]
                elif filter_choi == 3:
                    out[i][j] = res_array[0]
                else:
                    print("选择操作输入错误，请重新输入：")
                    return

        cv2.imwrite("results/result.jpg", out)
        print("结果请查看根目录下的results文件夹")

        return
    # 频域滤波
    elif repa_choi == 5:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 0)
        out = np.zeros(img.shape, np.uint8)

        try:
            min, max = input("请输入允许通过的像素点的像素值范围，最大值请输入MAX（格式为\"min max\"）：").split()
        except ValueError:
            print("输入参数个数不足，请重新输入：")
            return

        array = []

        try:
            if max == "MAX":
                max = sys.maxsize

            min = int(min)
            max = int(max)
        except ValueError:
            print("范围输入错误，请重新输入：")
            return

        if (min <= 0) or (max <= 0):
            print("范围输入错误，请重新输入：")
            return

        if min >= max:
            min, max = usage.swap(min, max)

        print("原来的像素值可设置的zhi：\n1.0\n2.255")

        try:
            color_choi = int(input("请选择要设置的颜色："))
        except ValueError:
            print("选择颜色输入错误，请重新输入：")
            return

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # 滤波器内像素值的和
                array.clear()
                if min < img[i][j] < max:
                    out[i][j] = img[i][j]
                else:
                    if color_choi == 1:
                        out[i][j] = 0
                    elif color_choi == 2:
                        out[i][j] = 255
                    else:
                        print("选择颜色输入错误，请重新输入：")
                        return

        cv2.imwrite("results/result.jpg", out)
        print("结果请查看根目录下的results文件夹")

        return

    else:
        print("选择操作输入错误，请重新输入：")


def img_segmentation():
    print("可选操作：\n0.返回上一步\n1.Roberts算子边缘检测\n2.Sobel算子边缘检测\n3.Laplacian算子边缘检测\n4.LoG算子边缘检测\n5.Canny算子边缘检测"
          "\n6.HoughLines算法线条变化检测\n7.HoughLinesP算法线条变化检测")

    try:
        seg_choi = int(input("请选择要进行的基本操作："))
    except ValueError:
        print("选择基本操作输入错误，请重新输入：")
        return

    if seg_choi == 0:

        return
    # Roberts
    elif seg_choi == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path

        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return

        img = cv2.imread(img_path)

        try:
            val1 = float(input("请输入处理后第一张图的权值（推荐设为0.5）："))
            val2 = float(input("请输入处理后第二张图的权值（推荐设为0.5）："))
        except ValueError:
            print("权值输入错误，请重新输入：")
            return

        if (val1 < 0) or (val2 < 0):
            print("权值输入错误，请重新输入：")
            return

        try:
            exp = float(input("请输入偏置值（推荐设为0）："))
        except ValueError:
            print("偏置值输入错误，请重新输入：")
            return

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)

        cal_x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        cal_y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(cal_x)
        absY = cv2.convertScaleAbs(cal_y)

        Roberts = cv2.addWeighted(absX, val1, absY, val2, exp)

        cv2.imwrite("results/result.jpg", Roberts)
        print("结果请查看根目录下的results文件夹")
    # Sobel
    elif seg_choi == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        try:
            val1 = float(input("请输入处理后第一张图的权值（推荐设为0.5）："))
            val2 = float(input("请输入处理后第二张图的权值（推荐设为0.5）："))
        except ValueError:
            print("权值输入错误，请重新输入：")
            return

        if (val1 < 0) or (val2 < 0):
            print("权值输入错误，请重新输入：")
            return

        try:
            exp = float(input("请输入偏置值（推荐设为0）："))
        except ValueError:
            print("偏置值输入错误，请重新输入：")
            return

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernelx = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
        kernely = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(kernelx)
        absY = cv2.convertScaleAbs(kernely)

        Sobel = cv2.addWeighted(absX, val1, absY, val2, exp)

        cv2.imwrite("results/result.jpg", Sobel)
        print("结果请查看根目录下的results文件夹")
    # Laplacian
    elif seg_choi == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        try:
            kernel_size = int(input("请输入高斯滤波的卷积核大小k（奇数，大小为k*k）（推荐设为5）："))
        except ValueError:
            print("卷积核大小输入错误，请重新输入：")
            return

        if kernel_size <= 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        if (kernel_size % 2) == 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        try:
            exp = float(input("请输入偏差值（推荐设为0）："))
        except ValueError:
            print("偏差值输入错误，请重新输入：")
            return

        try:
            k_size = int(input("请输入Laplacian算子的核大小（奇数，推荐设为3）："))
        except ValueError:
            print("核大小输入错误，请重新输入：")
            return

        if k_size <= 0:
            print("核大小输入错误，请重新输入：")
            return

        if (k_size % 2) == 0:
            print("核大小输入错误，请重新输入：")
            return

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grayImage, (kernel_size, kernel_size), exp)

        dst = cv2.Laplacian(blur, cv2.CV_16S, ksize=k_size)

        Laplacian = cv2.convertScaleAbs(dst)

        cv2.imwrite("results/result.jpg", Laplacian)
        print("结果请查看根目录下的results文件夹")
    # LoG
    elif seg_choi == 4:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
        image = cv2.GaussianBlur(image, (3, 3), 0, 0)

        print("默认LoG算子为：")
        print(" 0,  0, -1,  0,  0")
        print(" 0, -1, -2, -1,  0")
        print("-1, -2, 16, -2, -1")
        print(" 0, -1, -2, -1,  0")
        print(" 0,  0, -1,  0,  0")

        print("可选LoG算子：\n1手动输入\n2使用默认算子")

        try:
            cal_choi = int(input("请选择要选择的LoG算子："))
        except ValueError:
            print("选择算子输入错误，请重新输入：")
            return

        if cal_choi == 1:
            LoGMatr = [[] for i in range(5)]
            print("请输入LoG算子矩阵。每列用半角逗号隔开。每行之间用enter换行：")
            for i in range(5):
                LoGMatr[i] = [eval(x) for x in input().split(',')]
        elif cal_choi == 2:
            LoGMatr = [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]
        else:
            print("选择算子输入错误，请重新输入：")
            return

        img_m1 = np.array(LoGMatr)

        rows = image.shape[0]
        cols = image.shape[1]

        image1 = np.zeros(image.shape)

        for k in range(0, 2):
            for i in range(2, rows - 2):
                for j in range(2, cols - 2):
                    image1[i, j] = np.sum((img_m1 * image[i - 2:i + 3, j - 2:j + 3, k]))

        image1 = cv2.convertScaleAbs(image1)

        cv2.imwrite("results/result.jpg", image1)
        print("结果请查看根目录下的results文件夹")
    # Canny
    elif seg_choi == 5:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        try:
            kernel_size = int(input("请输入高斯滤波的卷积核大小k（奇数，大小为k*k）（推荐设为5）："))
        except ValueError:
            print("卷积核大小输入错误，请重新输入：")
            return

        if kernel_size <= 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        if (kernel_size % 2) == 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        try:
            exp = float(input("请输入偏差值（推荐设为0）："))
        except ValueError:
            print("偏差值输入错误，请重新输入：")
            return

        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), exp)

        grayImage = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        gradx = cv2.Sobel(grayImage, cv2.CV_16SC1, 1, 0)
        grady = cv2.Sobel(grayImage, cv2.CV_16SC1, 0, 1)

        edge_output = cv2.Canny(gradx, grady, 50, 150)

        cv2.imwrite("results/result.jpg", edge_output)
        print("结果请查看根目录下的results文件夹")
    # HoughLines
    elif seg_choi == 6:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 2, 118)

        result = img.copy()
        for i_line in lines:
            for line in i_line:
                rho = line[0]
                theta = line[1]
                if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                    pt1 = (int(rho / np.cos(theta)), 0)
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    cv2.line(result, pt1, pt2, (0, 0, 255))
                else:
                    pt1 = (0, int(rho / np.sin(theta)))
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    cv2.line(result, pt1, pt2, (0, 0, 255), 1)

        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # HoughLinesP
    elif seg_choi == 7:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        minLineLength = 200
        maxLineGap = 15

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

        result_P = img.copy()
        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imwrite("/results/result.jpg", result_P)
        print("结果请查看根目录下的results文件夹")

    else:
        print("选择操作输入错误，请重新输入：")


def histogram():
    print("可选操作：\n0.返回上一步\n1.灰度直方图\n2.彩色直方图\n3.分段线性处理")

    try:
        hist_choi = int(input("请选择要进行的基本操作："))
    except ValueError:
        print("输入错误，请重新输入：")
        return

    if hist_choi == 0:

        return
    # 灰度直方图
    elif hist_choi == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 0)

        plt.subplot(121)
        plt.imshow(img, "gray")
        plt.subplot(122)
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        print('最大像素值：', max(hist))
        print('最小像素值：', min(hist))
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.savefig("results/result.jpg")
        plt.show()
        print("结果请查看根目录下的results文件夹")

        return
    # 彩色直方图
    elif hist_choi == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)
        color = ["r", "g", "b"]
        img_b, img_g, img_r = cv2.split(img)
        img = cv2.merge([img_r, img_g, img_b])
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        for index, c in enumerate(color):
            hist = cv2.calcHist([img], [index], None, [256], [0, 255])

            print(c + "通道：")
            print('最大像素值：', max(hist))
            print('最小像素值：', min(hist))
            plt.plot(hist, color=c)
            plt.xlim([0, 255])
        plt.savefig("results/result.jpg")
        plt.show()
        print("结果请查看根目录下的results文件夹")

        return
    # 分段线性处理
    elif hist_choi == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 0)
        img_h, img_w = img.shape[:2]
        out = np.zeros(img.shape, np.uint8)

        try:
            amoun = int(input("请输入分段函数的段数："))
        except ValueError:
            print("段数输入错误，请重新输入：")
            return

        if amoun <= 0:
            print("段数不为正整数，输入错误，请重新输入：")
            return

        funcIn = [[] for i in range(amoun)]

        print("请输入分段函数。对于每一段在范围[min, max)、形如y=kx+b的函数，依次键入min,max,k和b，用半角逗号隔开。每一段之间换行.最后一段max输入任意数字：")
        for i in range(amoun):
            funcIn[i] = [eval(x) for x in input().split(',')]

        for p in range(amoun - 1):
            if (funcIn[p][0] < 0) or (funcIn[p][1] < 0):
                print("极值小于零，输入错误，请重新输入：")
                return

            if funcIn[p][0] > funcIn[p][1]:
                funcIn[p][0], funcIn[p][1] = usage.swap(funcIn[p][0], funcIn[p][1])

        if (funcIn[amoun - 1][0] < 0) or (funcIn[amoun - 1][1] < 0):
            print("极值小于零，输入错误，请重新输入：")
            return

        if funcIn[amoun - 1][0] > funcIn[amoun - 1][1]:
            funcIn[amoun - 1][0], funcIn[amoun - 1][1] = usage.swap(funcIn[amoun - 1][0], funcIn[amoun - 1][1])

        for i in range(img_h):
            for j in range(img_w):
                pix = img[i][j]
                for p in range(amoun - 1):
                    if (pix >= funcIn[p][0]) and (pix < funcIn[p][1]):
                        out[i][j] = funcIn[p][2] * pix + funcIn[p][3]
                if pix >= funcIn[amoun - 1][0]:
                    out[i][j] = funcIn[amoun - 1][2] * pix + funcIn[amoun - 1][3]

        out = np.around(out)
        out = out.astype(np.uint8)
        usage.grayHist(out, "results/result.jpg")
        print("结果请查看根目录下的results文件夹")

        return
    else:
        print("选择操作输入错误，请重新输入：")


def age_transform():
    img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")

    index = img_path.rfind('.')

    result_path = ""

    for i in range(index):
        result_path += img_path[i]

    result_path += ".mp4"
    img_path = "assets/" + img_path

    path_file = open("assets/path.txt", 'w')
    path_file.seek(0)
    path_file.truncate()
    path_file.write(img_path)

    gen = input("请输入图片中人物的性别，男性请输入\"male\"，女性请输入\"female\"：")
    gen += "_model "

    os.system("""
    cd Lifespan_Age_Transformation_Synthesis;
    pip install -r requirements.txt;
    python download_models.py;
    CUDA_VISIBLE_DEVICES=0 python test.py --name 
    """ + gen + """
    --which_epoch latest --display_id 0 --traverse --interp_step 0.05 --image_path_file path.txt --make_video --in_the_wild --verbose 
    """)

    res_vid = open("Lifespan_Age_Transformation_Synthesis/results/" + gen + "/test_latest/traversal/" + result_path)
    content = res_vid.read()
    res = open("results" + result_path, 'wb')
    res.write(content)
    res_vid.close()
    res.close()
    print("结果请查看根目录下的results文件夹")

    return


def basic_func():
    print("可选操作：\n0.返回上一步\n1.逻辑与\n2.逻辑或\n3.逻辑非\n4.加法\n5.减法\n6.乘法\n7.除法\n8.翻转\n9.平移\n10.旋转\n11.放大缩小"
          "\n12.仿射变换\n13.灰度化\n14.二值化")

    try:
        basic_choi = int(input("请选择要进行的基本操作："))
    except ValueError:
        print("选择基本操作输入错误，请重新输入：")
        return

    if basic_choi == 0:
        return
    # 逻辑与
    elif basic_choi == 1:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path1 = "assets/" + img_path1
        if not (os.path.exists(img_path1)):
            print("文件不存在！")

            return
        img_path2 = input("第二张图片：")
        img_path2 = "assets/" + img_path2

        if not (os.path.exists(img_path2)):
            print("文件不存在！")

            return
        img1 = cv2.imread(img_path1, 0)
        img2 = cv2.imread(img_path2, 0)
        result = img1 & img2
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 逻辑或
    elif basic_choi == 2:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path1 = "assets/" + img_path1
        if not (os.path.exists(img_path1)):
            print("文件不存在！")

            return
        img_path2 = input("第二张图片：")
        img_path2 = "assets/" + img_path2
        if not (os.path.exists(img_path2)):
            print("文件不存在！")

            return
        img1 = cv2.imread(img_path1, 0)
        img2 = cv2.imread(img_path2, 0)
        result = img1 | img2
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 逻辑非
    elif basic_choi == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 0)
        result = ~img
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 加法
    elif basic_choi == 4:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path1 = "assets/" + img_path1
        if not (os.path.exists(img_path1)):
            print("文件不存在！")

            return
        img_path2 = input("第二张图片：")
        img_path2 = "assets/" + img_path2
        if not (os.path.exists(img_path2)):
            print("文件不存在！")

            return
        img1 = cv2.imread(img_path1, 1)
        img2 = cv2.imread(img_path2, 1)
        result = cv2.add(img1, img2)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 减法
    elif basic_choi == 5:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path1 = "assets/" + img_path1
        if not (os.path.exists(img_path1)):
            print("文件不存在！")

            return
        img_path2 = input("第二张图片：")
        img_path2 = "assets/" + img_path2
        if not (os.path.exists(img_path2)):
            print("文件不存在！")

            return
        img1 = cv2.imread(img_path1, 1)
        img2 = cv2.imread(img_path2, 1)
        result = cv2.subtract(img1, img2)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 乘法
    elif basic_choi == 6:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path1 = "assets/" + img_path1
        if not (os.path.exists(img_path1)):
            print("文件不存在！")

            return
        img_path2 = input("第二张图片：")
        img_path2 = "assets/" + img_path2
        if not (os.path.exists(img_path2)):
            print("文件不存在！")

            return
        img1 = cv2.imread(img_path1, 1)
        img2 = cv2.imread(img_path2, 1)
        result = cv2.multiply(img1, img2)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 除法
    elif basic_choi == 7:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path1 = "assets/" + img_path1
        if not (os.path.exists(img_path1)):
            print("文件不存在！")

            return
        img_path2 = input("第二张图片：")
        img_path2 = "assets/" + img_path2
        if not (os.path.exists(img_path2)):
            print("文件不存在！")

            return
        img1 = cv2.imread(img_path1, 1)
        img2 = cv2.imread(img_path2, 1)
        result = cv2.divide(img1, img2)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 翻转
    elif basic_choi == 8:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)

        print("可选翻转方向：\n1.水平\n2.垂直\n3.对角）")

        try:
            fli_choi = int(input("请选择翻转方向："))
        except ValueError:
            print("选择方向输入错误，请重新输入：")
            return

        if fli_choi == 1:
            result = cv2.flip(img, 1)
        elif fli_choi == 2:
            result = cv2.flip(img, 0)
        elif fli_choi == 3:
            result = cv2.flip(img, -1)
        else:
            print("选择方向输入错误，不执行翻转操作")
            result = img
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 平移
    elif basic_choi == 9:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)
        img_height, img_width, img_channel = img.shape
        pix_x = input("请输入x轴移动像素数（向左为正，向右为负）：")
        pix_y = input("请输入y轴移动像素数（向下为正，向上为负）：")
        pix_M = np.float32([[1, 0, pix_x], [0, 1, pix_y]])
        result = cv2.warpAffine(img, pix_M, (img_width, img_height))
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 旋转
    elif basic_choi == 10:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)

        try:
            ang = int(input("请输入旋转角度（默认顺时针，缩小以适应变换）："))
        except ValueError:
            print("选择角度输入错误，请重新输入：")
            return

        result = usage.rotate(img, ang)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 放缩
    elif basic_choi == 11:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)

        try:
            siz_x = float(input("请输入x方向放大倍数："))
        except ValueError:
            print("放大倍数输入错误，请重新输入：")
            return

        if siz_x <= 0:
            print("放缩倍数只能为正数！")
            print("将使用默认倍数：1")
            siz_x = 1

        try:
            siz_y = float(input("请输入y方向放大倍数："))
        except ValueError:
            print("放大倍数输入错误，请重新输入：")
            return

        if siz_y <= 0:
            print("放缩倍数只能为正数！")
            print("将使用默认倍数：1")
            siz_y = 1

        result = cv2.resize(img, (0, 0), fx=siz_x, fy=siz_y, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 仿射变换
    elif basic_choi == 12:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (256, 256))
        img_rows, img_cols = img.shape[: 2]

        try:
            pos_x1, pos_y1 = input("请输入a点变换前坐标（格式为\"x y\"）：").split()
            pos_x1_, pos_y1_ = input("请输入a点变换后坐标（格式为\"x y\"）：").split()
            pos_x2, pos_y2 = input("请输入b点变换前坐标（格式为\"x y\"）：").split()
            pos_x2_, pos_y2_ = input("请输入b点变换后坐标（格式为\"x y\"）：").split()
            pos_x3, pos_y3 = input("请输入c点变换前坐标（格式为\"x y\"）：").split()
            pos_x3_, pos_y3_ = input("请输入c点变换后坐标（格式为\"x y\"）：").split()
        except ValueError:
            print("输入参数个数不足，请重新输入：")
            return

        try:
            pos_x1 = int(pos_x1)
            pos_y1 = int(pos_y1)
            pos_x1_ = int(pos_x1_)
            pos_y1_ = int(pos_y1_)
            pos_x2 = int(pos_x2)
            pos_y2 = int(pos_y2)
            pos_x2_ = int(pos_x2_)
            pos_y2_ = int(pos_y2_)
            pos_x3 = int(pos_x3)
            pos_y3 = int(pos_y3)
            pos_x3_ = int(pos_x3_)
            pos_y3_ = int(pos_y3_)
        except ValueError:
            print("坐标输入错误，请重新输入：")
            return

        post1 = np.float32([[pos_x1, pos_y1], [pos_x2, pos_y2], [pos_x3, pos_y3]])
        post2 = np.float32([[pos_x1_, pos_y1_], [pos_x2_, pos_y2_], [pos_x3_, pos_y3_]])
        pix_M = cv2.getAffineTransform(post1, post2)
        result = cv2.warpAffine(img, pix_M, (img_rows, img_cols))
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 灰度化
    elif basic_choi == 13:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 二值化
    elif basic_choi == 14:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, result = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")

    else:
        print("选择操作输入错误，请重新输入：")


def morphological():
    print("可选操作：\n0.返回上一步\n1.腐蚀\n2.膨胀\n3.开运算\n4.闭运算")

    try:
        morp_choi = int(input("请选择要进行的基本操作："))
    except ValueError:
        print("选择基本操作输入错误，请重新输入：")
        return

    if morp_choi == 0:

        return
    # 腐蚀
    elif morp_choi == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        print("可选结构元类型：\n1.交叉型\n2.矩形）")

        try:
            kernel_type = int(input("请输入要使用的结构元类型："))
            kernel_size = int(input("请输入要使用的结构元大小k（奇数，大小为k*k）："))
        except ValueError:
            print("结构元类型或结构元大小输入错误，请重新输入：")
            return

        if kernel_size <= 0:
            print("结构元大小输入错误，请重新输入：")
            return

        if (kernel_size % 2) == 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        if kernel_type == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_type == 2:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:
            print("结构元类型输入错误，将使用默认结构元类型：矩型")
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = cv2.erode(img, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 膨胀
    elif morp_choi == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        print("可选结构元类型：\n1.交叉型\n2.矩形）")

        try:
            kernel_type = int(input("请输入要使用的结构元类型："))
            kernel_size = int(input("请输入要使用的结构元大小k（奇数，大小为k*k）："))
        except ValueError:
            print("结构元类型或结构元大小输入错误，请重新输入：")
            return

        if kernel_size <= 0:
            print("结构元大小输入错误，请重新输入：")
            return

        if (kernel_size % 2) == 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        if kernel_type == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_type == 2:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:
            print("结构元类型输入错误，将使用默认结构元类型：矩型")
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = cv2.dilate(img, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 开运算
    elif morp_choi == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        print("可选结构元类型：\n1.交叉型\n2.矩形）")

        try:
            kernel_type = int(input("请输入要使用的结构元类型："))
            kernel_size = int(input("请输入要使用的结构元大小k（奇数，大小为k*k）："))
        except ValueError:
            print("结构元类型或结构元大小输入错误，请重新输入：")
            return

        if kernel_size <= 0:
            print("结构元大小输入错误，请重新输入：")
            return

        if (kernel_size % 2) == 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        if kernel_type == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_type == 2:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:
            print("结构元类型输入错误，将使用默认结构元类型：矩型")
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")
    # 闭运算
    elif morp_choi == 4:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        print("可选结构元类型：\n1.交叉型\n2.矩形）")

        try:
            kernel_type = int(input("请输入要使用的结构元类型："))
            kernel_size = int(input("请输入要使用的结构元大小k（奇数，大小为k*k）："))
        except ValueError:
            print("结构元类型或结构元大小输入错误，请重新输入：")
            return

        if kernel_size <= 0:
            print("结构元大小输入错误，请重新输入：")
            return

        if (kernel_size % 2) == 0:
            print("卷积核大小输入错误，请重新输入：")
            return

        if kernel_type == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_type == 2:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:
            print("结构元类型输入错误，将使用默认结构元类型：矩型")
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("results/result.jpg", result)
        print("结果请查看根目录下的results文件夹")

    else:
        print("选择操作输入错误，请重新输入：")


def smooth_or_sharpen():
    print("可选操作：\n0.返回上一步\n1.空域平滑\n2.空域锐化\n3.频域平滑\n4.频域锐化")

    try:
        sos_choi = int(input("请选择要进行的基本操作："))
    except ValueError as e:
        print("选择基本操作输入错误，请重新输入：")
        return

    if sos_choi == 0:
        return

    elif sos_choi == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        print("可选空域平滑方法：\n1.邻域平均法\n2.中值滤波法")
        try:
            fun_choi = int(input("请选择要使用的空域平滑方法："))
        except ValueError as e:
            print("选择方法输入错误，请重新输入：")
            return

        if fun_choi == 1:
            try:
                kernel_size = int(input("请输入计算单元核数："))
            except ValueError as e:
                print("核数输入错误，请重新输入：")
                return

            if kernel_size <= 0:
                print("结构元大小输入错误，请重新输入：")
                return

            if (kernel_size % 2) == 0:
                print("卷积核大小输入错误，请重新输入：")
                return

            source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = cv2.blur(source, (kernel_size, kernel_size))

            cv2.imshow("result", result)
            cv2.imwrite("results/result.jpg", result)
            print("结果请查看根目录下的results文件夹")

        elif fun_choi == 2:
            try:
                kernel_size = int(input("请输入计算单元核数："))
            except ValueError as e:
                print("核数输入错误，请重新输入：")
                return

            if kernel_size <= 0:
                print("结构元大小输入错误，请重新输入：")
                return

            if (kernel_size % 2) == 0:
                print("卷积核大小输入错误，请重新输入：")
                return

            result = cv2.medianblur(img, kernel_size)

            cv2.imshow("result", result)
            cv2.imwrite("results/result.jpg", result)
            print("结果请查看根目录下的results文件夹")

    elif sos_choi == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path = "assets/" + img_path
        if not (os.path.exists(img_path)):
            print("文件不存在！")

            return
        img = cv2.imread(img_path)

        print("可选空域平滑方法：\n1.Robert梯度算子\n2.Laplacian梯度算子\n3.Sobel算子")
        try:
            fun_choi = int(input("请选择要使用的空域平滑方法："))
        except ValueError as e:
            print("选择方法输入错误，请重新输入：")
            return

        if fun_choi == 1:
            h = img.shape[0]
            w = img.shape[1]
            result = np.zeros(img.shape, np.uint8)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    result[i][j] = np.abs(img[i][j].astype(int) - img[i + 1][j + 1].astype(int)) + np.abs(
                        img[i + 1][j].astype(int) - img[i][j + 1].astype(int))
            cv2.imshow("result", result)
            cv2.imwrite("results/result.jpg", result)
            print("结果请查看根目录下的results文件夹")

        elif fun_choi == 2:
            h = img.shape[0]
            w = img.shape[1]
            result = np.zeros(img.shape, np.uint8)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    result[i][j] = 4 * img[i][j].astype(int) - img[i + 1][j].astype(int) - img[i - 1][j].astype(int) - \
                                   img[i][j + 1].astype(int) - img[i][j - 1].astype(int)
            cv2.imshow("result", result)
            cv2.imwrite("results/result.jpg", result)
            print("结果请查看根目录下的results文件夹")

        elif fun_choi == 3:
            kernx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kerny = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            imgx = cv2.filter2D(img, -1, kernx, borderType=cv2.BORDER_REFLECT)
            imgy = cv2.filter2D(img, -1, kerny, borderType=cv2.BORDER_REFLECT)
            absx = cv2.convertScaleAbs(imgx)
            absy = cv2.convertScaleAbs(imgy)
            result = cv2.addWeighted((absx, 0.5, absy, 0.5, 0))
            cv2.imshow("result", result)
            cv2.imwrite("results/result.jpg", result)
            print("结果请查看根目录下的results文件夹")

        elif fun_choi == 4:
            kernx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kerny = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            imgx = cv2.filter2D(img, cv2.CV_16S, kernx)
            imgy = cv2.filter2D(img, cv2.CV_16S, kerny)
            absx = cv2.convertScaleAbs(imgx)
            absy = cv2.convertScaleAbs(imgy)
            result = cv2.addWeighted((absx, 0.5, absy, 0.5, 0))
            cv2.imshow("result", result)
            cv2.imwrite("results/result.jpg", result)
            print("结果请查看根目录下的results文件夹")
