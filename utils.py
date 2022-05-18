import os
import sys

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

import usage


def img_segmentation():
    print("可选操作：\n0.返回上一步\n1.图像增强\n2.Roberts算子边缘检测\n3.Sobel算子边缘检测\n4.Laplacian算子边缘检测\n5.LoG算子边缘检测\n6.Canny算子边缘检测"
          "\n7.HoughLines算法线条变化检测\n8.HoughLinesP算法线条变化检测")
    a = int(input("请选择要进行的基本操作（输入数字）："))

    if a == 0:
        return
    if a == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        CRH = cv2.imread(img_path)

        row, column = CRH.shape

        CRH_f = np.copy(CRH)
        CRH_f = CRH_f.astype("float")

        gradient = np.zeros((row, column))

        for x in range(row - 1):
            for y in range(column - 1):
                gx = abs(CRH_f[x + 1, y] - CRH_f[x, y])
                gy = abs(CRH_f[x, y + 1] - CRH_f[x, y])
                gradient[x, y] = gx + gy

        sharp = CRH + gradient

        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        gradient = gradient.astype('uint8')
        sharp = sharp.astype('uint8')
        cv2.imwrite("/results/result.jpg", gradient)
    if a == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)

        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        cv2.imwrite("/results/result.jpg", Roberts)
    if a == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernelx = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
        kernely = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)

        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        cv2.imwrite("/results/result.jpg", Sobel)
    if a == 4:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grayImage, (5, 5), 0)

        dst = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)

        Laplacian = cv2.convertScaleAbs(dst)

        cv2.imwrite("/results/result.jpg", Laplacian)
    if a == 5:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
        image = cv2.GaussianBlur(image, (3, 3), 0, 0)


        print("默认LoG算子为：[[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]")
        choi = int(input("请输入是否使用默认LoG算子，输入1使用默认算子，输入2手动输入："))

        if choi == 1:
            LoGMatr = [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]

        if choi == 2:
            LoGMatr = [[] for i in range(5)]
            print("请输入LoG算子矩阵。每列用半角逗号隔开。每行之间用enter换行：")
            for i in range(5):
                LoGMatr[i] = [eval(x) for x in input().split(',')]


        m1 = np.array(LoGMatr)

        rows = image.shape[0]
        cols = image.shape[1]

        image1 = np.zeros(image.shape)

        for k in range(0, 2):
            for i in range(2, rows - 2):
                for j in range(2, cols - 2):
                    image1[i, j] = np.sum((m1 * image[i - 2:i + 3, j - 2:j + 3, k]))

        image1 = cv2.convertScaleAbs(image1)

        cv2.imwrite("/results/result.jpg", image1)
    if a == 6:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)

        blur = cv2.GaussianBlur(img, (3, 3), 0)

        grayImage = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        gradx = cv2.Sobel(grayImage, cv2.CV_16SC1, 1, 0)
        grady = cv2.Sobel(grayImage, cv2.CV_16SC1, 0, 1)

        edge_output = cv2.Canny(gradx, grady, 50, 150)

        cv2.imwrite("/results/result.jpg", edge_output)
    if a == 7:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
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

        minLineLength = 200
        maxLineGap = 15
        cv2.imwrite("/results/result.jpg", result)
    if a == 8:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

        result_P = img.copy()
        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imwrite('./step5/test_result/result.png', result)
        cv2.imwrite('./step5/test_result/result_P.png', result_P)

        cv2.destroyAllWindows()


def histogram():
    print("可选操作：\n0.返回上一步\n1.灰度直方图\n2.彩色直方图\n3.分段线性处理")
    a = int(input("请选择要进行的基本操作（输入数字）："))

    if a == 0:
        return
    if a == 1:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path, 0)

        plt.figure("/results/result.jpg", figsize=(16, 8))
        plt.subplot(121)
        plt.imshow(img, "gray")
        plt.subplot(122)
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        print('最大像素值：', max(hist))
        print('最小像素值：', min(hist))
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.show()

        return
    if a == 2:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path)
        color = ["r", "g", "b"]
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
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
        plt.savefig("/results/result.jpg")
        plt.show()

        return
    if a == 3:
        img_path = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img = cv2.imread(img_path, 0)
        h, w = img.shape[:2]
        out = np.zeros(img.shape, np.uint8)

        amoun = input("请输入分段函数的段数：")
        funcIn = [[] for i in range(amoun)]

        print("请输入分段函数。对于每一段在范围[min, max)、形如y=kx+b的函数，依次键入min,max,k和b，用半角逗号隔开。每一段之间换行：")
        for i in range(amoun):
            funcIn[i] = [eval(x) for x in input().split(',')]

        for i in range(h):
            for j in range(w):
                pix = img[i][j]
                for p in range(amoun):
                    if (pix >= funcIn[p][0]) and (pix < funcIn[p][1]):
                        out[i][j] = funcIn[p][2] * pix + funcIn[p][3]

        out = np.around(out)
        out = out.astype(np.uint8)
        usage.grayHist(out, "/results/result.jpg")

        return


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

    gen = input("请输入图片中人物的性别，男性请输入\'male\'，女性请输入\'female\'：")
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
    a = int(input("请选择要进行的基本操作（输入数字）："))
    if a == 0:
        return
    if a == 1:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path2 = input("第二张图片：")
        X = cv2.imread('/assets' + img_path1, 0)
        Y = cv2.imread('/assets' + img_path2, 0)
        result = X & Y
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 2:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path2 = input("第二张图片：")
        X = cv2.imread('/assets' + img_path1, 0)
        Y = cv2.imread('/assets' + img_path2, 0)
        result = X | Y
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 3:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 0)
        result = ~X
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 4:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path2 = input("第二张图片：")
        X = cv2.imread('/assets' + img_path1, 1)
        Y = cv2.imread('/assets' + img_path2, 1)
        result = cv2.add(X, Y)
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 5:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path2 = input("第二张图片：")
        X = cv2.imread('/assets' + img_path1, 1)
        Y = cv2.imread('/assets' + img_path2, 1)
        result = cv2.subtract(X, Y)
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 6:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path2 = input("第二张图片：")
        X = cv2.imread('/assets' + img_path1, 1)
        Y = cv2.imread('/assets' + img_path2, 1)
        result = cv2.multiply(X, Y)
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 7:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        img_path2 = input("第二张图片：")
        X = cv2.imread('/assets' + img_path1, 1)
        Y = cv2.imread('/assets' + img_path2, 1)
        result = cv2.divide(X, Y)
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 8:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        x = input("请输入翻转方向（1.水平2.垂直3.对角）：")
        if x == 1:
            result = cv2.flip(X, 1)
        if x == 2:
            result = cv2.flip(X, 0)
        if x == 3:
            result = cv2.flip(X, -1)
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 9:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        height, width, channel = X.shape
        x = input("请输入x轴移动像素数（向左为正，向右为负）：")
        y = input("请输入y轴移动像素数（向下为正，向上为负）：")
        M = np.float32([[1, 0, x], [0, 1, y]])
        result = cv2.warpAffine(X, M, (width, height))
        cv2.imshow(result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 10:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        height, width, channel = X.shape
        z = input("请输入旋转角度：")
        M = cv2.getRotationMatrix2D((width, height), z, 1)
        result = cv2.warpAffine(X, M, (width, height))
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 11:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        x = int(input("请输入x方向放大倍数："))
        y = int(input("请输入y方向放大倍数："))
        result = cv2.resize(X, (0, 0), fx=x, fy=y, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 12:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        X = cv2.resize(X, (256, 256))
        rows, cols = X.shape[: 2]
        x1, y1 = input("请输入a点变换前坐标（格式为”x y“：")
        x1_, y1_ = input("请输入a点变换后坐标（格式为”x y“：")
        x2, y2 = input("请输入b点变换前坐标（格式为”x y“：")
        x2_, y2_ = input("请输入b点变换后坐标（格式为”x y“：")
        x3, y3 = input("请输入c点变换前坐标（格式为”x y“：")
        x3_, y3_ = input("请输入c点变换后坐标（格式为”x y“：")
        x1 = int(x1)
        y1 = int(y1)
        x1_ = int(x1_)
        y1_ = int(y1_)
        x2 = int(x2)
        y2 = int(y2)
        x2_ = int(x2_)
        y2_ = int(y2_)
        x3 = int(x3)
        y3 = int(y3)
        x3_ = int(x3_)
        y3_ = int(y3_)
        post1 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
        post2 = np.float32([[x1_, y1_], [x2_, y2_], [x3_, y3_]])
        M = cv2.getAffineTransform(post1, post2)
        result = cv2.warpAffine(X, M, (rows, cols))
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 13:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        result = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 14:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, 1)
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        ret, result = cv2.threshold(X, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    print("结果请查看根目录下的results文件夹")
    return


def morphological():
    print("可选操作：\n0.返回上一步\n1.腐蚀\n2.膨胀\n3.开运算\n4.闭运算")
    a = int(input("请选择要进行的基本操作（输入数字）："))
    if a == 0:
        return
    if a == 1:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, cv2.IMREAD_UNCHANGED)
        b = int(input("请输入要使用的结构元类型（0交叉型1矩形）："))
        if b == 0:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        if b == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.erode(X, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 2:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, cv2.IMREAD_UNCHANGED)
        b = int(input("请输入要使用的结构元类型（0交叉型1矩形）："))
        if b == 0:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        if b == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.dilate(X, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 3:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, cv2.IMREAD_UNCHANGED)
        b = int(input("请输入要使用的结构元类型（0交叉型1矩形）："))
        if b == 0:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        if b == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.morphologyEx(X, cv2.MORPH_OPEN, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    if a == 4:
        img_path1 = input("请将图像放置于根目录下的assets文件夹中，并输入图像的名称：")
        X = cv2.imread('/assets' + img_path1, cv2.IMREAD_UNCHANGED)
        b = int(input("请输入要使用的结构元类型（0交叉型1矩形）："))
        if b == 0:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        if b == 1:
            kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.morphologyEx(X, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("result", result)
        cv2.imwrite("/results/result.jpg", result)
    print("结果请查看根目录下的results文件夹")
    return
