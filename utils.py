import cv2
import numpy as np


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
