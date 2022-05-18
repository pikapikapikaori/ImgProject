import cv2
import numpy as np
import utils


def main():
    print("图像处理程序")
    print("-------------------")
    print("请选择要执行的操作")
    operation = -1;
    while operation != 0:
        print("可选操作：\n0.结束程序\n1.图像基本操作\n2.图像直方图绘制与图像增强\n3.图像分割\n4.图像平滑与锐化\n5.图像形态学操作\n6.图像恢复\n7.年龄变换")
        operation = int(input("请选择要进行的基本操作（输入数字）："))
        if operation == 0:
            break
        if operation ==1:
            utils.basic_func()
        if operation == 2:
            utils.histogram()
        if operation == 3:
            utils.img_segmentation()
        if operation == 5:
            utils.morphological()
        if operation == 7:
            utils.age_transform()


if __name__ == "__main__":
    main()
