import  time

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

        try:
            operation = int(input("请选择要进行的基本操作（输入数字）："))
        except ValueError as e:
            print("输入错误，请重新输入：")
            continue

        if operation == 0:
            break
        elif operation == 1:
            utils.basic_func()
        elif operation == 2:
            utils.histogram()
        elif operation == 3:
            utils.img_segmentation()
        elif operation == 5:
            utils.morphological()
        elif operation == 6:
            utils.img_repair()
        elif operation == 7:
            utils.age_transform()
        else:
            print("输入错误，请重新输入：")



if __name__ == "__main__":
    main()
