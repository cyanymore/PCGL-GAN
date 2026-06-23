import cv2
import numpy as np


def image_mean(image):
    mean = np.mean(image)
    return mean


def image_var(image, mean):
    m, n = image.shape
    var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
    return var


def images_cov(image1, image2, mean1, mean2):
    m, n = image1.shape
    cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
    return cov


def UQI(O, F):
    meanO = image_mean(O)
    meanF = image_mean(F)
    varO = image_var(O, meanO)
    varF = image_var(F, meanF)
    covOF = images_cov(O, F, meanO, meanF)
    UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
    return UQI


# # 加载原始图像和失真图像
# original_image = cv2.imread('05303.jpg', cv2.IMREAD_GRAYSCALE)
# distorted_image = cv2.imread('00001.jpg', cv2.IMREAD_GRAYSCALE)
#
# # 调用 UQI 函数
# uqi_value = UQI(original_image, distorted_image)
#
# # 打印 UQI 值
# print("UQI:", uqi_value)
