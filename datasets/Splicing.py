import cv2
import numpy as np

image = cv2.imread('F://test.png')
# 第一个局部放大图
pt1 = (150, 70)  # 长方形框左上角坐标
pt2 = (250, 130)  # 长方形框右下角坐标
cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
patch1 = image[70:130, 150:250, :]
patch1 = cv2.resize(patch1, (200, 120))
# 第二个局部放大图
pt1 = (265, 70)  # 长方形框左上角坐标
pt2 = (365, 130)  # 长方形框右下角坐标
cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
patch2 = image[70:130, 265:365, :]
patch2 = cv2.resize(patch2, (200, 120))
# 拼接
patch = np.hstack((patch1, patch2))
image = np.vstack((image, patch))
cv2.imshow('demo', image)
cv2.imwrite('F://test_result.png', image)
