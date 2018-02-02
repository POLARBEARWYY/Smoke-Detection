# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import feature as ft


# 读取图片
img = cv2.imread('test8/000001.png')
# 计算灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 显示灰度图
# cv2.imshow('', img_gray)
# cv2.waitKey(5000)
# 计算hog特征和hog图
vec_hog, img_hog = ft.hog(img_gray, visualise=True)
# 显示hog图
# cv2.imshow('', img_hog)
# cv2.waitKey(5000)
# 计算LBP图
img_lbp = ft.local_binary_pattern(img_gray, 8 * 8, 8)
# 显示LBP图
# cv2.imshow('', img_lbp)
# cv2.waitKey(5000)
cv2.imshow('', np.hstack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_hog, img_lbp]))
cv2.waitKey(0)