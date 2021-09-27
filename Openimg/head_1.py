# -*- coding:utf-8 -*-
"""
@author:Skl
@file:head.py
@time:2018/5/322:01
"""

import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(
    '../data/shape_predictor_68_face_landmarks.dat')

def mywindow(image_path, save_path):
        img = cv2.imread(image_path)  # 读取
        imgg = img.copy()
        t = img.shape
        faces = detector(img, 1)

        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if (len(faces) > 0):
            for k, d in enumerate(faces):
                left = max(int((3 * d.left() - d.right()) / 2), 1)
                top = max(int((3 * d.top() - d.bottom()) / 2) - 50, 1)
                right = min(int((3 * d.right() - d.left()) / 2), t[1])
                bottom = min(int((3 * d.bottom() - d.top()) / 2), t[0])
                rect = (left, top, right, bottom)
                rect_reg = (d.left(), d.top(), d.right(), d.bottom())
                shape = landmark_predictor(img, d)
                xx, yy = 0, 0
                for i in range(68):
                    xx += shape.part(i).x
                    yy += shape.part(i).y
                xx, yy = int(xx / 68), int(yy / 68)
        else:
            exit(0)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5,
                    cv2.GC_INIT_WITH_RECT)  # 函数返回值为mask,bgdModel,fgdModel
        mask2 = np.where(
            (mask == 2) | (
                mask == 0),
            0,
            1).astype('uint8')  # 0和2做背景

        img = img * mask2[:, :, np.newaxis]  # 使用蒙板来获取前景区域
        erode = cv2.erode(img, None, iterations=1)
        dilate = cv2.dilate(erode, None, iterations=1)
        for i in range(t[0]):  # 高
            for j in range(t[1]):
                if max(dilate[i, j]) <= 0:
                    dilate[i, j] = (225, 255, 255)  # BGR
        img = img[rect[1]:rect[3], rect[0]:rect[2]]
        dilate = dilate[rect[1]:rect[3], rect[0]:rect[2]]
        output_im = cv2.resize(dilate, (361, 381))  #证件照
        output_imreg = cv2.resize(img, (361, 381))  #轮廓

        # 截取图片，预处理
        imgg = imgg[rect[1]
            :rect[3], rect[0]:rect[2]]
        imgg = cv2.resize(imgg, (361, 381))  #截取后的图片

        # 保存图片
        cv2.imwrite(save_path, output_im, None)

if __name__ == '__main__':
    mywindow("11.jpg", "save.jpg")