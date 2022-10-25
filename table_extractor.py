# coding=utf-8
import cv2
import os
import numpy as np


# from imutils.perspective import four_point_transform


def FindContours(img_path):
    src_img = cv2.imread(img_path)
    src_img0 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_img0 = cv2.GaussianBlur(src_img0, (3, 3), 0)
    src_img1 = cv2.bitwise_not(src_img0)
    AdaptiveThreshold = cv2.adaptiveThreshold(src_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()
    scale = 20

    horizontalSize = int(horizontal.shape[1] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imshow("horizontal", horizontal)
    cv2.waitKey(0)

    verticalsize = int(vertical.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    cv2.imshow("verticalsize", vertical)
    cv2.waitKey(0)

    mask = horizontal + vertical
    # np.count_nonzero(mask, axis=0) #对列统计白色（非零值）
    # np.count_nonzero(mask, axis=1) #对行统计白色（非零值）
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

    Net_img = cv2.bitwise_and(horizontal, vertical)
    cv2.imshow("Net_img", Net_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # =========================绘出所有轮廓=========================
    # IMG = cv2.drawContours(src_img0, contours, -1, (0, 255, 255), 2)
    # cv2.imshow('IMG', IMG)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # =============================================================
    return src_img, Net_img, contours


def get_Affine_Location(src_img, Net_img, contours, cutImg_name, cutImg_path):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(len(contours)):
        area0 = cv2.contourArea(contours[i])
        if area0 < 20: continue

        # =======================查找每个表的关节数====================
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  # 获取近似轮廓
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        roi = Net_img[int(y1):int(y1 + h1), int(x1):int(x1 + w1)]
        _, roi_contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print('len(roi_contours):', len(roi_contours))
        if len(roi_contours) < 4: continue

        src_img1 = cv2.rectangle(src_img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cut_img = src_img[y1:y1 + h1, x1:x1 + w1]
        cv2.imwrite(cutImg_path + '/' + cutImg_name + '_' + str(i) + '.png', cut_img)  # 保存截取的图片
        cv2.imshow('src_img_' + str(i), src_img1)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # =========================绘出最大轮廓凸包并矫正图像==========================
    # epsilon = 0.1 * cv2.arcLength(contours[0], True)
    # approx = cv2.approxPolyDP(contours[0], epsilon, True)         # 获取近似轮廓
    # hull = cv2.convexHull(approx)                                 # 默认返回坐标点
    # hull_img = cv2.polylines(src_img, [hull], True, (0, 255, 0), 2)
    # cv2.imshow('hull_img', hull_img)
    # cv2.waitKey(0)
    #
    # if len(hull) == 4:
    #     dst = four_point_transform(src_img, hull.reshape(4,2))    # 矫正变换
    #     cv2.imwrite(cutImg_path+'/'+cutImg_name+'max.png', dst)   # 保存截取的图片
    #     cv2.imshow("result", dst)
    #     cv2.waitKey(0)
    #
    # Img_max = cv2.drawContours(src_img, contours, 0, (0, 255, 0), 2, 1)
    # cv2.imshow('ImgMax', Img_max)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    print("识别图片中的表格，比如论文里面的表格，将每个表格单独扣出来")
    input_Path = '1.jpg'
    cutImg_path = '.'
    cutImg_name = input_Path.split('/')[-1][:-4]
    src_img, Net_img, contours = FindContours(input_Path)
    get_Affine_Location(src_img, Net_img, contours, cutImg_name, cutImg_path)
