# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("1.png", 0)
cv2.imshow("Image", img)
cv2.waitKey(0)

binary = cv2.adaptiveThreshold(~img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -2)
cv2.imshow("Image", binary)
cv2.waitKey(0)

rows, cols = binary.shape
print(rows, cols)
scale = 20

# 识别横线
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)

# 由于图像像素质量的原因，一次膨胀不够，往往有些线无法识别，我用二次膨胀。
dilatedcol = cv2.dilate(eroded, kernel, iterations=2)
scale = 10
# 识别竖线
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)

cv2.imshow("Eroded Image", eroded)
cv2.waitKey(0)

dilatedrow = cv2.dilate(eroded, kernel, iterations=2)

# 标识交点, 图片的交集
bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)

# 标识表格

merge = cv2.add(dilatedcol, dilatedrow)
cv2.imshow("Eroded Image", merge)
cv2.waitKey(0)
# 识别黑白图中的白色交叉点，将横纵坐标取出

print("bitwiseAnd:", bitwiseAnd)
ys, xs = np.where(bitwiseAnd > 0)
print(ys)
print(xs)

mylisty = []  # 纵坐标

mylistx = []  # 横坐标

# 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点

# 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变），对于多个点，我取x，y中间值mean

i = 0

myxs = np.sort(xs)

tmpmy = []

for i in range(len(myxs) - 1):

    if (myxs[i + 1] - myxs[i] > 10):
        tmpmy.append(myxs[i])

        mylistx.append(int(np.mean(tmpmy)))

        tmpmy = []

    else:

        tmpmy.append(myxs[i])

        i = i + 1

mylistx.append(int(np.mean(tmpmy)))  # 要将最后一个点加入

i = 0

myys = np.sort(ys)

tmpmy = []

for i in range(len(myys) - 1):

    if myys[i + 1] - myys[i] > 10:

        tmpmy.append(myys[i])

        mylisty.append(int(np.mean(tmpmy)))

        tmpmy = []

    else:

        tmpmy.append(myys[i])

        i = i + 1

mylisty.append(int(np.mean(tmpmy)))  # 要将最后一个点加入

print("mylisty:", mylisty)
print("mylistx:", mylistx)
cv2.imshow("bitwiseAnd", bitwiseAnd)
cv2.waitKey(0)

cv2.destroyAllWindows()

for i in range(len(mylisty) - 1):
    j = 0
    # for j in range(len(mylistx) - 1):
    while j < len(mylistx) - 1:
        print("j", j)
        # 在分割时，第一个参数为y坐标，第二个参数为x坐标
        if bitwiseAnd[mylisty[i]][mylistx[j]] > 0 and bitwiseAnd[mylisty[i + 1]][mylistx[j + 1]] > 0 and \
                bitwiseAnd[mylisty[i]][mylistx[j+1]] > 0 and bitwiseAnd[mylisty[i + 1]][mylistx[j]] > 0:
            # print(mylisty[i], mylistx[j])
            # print(mylisty[i+1], mylistx[j+1])
            # ROI = img[mylisty[i] + 3:mylisty[i + 1] - 3, mylistx[j] + 3:mylistx[j + 1] - 3]  # 减去3的原因是由于我缩小ROI范围

            ptLeftTop = (mylistx[j], mylisty[i])
            ptRightBottom = (mylistx[j+1], mylisty[i+1])
            point_color = (255, 0, 255)  # BGR
            thickness = 1
            lineType = 4
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            cv2.imshow("ROI", img)

            #cv2.imshow("ROI", ROI)
            cv2.waitKey(0)
        else:
            k = j+1
            while k < len(mylistx) - 1:
                # print(mylisty[i], mylistx[j], mylisty[i + 1], mylistx[k + 1])
                # print(mylisty[i+1], mylistx[j], mylisty[i], mylistx[k+1])
                if bitwiseAnd[mylisty[i]][mylistx[j]] > 0 and bitwiseAnd[mylisty[i + 1]][mylistx[k + 1]] > 0 and \
                        bitwiseAnd[mylisty[i+1]][mylistx[j]] > 0 and bitwiseAnd[mylisty[i]][mylistx[k+1]] > 0:
                    # ROI = img[mylisty[i] + 3:mylisty[i + 1] - 3, mylistx[j] + 3:mylistx[k + 1] - 3]  # 减去3的原因是由于我缩小ROI范围
                    # print(mylisty[i], mylistx[j], mylisty[i + 1], mylistx[k + 1])

                    ptLeftTop = (mylistx[k], mylisty[i])
                    ptRightBottom = (mylistx[k + 1], mylisty[i + 1])
                    point_color = (255, 0, 255)  # BGR
                    thickness = 1
                    lineType = 4
                    cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
                    cv2.imshow("ROI", img)

                    # cv2.imshow("ROI", ROI)
                    cv2.waitKey(0)
                    j = k
                    break
                k = k+1
        j = j + 1

