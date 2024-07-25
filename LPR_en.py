import cv2
import numpy as np
import pytesseract
import imutils
import timeit

from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
start = timeit.timeit()


def Image_feature_enhancement(img, debug=False):
    # 预处理与边缘特征增强
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度图
    filtered = cv2.bilateralFilter(gray, 5, 75, 75)     # 滤波
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # 直方图均衡化
    histogram_e = clahe.apply(filtered)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphology = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel,iterations=15) #
    gcikarilmisresim = cv2.subtract(histogram_e, morphology) # 关键区域突出

    edged = cv2.Canny(gcikarilmisresim, 200, 250)  # 边缘检测
    if debug:
        cv2.imshow("Original License Plate", img)
        cv2.imwrite("Original_image.jpg", img)
        cv2.imshow("Gray License Plate", gray)
        cv2.imwrite("Gray-scale.jpg", gray)
        cv2.imshow("Filtered License Plate", filtered)
        cv2.imwrite("Filtered.jpg", filtered)
        cv2.imshow("histogram eşikleme", histogram_e)
        cv2.imwrite("Histogram_Esikleme.jpg", histogram_e)
        cv2.imshow("Morphology", morphology)
        cv2.imwrite("Morphology.jpg", morphology)
        cv2.imshow("goruntuden cikarilmis resim", gcikarilmisresim)
        cv2.imwrite("Cikarilmisresim.jpg", gcikarilmisresim)
        cv2.imshow("Egded License Plate", edged)
        cv2.imwrite("Edges.jpg", edged)
    return edged


def License_plate_area(edged, debug=False):
    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # 轮廓排序
    screen = None

    if debug:
        i_c = np.copy(img)
        cv2.drawContours(i_c, cnts, -1, (0, 0, 255), 2)
        plt.imshow(i_c[:, :, ::-1])
        plt.show()

    l = []
    for c in cnts:
        epsilon = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True) # 轮廓多项式逼近
        #print("-----------------------")
        #print(len(approx))
        if len(approx) == 4 :#and abs(approx[3][0][1] - approx[0][0][1])>50 and abs(approx[0][0][0] - approx[1][0][0])>100:                                                            # It is a rectangle if it has 4 corners
            l.append(approx)
            screen = approx
            break
    if screen is None:
        print("未检测到矩形车牌区域")
        exit()
    if debug:
        cv2.drawContours(img, l, -1, (0, 0, 255), 2)
        plt.imshow(img[:, :, ::-1])
        plt.show()
    return screen


def Character_segmentation(gray,mask,screen,debug=False):
    #plt.imshow(mask)
    #plt.show()

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped1 = gray[topx:bottomx + 1, topy:bottomy + 1]
    cropped = Affine_transformation(gray, screen)

    if debug:
        print(img.shape)
        #print(topx, bottomx, topy, bottomy)
        #plt.imshow(cropped1)
        #plt.show()
        cv2.imshow("Character_segmentation", cropped)
        cv2.imwrite("Character_segmentation.jpg", cropped)
    return cropped1

    #pts_src = np.array([pts[0][0], pts[1][0], pts[2][0], pts[3][0]])
def Affine_transformation(img, pts):
    pts_src = np.squeeze(pts)
    pts_dst = np.array([[0, 0], [600-1, 0], [600-1, 100-1], [0, 100-1]])

    matrix = cv2.getPerspectiveTransform(pts_src.astype(np.float32), pts_dst.astype(np.float32))
    result = cv2.warpPerspective(img, matrix, (600, 100))
    #result  = cv2.rotate(result , cv2.ROTATE_90_CLOCKWISE)
    result = cv2.flip(result, 1)
    return result

def ROI_reseach(gray,edged, debug=False):
    screen = License_plate_area(edged, debug)
    mask = np.zeros(gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    cropped = Character_segmentation(gray, mask, screen,debug=False)
    if debug:
        cv2.imshow("Mask License Plate", mask)
        cv2.imwrite("mask.jpg", mask)
        cv2.imshow("New License Plate", new_img)
        cv2.imwrite("New_image.jpg", new_img)
        cv2.imshow("Cropped License Plate", cropped)
        cv2.imwrite("Cropped.jpg", cropped)
    return cropped

def Character_enhancement(cropped,debug=False):
    ret, binary = cv2.threshold(cropped, 110, 255, cv2.THRESH_BINARY)
    binary = cv2.resize(binary, (600, 100))
    kernel = np.ones((3, 3), np.uint8)
    binaryerosion = cv2.erode(binary, kernel, iterations=1)
    binarydilation = cv2.dilate(binaryerosion, kernel, iterations=2)
    #binarydilation = cv2.bitwise_not(binarydilation)
    binarydilation = binarydilation[5:95,10:590]
    binarydilation = cv2.dilate(binarydilation, kernel, iterations=3)
    if debug:

        cv2.imshow("cropped", cropped)
        cv2.imwrite("cropped.jpg", cropped)
        cv2.imshow("Binary", binary)
        cv2.imwrite("Binary.jpg", binary)
        cv2.imshow("binaryerosion", binaryerosion)
        cv2.imwrite("Binaryerosion.jpg", binaryerosion)
        cv2.imshow("binarydilation", binarydilation[5:95,10:595])
        cv2.imwrite("Binarydilation.jpg", binarydilation[5:95,10:580])
    return binarydilation

def OCR(binarydilation,debug=False):
    # 定义边框参数
    top, bottom, left, right = 30, 30, 30, 30  # 增加白色边框
    binarydilation1 = cv2.copyMakeBorder(binarydilation, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        a = cv2.copyMakeBorder(binarydilation, 5, 5,5, 5, cv2.BORDER_CONSTANT, value=0)
        plt.imshow(a,cmap='gray')
        plt.title('binary')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        b = cv2.copyMakeBorder(binarydilation1, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)

        plt.imshow(b,cmap='gray')
        plt.title('binary++')
        plt.axis('off')
        plt.savefig('Contours.png', bbox_inches='tight')
        plt.show()
    custom_config = r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789 --psm 11'
    LP = pytesseract.image_to_string(binarydilation1, config=custom_config)
    print("Detecting License Plate:", LP[:-2])
    end = timeit.timeit()
    print("Operation time : ", end - start)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Plate_Recognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = Image_feature_enhancement(img,debug=True) # 增强图像特征，最终得到合适的边缘特征
    cropped = ROI_reseach(gray,edged,debug=True)  # 根据边缘特征，获得轮廓特征，寻找到矩形车牌位置
    binarydilation = Character_enhancement(cropped,debug=True)   # 将ROI进行增强，更容易识别
    OCR(binarydilation,debug=False) #获得车牌

if __name__ == '__main__':

    img = cv2.imread("dfn.jpg")
    Plate_Recognition(img)



