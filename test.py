import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import imutils
import timeit

pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
start = timeit.timeit()

img = cv2.imread("sample3.jpg")
# Image\\34 LZ 6622.jpg
# Pre-processing

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to gray scale image
'''cv2.imshow("Original License Plate",img)
cv2.imwrite("Original_image.jpg",img)
cv2.imshow("Gray License Plate",gray)
cv2.imwrite("Gray-scale.jpg",gray)'''

filtered = cv2.bilateralFilter(gray, 5, 75, 75)
filtered1 = cv2.GaussianBlur(gray, (5, 5), 0,
                             5)  # Adds a blur effect to remove unnecessary edges in the image (noise reduction)
# cv2.imshow("Filtered License Plate",filtered)
# cv2.imwrite("Filtered.jpg",filtered)
'''plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(gray,cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(filtered,cmap='gray')
plt.title('bilateralFilter')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(filtered1,cmap='gray')
plt.title('GaussianBlur')
plt.axis('off')
plt.savefig('filter.png', bbox_inches='tight')
plt.show()'''
# histogram_e = cv2.equalizeHist(filtered)                                            # Improved image with histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
histogram_e = clahe.apply(filtered)
'''plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(gray,cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(histogram_e,cmap='gray')
plt.title('Histogram_Esikleme')
plt.axis('off')
plt.savefig('histogram_e.png', bbox_inches='tight')
plt.show()
'''

# cv2.imshow("histogram eşikleme",histogram_e)
# cv2.imwrite("Histogram_Esikleme.jpg",histogram_e)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # A 5 by 5 matrix of 1
morphology = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel,
                              iterations=15)  # applying the kernel to the image 15 times
# 定义腐蚀核
# kernel = np.ones((5, 5), np.uint8)

# morphology = cv2.erode(morphology, kernel, iterations=1)
'''cv2.imshow("Morphology",morphology)
cv2.imwrite("Morphology.jpg",morphology)
'''
gcikarilmisresim = cv2.subtract(histogram_e,
                                morphology)  # 15 times kernel applied image is removed from the histogram and the plate region is brought to the fore
# gcikarilmisresim[gcikarilmisresim>125] = 255

# gcikarilmisresim[gcikarilmisresim<50] = 0
'''plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(histogram_e,cmap='gray')
plt.title('Histogram_Esikleme')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(morphology,cmap='gray')
plt.title('morphology')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(gcikarilmisresim,cmap='gray')
plt.title('Initial ROI extraction')
plt.axis('off')
plt.savefig('morp.png', bbox_inches='tight')
plt.show()
'''
'''cv2.imshow("goruntuden cikarilmis resim",gcikarilmisresim)
cv2.imwrite("Cikarilmisresim.jpg",gcikarilmisresim)
'''

edged = cv2.Canny(gcikarilmisresim, 30, 200)  # Edges are detected with Canny edge detection.
contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finds edges
'''cv2.imshow("Egded License Plate",edged)
cv2.imwrite("Edges.jpg",edged)'''
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sobel = cv2.convertScaleAbs(sobel)
'''plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(gray,cmap='gray')
plt.title('Origin')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(sobel,cmap='gray')
plt.title('Sobel')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(edged,cmap='gray')
plt.title('Canny')
plt.axis('off')
plt.savefig('egde.png', bbox_inches='tight')
plt.show()'''

rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated_image = cv2.dilate(edged, rectangular_kernel, iterations=1)
'''plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(edged,cmap='gray')
plt.title('Canny')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(dilated_image,cmap='gray')
plt.title('dilated')
plt.axis('off')
plt.savefig('dilated.png', bbox_inches='tight')
plt.show()'''
'''cv2.imshow("add Egded",dilated_image)
cv2.imwrite("add_Edges.jpg",dilated_image)'''
# cv2.waitKey(0)
# cv2.destroyAllWindows()
contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finds edges
cnts = imutils.grab_contours(contours)  # catch, grab the countours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:100]  # The found edges are sorted by their area.
screen = None
i_c = np.copy(img)
cv2.drawContours(i_c, cnts, -1, (0, 0, 255), 2)
#plt.imshow(i_c[:, :, ::-1], cmap='gray')
#plt.show()

'''plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img[:,:,::-1])
plt.title('image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(i_c[:,:,::-1])
plt.title('Contours')
plt.axis('off')
plt.savefig('Contours.png', bbox_inches='tight')
plt.show()'''

l = []
for c in cnts:
    epsilon = 0.018 * cv2.arcLength(c,
                                    True)  # finds the arc length of the contours with an error of 0.018 (approximately)
    approx = cv2.approxPolyDP(c, epsilon, True)  # It serves to form the rectangle in the plate region properly.
    print("-----------------------")
    print(approx)
    if len(approx) == 4 #and (approx[3][0][1] - approx[0][0][1])>20 and (approx[0][0][0] - approx[1][0][0])>60:                                                            # It is a rectangle if it has 4 corners
        l.append(approx)
        screen = approx
        break
mask = np.zeros(gray.shape, np.uint8)  # creates a black screen with the same dimensions of the gray format image
new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255),
                           -1)  # makes the plate region part of the resulting black screen white
new_img = cv2.bitwise_and(img, img, mask=mask)  # The white region of the plate region with the original image is summed with and

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(i_c[:,:,::-1])
plt.title('Contours')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(new_img[:,:,::-1])
plt.title('plate')
plt.axis('off')
plt.savefig('Contours.png', bbox_inches='tight')
plt.show()