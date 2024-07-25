import cv2
import numpy as np
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 5, 75,75)
    histogram_e = cv2.equalizeHist(filtered)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphology = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel,iterations=15)
    gcikarilmisresim = cv2.subtract(histogram_e,morphology)
    edged = cv2.Canny(gcikarilmisresim, 30, 250)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screen = None

    for c in cnts:
        epsilon = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            screen = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_cap = cv2.drawContours(mask, [approx], 0, (255, 255, 255), -1)
    new_cap = cv2.bitwise_and(frame,frame,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    ret, binary = cv2.threshold(cropped, 110, 255,cv2.THRESH_BINARY)


    kernel = np.ones((3, 3), np.uint8)
    binaryerosion = cv2.erode(binary, kernel, iterations=1)
    binaryopening = cv2.dilate(binaryerosion, kernel, iterations=1)
    cv2.imshow("Original License Plate", frame)
    cv2.imshow("Cropped License Plate", cropped)
    custom_config = r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 11'
    LP = pytesseract.image_to_string(cropped, config=custom_config)
    print("detecting the license plate: ", LP[:-2])
    if cv2.waitKey(500) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()