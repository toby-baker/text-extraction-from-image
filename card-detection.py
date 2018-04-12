# Author: Toby Baker
# Date: 27 Mar 2018
# Title: Text Extraction From Images of UC ID Cards

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import pytesseract

img = cv2.imread("TobyIDOut.jpg")
img = cv2.resize(img, (1000,600))

lower_green = np.array([100,100,100])
upper_green = np.array([255,255,255])
mask = cv2.inRange(img, lower_green, upper_green)
img = cv2.bitwise_and(img, img, mask=mask)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
mask = cv2.inRange(img_blur, 100, 255)

gradX = cv2.Sobel(mask, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)   
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

while True:
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()    