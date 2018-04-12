# Author: Toby Baker
# Date: 27 Mar 2018
# Title: Text Extraction From Images of UC ID Cards

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import pytesseract

# cv2.imwrite("TobyIDNew.bmp", img) #saves image to folder

# cap = cv2.VideoCapture(0) # Load ID Photo

img = cv2.imread("TobyIDClean.jpg")
img = cv2.resize(img, (1000,600))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blur = cv2.GaussianBlur(gray,(7,7),0)

mask = cv2.inRange(img_blur, np.array([100,100,100]), np.array([255,255,255]))
img_mask = cv2.bitwise_and(img_blur, img_blur, mask=mask)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

BoxList = []

gradX = cv2.Sobel(mask, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)   
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and use the contour to
	# compute the aspect ratio and coverage ratio of the bounding box
	# width to the width of the image
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	
	if (w > 70 and h > 25 and ar > 4 and y>275 and y<380):
		# pad the bounding box since we applied erosions and now need
		# to re-grow it
		pX = int((x + w) * 0.03)
		pY = int((y + h) * 0.03)
		(x, y) = (x - pX, y - pY)
		(w, h) = (w + (pX * 2), h + (pY * 2))
    
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		rectangle = [x,y,w,h]
		BoxList.append(rectangle)
		
while True:
	cv2.imshow("Image", mask)
    
	# Close the script when q is pressed.
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()

for rect in BoxList:
	x=  rect[0]
	y = rect[1]
	w = rect[2]
	h = rect[3]
	
	textImg = gray[y:y+h,x:x+w]
	#textImg = cv2.resize(textImg,100,50)
	name = pytesseract.image_to_string(textImg) #perform OCR on the name
	print(name)

#name = pytesseract.image_to_string(name) #perform OCR on the name
#user = pytesseract.image_to_string(user) #perform OCR on the user code
#id_num = pytesseract.image_to_string(id_num) #perform OCR on the ID Number

#print(name)
#print(user)
#print(id_num)