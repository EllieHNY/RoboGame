import cv2
import numpy as np
# frame = cv2.imread('low_bright.jpeg')

vid = cv2.VideoCapture(0)

def Adjust(img, brightness=0):
	brightness = cv2.getTrackbarPos('Brightness', 'Control Panel')
	contrast = cv2.getTrackbarPos('Contrast', 'Control Panel')
	effect = controller(img, brightness, contrast)
	return effect

def controller(img, brightness=255, contrast=127): 
	brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
	contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
	
	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			max = 255
		else:
			shadow = 0
			max = 255 + brightness
		
		al_pha = (max - shadow) / 255
		ga_mma = shadow 
		cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
	else:
		cal = img 
	
	if contrast != 0: 
		Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
		Gamma = 127 * (1 - Alpha)
		cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)  
	
	return cal 

#create edge image
def process(frame):
	t1 = cv2.getTrackbarPos('Threshold 1', 'Control Panel')
	t2 = cv2.getTrackbarPos('Threshold 2', 'Control Panel') 
	if (t1 > t2):
		t1 = 50
		t2 = 100
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(frame,(3,3), sigmaX=0, sigmaY=0)
	sobelx = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
	sobely = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
	sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
	edges = cv2.Canny(image=blur, threshold1=t1, threshold2=t2)
	return edges

cv2.namedWindow('Control Panel') 
cv2.createTrackbar('Threshold 1', 'Control Panel', 50, 200, process)
cv2.createTrackbar('Threshold 2', 'Control Panel', 100, 300, process)
cv2.createTrackbar('Brightness', 'Control Panel', 255, 2 * 255, Adjust)
cv2.createTrackbar('Contrast', 'Control Panel', 127, 2 * 127, Adjust)

while (True):
	result, frame = vid.read()
	frame = Adjust(frame, 0)
	cv2.imshow('frame', frame)
	cv2.imshow('Edge', process(frame))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
