# -- Version --
# Python 3.12.2
# opencv2 4.9.0.80

import cv2
import numpy as np
from typing import Iterable, List, Tuple, Union
import matplotlib.pyplot as plt

# frame = cv2.imread('low_bright.jpeg')

vid = cv2.VideoCapture(0)

def Adjust(img, brightness=0 , contrast=127):
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
def process(frame, threshold1=50, threshold2=100):
	if (threshold1 > threshold2):
		threshold1 = 50
		threshold2 = 100
	
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(frame,(3,3), sigmaX=0, sigmaY=0)
	sobelx = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
	sobely = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
	sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
	blur = blur.astype(np.uint8)
	edges = cv2.Canny(image=blur, threshold1=threshold1, threshold2=threshold2)
	return edges

def convert(image : np.ndarray, output_path : str, plot_dict : dict = {"color" : "k", "linewidth" : 2.0}, default_height : float = 8) -> List[np.ndarray]:
	contour_tuple = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	contours = contour_tuple[0]
	rings = [np.array(c).reshape([-1, 2]) for c in contours]

	max_x, max_y, min_x, min_y = 0, 0, 0, 0
	for ring in rings:
		max_x = max(max_x, ring.max(axis=0)[0])
		max_y = max(max_y, ring.max(axis=0)[1])
		min_x = max(min_x, ring.min(axis=0)[0])
		min_y = max(min_y, ring.min(axis=0)[1])
	
	for _, ring in enumerate(rings):
		close_ring = np.vstack((ring, ring[0]))
		xx = close_ring[..., 0]
		yy = max_y - close_ring[..., 1]
		plt.plot(xx, yy, **plot_dict)
	
	plt.axis("off")
	plt.savefig(output_path)

cv2.namedWindow('Control Panel') 
cv2.createTrackbar('Threshold 1', 'Control Panel', 50, 200, process)
cv2.createTrackbar('Threshold 2', 'Control Panel', 100, 300, process)
cv2.createTrackbar('Brightness', 'Control Panel', 255, 2 * 255, Adjust)
cv2.createTrackbar('Contrast', 'Control Panel', 127, 2 * 127, Adjust)

while (True):
	result, frame = vid.read()
	brightness = cv2.getTrackbarPos('Brightness', 'Control Panel')
	contrast = cv2.getTrackbarPos('Contrast', 'Control Panel')
	frame = Adjust(frame, brightness, contrast)
	cv2.imshow('frame', frame)

	t1 = cv2.getTrackbarPos('Threshold 1', 'Control Panel')
	t2 = cv2.getTrackbarPos('Threshold 2', 'Control Panel')
	edged = process(frame, t1, t2)
	cv2.imshow('Edge', edged)

	if cv2.waitKey(1) == ord('s'):
		convert(edged, output_path="saved.svg")
	elif cv2.waitKey(1) == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
