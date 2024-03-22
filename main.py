import cv2
import numpy as np

frame = cv2.imread('low_bright.jpeg')

vid = cv2.VideoCapture(0)

def process(frame):
    threshold1 = 50
    threshold2 = 100
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame,(3,3), sigmaX=0, sigmaY=0)
    sobelx = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    edges = cv2.Canny(image=blur, threshold1=50, threshold2=100) 
    return edges

while (True):
    result, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imshow('Edge', process(frame))
    process(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()



def process(frame):
    threshold1 = 50
    threshold2 = 100
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame,(3,3), sigmaX=0, sigmaY=0)
    sobelx = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    edges = cv2.Canny(image=blur, threshold1=50, threshold2=100) 
    cv2.imshow('Edge', edges)
    cv2.waitKey(0)

cv2.destroyAllWindows()