import numpy as np
import cv2 as cv2
from PIL import Image

def imenergy(RGB_img):

	blur = cv2.bilateralFilter(RGB_img,9,75,75)
	blur = cv2.medianBlur(blur,3)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	gx = cv2.Sobel(norm,cv2.CV_64F,1,0,ksize=3)
	gy = cv2.Sobel(norm,cv2.CV_64F,0,1,ksize=3)
	grad_mag = cv2.addWeighted(np.absolute(gx), 0.5, np.absolute(gy), 0.5, 0)
	grad_mag = grad_mag.astype('float32')
	np.savetxt('emap.csv', grad_mag, delimiter=',')
	return grad_mag