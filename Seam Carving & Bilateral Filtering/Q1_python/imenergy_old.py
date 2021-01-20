import numpy as np
import cv2 as cv2

def imenergy(RGB_img):

	#print(type(RGB_img))
	#print('imenergy() image size: ', RGB_img.shape)
	


	gray = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)

	norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	gx = cv2.Sobel(norm,cv2.CV_64F,1,0,ksize=3)
	gy = cv2.Sobel(norm,cv2.CV_64F,0,1,ksize=3)
	grad_mag = cv2.addWeighted(np.absolute(gx), 0.5, np.absolute(gy), 0.5, 0)
	grad_mag = grad_mag.astype('float32')
	return grad_mag
