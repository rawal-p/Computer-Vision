import numpy as np
import cv2
from imenergy import imenergy
from horizontal_seam import horizontal_seam
from remove_horizontal_seam import remove_horizontal_seam
from shrink import shrink



img_original = cv2.imread('union-terrace.jpg',1)
#norm = cv2.normalize(img_original.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

E_map = imenergy(img_original)


#print(E_map[328])
#S = horizontal_seam(img_original)

cv2.imshow('img_original',img_original)
cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()

cv2.imshow('E_map',E_map)
cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()