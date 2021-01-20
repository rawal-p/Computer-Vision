import numpy as np
import cv2
from horizontal_seam import horizontal_seam
from remove_horizontal_seam import remove_horizontal_seam

def shrink(I, num_rows_removed, num_cols_removed):
	temp = I
	

	#print('shrink() image size: ', temp.shape)
	

	if num_rows_removed > 0:

		for i in range(num_rows_removed):
			S = horizontal_seam(temp)
			#print(f'seam {i} created : ', temp.shape)

			temp = remove_horizontal_seam(temp,S)
			

			print(f'horizontal seam{i} removed')
	

	if num_cols_removed > 0:

		temp = np.transpose(temp, (1,0,2))

		for i in range(num_cols_removed):
			S = horizontal_seam(temp)
			temp = remove_horizontal_seam(temp,S)
			print(f'vertical seam{i} removed')
		temp = np.transpose(temp, (1,0,2))
	J = temp
	return J