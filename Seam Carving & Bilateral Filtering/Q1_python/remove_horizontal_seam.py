import numpy as np
import cv2 as cv2
from imenergy import imenergy

def remove_horizontal_seam(I,S):
	

	#E = imenergy(I)

	rows,cols,depth = I.shape
	temp = np.zeros((rows-1,cols,depth))

	for k in range(3):

		for i in range(cols):

			ind = S[0,i]

			for j in range(ind,rows - 1):

				I[j,i,k] = I[j+1,i,k]


		temp[:,:,k] = I[:rows - 1, :cols, k]
	
	J = temp.astype(np.float32)
	return 	J	
