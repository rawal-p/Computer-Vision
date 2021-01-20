import numpy as np
import cv2 as cv2
from imenergy import imenergy

def horizontal_seam(I):

	#print('horizontal_seam() image size: ', I.shape)
	

	E = imenergy(I)
	M = np.zeros((E.shape))

	M[:,0] = E[:,0]
	rows,cols = E.shape

	inf = float("inf")

	for i in range(1,cols):
		for j in range(rows):
			backtrack = [inf, inf, inf]
			if j == 0:
				pass
			else:
				backtrack[0] = M[j-1,i-1]
			backtrack[1] = M[j,i-1]
			if (j == rows-1) or (i == cols-1):
				pass
			else:
				backtrack[2] = M[j+1,i-1]
			#print(f'backtrack_{j}: ',backtrack)
			M[j,i] = E[j,i] + np.amin(backtrack)
	np.savetxt('mmap.csv', M, delimiter=',')		
	Stemp = np.zeros((1,cols))

	j = np.argmin(M[:,cols-1])


	#print(f'j_init: ', j,f'M[{j},{cols-1}]: ', M[j,cols-1])


	for i in range(cols):
		back = cols - 1 - i
		Stemp[0,back] = j

		#print(f'back: ',back)
		#print(f'Stemp[0,{back}]: ',Stemp[0,back])


		if (j == 0) and (back != 0):
			temp = [inf,M[j,back-1],M[j+1,back-1]]
			x = np.argmin(temp)

			#print(f'temp_1_{i}: ',temp)

			if x == 2:
				j = j

			if x == 3:
				j = j+1

		elif (j == rows-1) and (back != 0):
			temp = [M[j-1,back-1],M[j,back-1],inf]
			x = np.argmin(temp)

			#print(f'temp_2_{i}: ',temp)

			if x == 1:
				j = j-1

			if x == 2:
				j = j

		elif back != 0:
			temp = [M[j-1,back-1],M[j,back-1],M[j+1,back-1]]
			x = np.argmin(temp)

			#print(f'temp_3_{i}: ',temp)

			if x == 0:
				j = j-1

			if x == 1:
				j = j

			if x == 2:
				j = j+1
	np.savetxt('S.csv', Stemp, delimiter=',')
	S = np.int_(Stemp)
	np.savetxt('S_int.csv', S, delimiter=',')
	return S
