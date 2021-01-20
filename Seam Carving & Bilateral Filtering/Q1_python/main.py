import numpy as np
import cv2
import matplotlib.pyplot as plt
from imenergy import imenergy
from horizontal_seam import horizontal_seam
from remove_horizontal_seam import remove_horizontal_seam
from shrink import shrink

msg = 'Which Experiment would you like to run? \nAcceptable values are [1,4]\n'
experiment = int(input(msg))

while True:

	while (experiment > 4) or (experiment < 1) :
		print('Invalid input, input should be in the range [1,4]')
		experiment = int(input(msg))

	if (experiment == 1):

		print('Experiment 1 in progress...')

		img = cv2.imread('union-terrace.jpg',1)

		experiment1a = shrink(img, 0, 100)	
		cv2.imwrite('E1a.jpg',experiment1a)

		print('Experiment 1A completed, check E1a.jpg for results')

		experiment1b = shrink(img, 100, 0)
		cv2.imwrite('E1b.jpg',experiment1b)

		print('Experiment 1B completed, check E1b.jpg for results')

		experiment1c = shrink(img,100, 100)
		cv2.imwrite('E1c.jpg',experiment1c)

		print('Experiment 1C completed, check E1c.jpg for results\nExperiment 1 completed')

	elif (experiment == 2):

		print('Experiment 2 in progress...')

		img = cv2.imread('union-terrace.jpg',1)

		seam_h = horizontal_seam(img)
		seam_v = horizontal_seam(np.transpose(img, (1,0,2)))

		img = plt.imread('union-terrace.jpg')
		fig, ax = plt.subplots()
		ax.imshow(img)
		ax.plot(range(seam_h.shape[1]),np.transpose(seam_h), '-', linewidth=3, color='red')
		ax.plot(np.transpose(seam_v), range(seam_v.shape[1]),'-', linewidth=3, color='red')
		plt.savefig('E2.jpg')
		plt.show()

		print('Experiment 2 completed, check E2.jpg for results')

	elif (experiment == 3):

		print('Experiment 3 in progress...')

		img = cv2.imread('daft_punk1.jpg',1)

		cv2.imshow('img',img)
		cv2.waitKey(0) & 0xff
		cv2.destroyAllWindows()	

		experiment3 = shrink(img, 60, 150)
		cv2.imwrite('E3.jpg',experiment3)

		print('Experiment 3 completed, check E3.jpg for results')


	elif (experiment == 4):

		print('Experiment 4 in progress...')

		img = cv2.imread('daft_punk2.jpg',1)

		cv2.imshow('img',img)
		cv2.waitKey(0) & 0xff
		cv2.destroyAllWindows()	

		experiment4 = shrink(img, 60, 150)
		cv2.imwrite('E4.jpg',experiment4)

		print('Experiment 4 completed, check E4.jpg for results')

	msg2 = print('Would you liked to run another experiment (y/n)?')
	rerun = input(msg2)

	if rerun == "y":
		msg = 'Which Experiment would you like to run? \nAcceptable values are [1,4]\n'
		experiment = int(input(msg))

	else:
		break
