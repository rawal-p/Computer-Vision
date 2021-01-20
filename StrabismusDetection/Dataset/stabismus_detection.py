import numpy as np
import cv2
import imutils

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#Input Image Requirements:
# 1) Taken by a cellphone by either the front or rear camera 
# 2) Taken with flash 
# 3) Take approximately 2ft away and at the same level 
#    as the user's face (or arms fully extended)
# 4) Only one person in the frame

#Reading the input image
img_original = cv2.imread('front_cam_bright.jpg',1)

#Converting the input image to grayscale
gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)   #convert to grayscale

#Searching for faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

faces = faces[faces[:,2].argsort()]          #Sorting the detection based on width in ascending order

faces = [faces[-1,:]]                        #Keeping the detection with the largest width
                                             #assuming smaller sized detections are False Positives
for (x,y,w,h) in faces:
    
    cv2.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),7) #Drawing a blue rectanlge for the detected faces
    
    roi_gray = gray[y:y+h, x:x+w]                           #Cropping the detected face image from the grayscale input image
    
    roi_color = img_original[y:y+h, x:x+w]                  #Cropping the detected face image from the color input image  
    
    eyes = eye_cascade.detectMultiScale(roi_gray,1.3,3)     #Searching for eyes in cropped grayscale face image
    
    eyes = eyes[eyes[:,2]>=100, :]                          #Removing any detections with widths smaller than 150 pixels
                                                            #Reduces the possibility of false positives 
    
    eyes = eyes[eyes[:,0].argsort()]                        #Sorting the eye detections in ascending order based on the x values
                                                            #this allow for categorization of left and right eye
                                                            #small x value --> closer to the left side of the face image --> person's right eye
                                                            #large x value --> closer to the right side of the face image --> person's left eye
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),7) #Drawing a green rectanlge for the detected faces

#Cropping the left and right color images from the cropped face images 
eyeR_color = roi_color[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
eyeL_color = roi_color[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]]

#Cropping the left and right grayscale images from the cropped face images 
eyeR_gray = roi_gray[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
eyeL_gray = roi_gray[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]]

#Properties of the Eye Detection with Haar Cascades: (Experimentally gathered)
# 1) Width and height are equal, square in shape
# 2) For most cases the iris is relatively cenetered within the detected patch
# 3) The eye detections tend to include eyebrows
#   -> Eyebrows tend to cause issues with iris detection therefore, further cropping is required 
#      where only 70% of the center of the detected eye image is retained

#Cropping the center 70% of the left and right color eye images
eyeR_color = eyeR_color[np.int(np.floor(eyeR_color.shape[0]*0.15)):np.int(np.ceil(eyeR_color.shape[0]*0.85)), np.int(np.floor(eyeR_color.shape[1]*0.15)):np.int(np.ceil(eyeR_color.shape[1]*0.85))]
eyeL_color = eyeL_color[np.int(np.floor(eyeL_color.shape[0]*0.15)):np.int(np.ceil(eyeL_color.shape[0]*0.85)), np.int(np.floor(eyeL_color.shape[1]*0.15)):np.int(np.ceil(eyeL_color.shape[1]*0.85))]

#Cropping the center 70% of the left and right gray eye images
eyeR_gray = eyeR_gray[np.int(np.floor(eyeR_gray.shape[0]*0.15)):np.int(np.ceil(eyeR_gray.shape[0]*0.85)), np.int(np.floor(eyeR_gray.shape[1]*0.15)):np.int(np.ceil(eyeR_gray.shape[1]*0.85))]
eyeL_gray = eyeL_gray[np.int(np.floor(eyeL_gray.shape[0]*0.15)):np.int(np.ceil(eyeL_gray.shape[0]*0.85)), np.int(np.floor(eyeL_gray.shape[1]*0.15)):np.int(np.ceil(eyeL_gray.shape[1]*0.85))]

#Resizing the image to display in OpenCv (large image aren't displayed properly)
img = imutils.resize(img_original, width=700)

#cv2.imshow('face',img)              #Viewing the detected face and eyes 
#cv2.imwrite('face_&_eyes.jpg', img)
#cv2.waitKey(0) & 0xff
#cv2.destroyAllWindows() 

#cv2.imshow('right_eye',eyeR_color)              #Viewing the detected face and eyes 
#cv2.imshow('left_eye',eyeL_color)
#cv2.imwrite('right_eye.jpg', eyeR_color)
#cv2.imwrite('left_eye.jpg', eyeL_color)
#cv2.waitKey(0) & 0xff
#cv2.destroyAllWindows() 
########################COMPUTING_THE_CENTROID_OF_THE_EYE_IMAGES#########################
#
#Recall the 2nd property of the Haar Cascade from above, we know that the iris will be 
#somewhat centered in the cropped eye image. We can use this to our advantage to create  
#a mask which can be used to give us a better understanding of where the center of the 
#iris should be located in the cropped eye image. 
#
#Steps to be taken:
# 1)Sample the center of the cropped eye image to get an understanding of the color of 
#   iris in the image
#       -> 20 (5x5) patches from the center of the cropped eye images will be sampled 
#       -> the average color of each of these patches will be computed
# 2)Iterate through the cropped eye image to check if each pixel in the image has a
#   similar color as the iris, if so then add that pixel location to the mask
# 3)Upon completion of step 2, you should have a binary mask with all the pixels  
#   with similar color values as the iris. Then calculate the centroid of the mask
#   to get the approximate center of the iris


#Computing the coordinates of the center of the left and right cropped eye images 
dR = eyeR_color.shape
dR = np.array([int(dR[0]/2),int(dR[1]/2)])  #the center coordinates of the right eye

dL = eyeL_color.shape
dL = np.array([int(dL[0]/2),int(dL[1]/2)])  #the center coordinates of the left eye

#initializing the scaling array which will move the center of the sampling patch
#across the center of the cropped eye image
scaling = np.linspace(-0.1,0.1,20,endpoint = True)

#initializing the patch size
patch_size = 5

#intializing the arrays in which the average color of each of the 20 patches will be stored
patchR_avg_color = np.zeros((len(scaling),4))
patchL_avg_color = np.zeros((len(scaling),4))

ind = 0 #indexing variable

#eyeR2_color = eyeR_color
#eyeL2_color = eyeL_color


for i in scaling:
    #computing the vertical and horizontal offset required by offsetting it 
    #by the scaling factor in the scaling array
    dR_offset = np.int_(np.array([x * i for x in dR]))
    dL_offset = np.int_(np.array([x * i for x in dL]))
    
    #computing the patch center by adding the offsets computed previously
    patchR_center = np.array(dR+dR_offset)
    patchL_center = np.array(dL+dL_offset)

    #computing the coordinates of bottom left corner of the patch for the right eye 
    patchR_x = int(patchR_center[0] - patch_size / 2)
    patchR_y = int(patchR_center[1] - patch_size / 2)
    
    #computing the coordinates of bottom left corner of the patch for the left eye
    patchL_x = int(patchL_center[0] - patch_size / 2)
    patchL_y = int(patchL_center[1] - patch_size / 2)

    #cv2.rectangle(eyeR2_color,(patchR_x,patchR_y),(patchR_x+patch_size,patchR_y+patch_size),(255,0,255),1)  
    #cv2.rectangle(eyeL2_color,(patchL_x,patchL_y),(patchL_x+patch_size,patchL_y+patch_size),(255,0,255),1)

    #if (ind == 0) or (ind == len(scaling)-1):
    #    cv2.rectangle(eyeR2_color,(patchR_x,patchR_y),(patchR_x+patch_size,patchR_y+patch_size),(255,0,255),1)  
    #    cv2.rectangle(eyeL2_color,(patchL_x,patchL_y),(patchL_x+patch_size,patchL_y+patch_size),(255,0,255),1)
    #    cv2.imshow('patches_right_eye',eyeR2_color)              #Viewing the detected face and eyes 
    #    cv2.imwrite((f'{ind+1}_patch_right_eye.jpg'), eyeR2_color)
    #    cv2.waitKey(0) & 0xff
    #    cv2.destroyAllWindows() 

    #    cv2.imshow('patches_left_eye',eyeL2_color)              #Viewing the detected face and eyes 
    #    cv2.imwrite((f'{ind+1}_patch_left_eye.jpg'), eyeL2_color)
    #    cv2.waitKey(0) & 0xff
    #    cv2.destroyAllWindows() 

    #cropping the patch image from the eye image
    patchR_image = eyeR_color[patchR_x:patchR_x+patch_size, patchR_y:patchR_y+patch_size]
    patchL_image = eyeL_color[patchL_x:patchL_x+patch_size, patchL_y:patchL_y+patch_size]

    #computing the average RBG values for the patch and saving it for both eyes 
    patchR_avg_color[ind,:3] = np.average(np.average(patchR_image, axis=0), axis=0)
    patchL_avg_color[ind,:3] = np.average(np.average(patchL_image, axis=0), axis=0)

    ind += 1


#Calculating the intensity color intensity for each of the 20 patches from the last step
for i in range(len(scaling)):

    patchR_avg_color[i,3] = np.sqrt(np.sum(np.square(patchR_avg_color[i])))
    patchL_avg_color[i,3] = np.sqrt(np.sum(np.square(patchL_avg_color[i])))

#Sorting the average color of the patches in ascending order based on intensity
patchR_avg_color = patchR_avg_color[patchR_avg_color[:,3].argsort()]
patchL_avg_color = patchL_avg_color[patchL_avg_color[:,3].argsort()]

#Computing the median intensity of the sorted list
patchR_median_intensity = np.median(patchR_avg_color[:,3])
patchL_median_intensity = np.median(patchL_avg_color[:,3])

#Removing all patches with intensity value greater than the median intensity 
#This must be done because the original image taken by the user is assumed to be taken 
#with flash, therefore, there will be patches in the iris with a very high intensities
#and these patches should not be considered when trying to assess the color of the iris
patchR_avg_color=patchR_avg_color[patchR_avg_color[:,3]<= patchR_median_intensity, :]
patchL_avg_color=patchL_avg_color[patchL_avg_color[:,3]<= patchL_median_intensity, :]



#Removing the color intensity column from the patch average color array
patchR_avg_color = patchR_avg_color[:,:3]
patchL_avg_color = patchL_avg_color[:,:3]

#Computing the average RGB values from the patches most representative of the iris
avg_colorR = np.average(patchR_avg_color, axis=0)
avg_colorL = np.average(patchL_avg_color, axis=0)

#Computing the higher and lower limits of the average color 
#by adding +/- 35% tolerance to the average color
avg_colorR_h = np.int_(np.ceil(avg_colorR*1.35))
avg_colorR_l = np.int_(np.floor(avg_colorR - avg_colorR*0.35))

avg_colorL_h = np.int_(np.ceil(avg_colorL*1.35))
avg_colorL_l = np.int_(np.floor(avg_colorL - avg_colorL*0.35))

#initializing the mask for each to be the same x and y dimension as the cropped eye images
maskR = np.zeros((eyeR_color.shape[0],eyeR_color.shape[1]))
maskL = np.zeros((eyeL_color.shape[0],eyeL_color.shape[1]))

#iterating through the cropped eye image to check if pixels in the 
#eye image fall within the range that defines iris color
for i in range(eyeR_color.shape[0]):
    for j in range(eyeR_color.shape[1]):
        if (eyeR_color[i,j] >= avg_colorR_l).all():
            if (eyeR_color[i,j] <= avg_colorR_h).all():
                maskR[i,j] = 255  
        
for i in range(eyeL_color.shape[0]):
    for j in range(eyeL_color.shape[1]):
        if (eyeL_color[i,j] >= avg_colorL_l).all():
            if (eyeL_color[i,j] <= avg_colorL_h).all():
                maskL[i,j] = 255

            
# computing the centroid of the masks
MR = cv2.moments(maskR)
ML = cv2.moments(maskL)

cRX = int(MR["m10"] / MR["m00"])
cRY = int(MR["m01"] / MR["m00"])
cR = np.array([cRX,cRY])        #coordinates of the iris centroid of the right eye

cLX = int(ML["m10"] / ML["m00"])
cLY = int(ML["m01"] / ML["m00"])
cL = np.array([cLX,cLY])        #coordinates of the iris centroid of the left eye 

#cv2.imshow('maskR',maskR)      #viewing the masks for debug purpose
#cv2.imshow('maskL',maskL)
#cv2.imwrite('right_eye_mask.jpg', maskR)
#cv2.imwrite('left_eye_mask.jpg', maskL)
#cv2.waitKey(0) & 0xff
#cv2.destroyAllWindows()  

####################################Circle_Detection#####################################
#
#In this section we will be using the HoughCircles function provided in the OpenCV 
#library to detect the exact location and radius of the iris and the flash. 
#Some preprocessing steps must be taken to improve the performance of the HoughCircles.
# 1) Binary Thresholding
# 2) Morphological transformations (Opening)
# 3) Running HoughCircles with lax constraints
# 4) As a result of the lax constraints we will get many candidate circle detections.
#    Therefore, we will choose the circle with closest to the iris centroid we computed earlier

#Thresholding the grayscale eye images
#the threshold value was determined experimentally, there is no particular reason for the 
#chosen formulation, it just happens to work best (probably room for improvement)

threshold_R = int(np.average([patchR_median_intensity, np.median(eyeR_gray)]))
threshold_L = int(np.average([patchL_median_intensity, np.median(eyeR_gray)]))
threshold = np.minimum(threshold_R,threshold_L)


__,eyeR_bin = cv2.threshold(eyeR_gray,threshold,255,cv2.THRESH_BINARY)
__,eyeL_bin = cv2.threshold(eyeL_gray,threshold,255,cv2.THRESH_BINARY)

#Applying Opening transformation on the thresholded images
kernel = np.ones((3,3), np.uint8) 
eyeR_bin_open = cv2.morphologyEx(eyeR_bin, cv2.MORPH_OPEN, kernel) 
eyeL_bin_open = cv2.morphologyEx(eyeL_bin, cv2.MORPH_OPEN, kernel)


#cv2.imshow('eyeR_bin_open',eyeR_bin_open)
#cv2.imshow('eyeL_bin_open',eyeL_bin_open)
#cv2.imwrite('processed_right_eye.jpg',eyeR_bin_open)
#cv2.imwrite('processed_left_eye.jpg',eyeL_bin_open)
#cv2.waitKey(0) & 0xff
#cv2.destroyAllWindows()


#Setting the max radius for the flash which should appear as a pinpoint white light in the iris
#set as 3% of the width of the cropped eye image
flash_max_rad = int(np.ceil((np.amin([eyeR_bin_open.shape[0],eyeL_bin_open.shape[0]]))*0.05))

#Setting the minimum radius of the iris as 15% of cropped eye image
iris_min_rad = int(np.floor((np.amin([eyeR_bin_open.shape[0],eyeL_bin_open.shape[0]]))*0.15))
iris_max_rad = int(np.floor((np.amin([eyeR_bin_open.shape[0],eyeL_bin_open.shape[0]]))*0.6))
#Running HoughCircles
circlesR_flash = cv2.HoughCircles(eyeR_bin_open,cv2.HOUGH_GRADIENT,1,10,param1=10,param2=5,maxRadius=flash_max_rad)
circlesL_flash = cv2.HoughCircles(eyeL_bin_open,cv2.HOUGH_GRADIENT,1,10,param1=10,param2=5,maxRadius=flash_max_rad)
circlesR_iris = cv2.HoughCircles(eyeR_bin_open,cv2.HOUGH_GRADIENT,1,30,param1=10,param2=10,minRadius=iris_min_rad,maxRadius=iris_max_rad)             
circlesL_iris = cv2.HoughCircles(eyeL_bin_open,cv2.HOUGH_GRADIENT,1,30,param1=10,param2=10,minRadius=iris_min_rad,maxRadius=iris_max_rad)


#print('circlesR_flash: ', circlesR_flash)
#print('circlesL_flash: ', circlesL_flash)
#print('circlesR_iris: ', circlesR_iris)
#print('circlesL_iris: ', circlesL_iris)

#Coverting the datatype to uint16
circlesR_flash = np.uint16(np.around(circlesR_flash))
circlesL_flash = np.uint16(np.around(circlesL_flash))
circlesR_iris = np.uint16(np.around(circlesR_iris))
circlesL_iris = np.uint16(np.around(circlesL_iris))

#Compute the Eclidean distances from the candidate flash circles to the centroid 
#initalize the two lists that will store the distances of the candidate flash circles from 
#the centroid for the two eyes
dist_flash_list1 = []
dist_flash_list2 = []
for i in range(circlesR_flash.shape[1]):
    dist = np.linalg.norm(cR-circlesR_flash[0,i,0:2])
    #print('distance1 =',dist)
    dist_flash_list1.append(dist)

for i in range(circlesL_flash.shape[1]):
    dist = np.linalg.norm(cL-circlesL_flash[0,i,0:2])
    #print('distance2 =',dist)
    dist_flash_list2.append(dist)


#Compute the Eclidean distances from the candidate iris circles to the centroid
#initalize the two lists that will store the distances of the candidate flash circles from 
#the centroid for the two eyes
dist_iris_list1 = []
dist_iris_list2 = []
for i in range(circlesR_iris.shape[1]):
    dist = np.linalg.norm(cR-circlesR_iris[0,i,0:2])
    #print('distance1 =',dist)
    dist_iris_list1.append(dist)

for i in range(circlesL_iris.shape[1]):
    dist = np.linalg.norm(cL-circlesL_iris[0,i,0:2])
    #print('distance2 =',dist)
    dist_iris_list2.append(dist)

#Retrieving the minimum index in the distance arrays
index_flash_minR = np.argmin(dist_flash_list1)
index_flash_minL = np.argmin(dist_flash_list2)
index_iris_minR = np.argmin(dist_iris_list1)
index_iris_minL = np.argmin(dist_iris_list2)


#Select the flash circle for the right eye
circlesR_flash_sel = circlesR_flash[0,index_flash_minR,:]
#print('circlesR_flash_sel: ', circlesR_flash_sel)
# draw the outer circle
cv2.circle(eyeR_color,(circlesR_flash_sel[0],circlesR_flash_sel[1]),circlesR_flash_sel[2],(255,255,0),3)
# draw the center of the circle
#cv2.circle(eyeR_color,(circlesR_flash_sel[0],circlesR_flash_sel[1]),1,(0,0,255),3)


#Select the flash circle for the left eye
circlesL_flash_sel = circlesL_flash[0,index_flash_minL,:]
#print('circlesL_flash_sel: ', circlesL_flash_sel)
# draw the outer circle
cv2.circle(eyeL_color,(circlesL_flash_sel[0],circlesL_flash_sel[1]),circlesL_flash_sel[2],(255,255,0),3)
# draw the center of the circle
#cv2.circle(eyeL_color,(circlesL_flash_sel[0],circlesL_flash_sel[1]),1,(0,0,255),3)


#Select the iris circle for the right eye
circlesR_iris_sel = circlesR_iris[0,index_iris_minR,:]
#print('circlesR_iris_sel: ', circlesR_iris_sel)
# draw the outer circle
cv2.circle(eyeR_color,(circlesR_iris_sel[0],circlesR_iris_sel[1]),circlesR_iris_sel[2],(0,255,255),2)
# draw the center of the circle
#cv2.circle(eyeR_color,(circlesR_iris_sel[0],circlesR_iris_sel[1]),1,(0,0,255),3)


#Select the iris circle for the left eye
circlesL_iris_sel = circlesL_iris[0,index_iris_minL,:]
#print('circlesL_iris_sel: ', circlesL_iris_sel)
# draw the outer circle
cv2.circle(eyeL_color,(circlesL_iris_sel[0],circlesL_iris_sel[1]),circlesL_iris_sel[2],(0,255,255),2)
# draw the center of the circle
#cv2.circle(eyeL_color,(circlesL_iris_sel[0],circlesL_iris_sel[1]),1,(0,0,255),3)


###########################ALL_THE_CANDIDATE_CIRCLE_DETECTIONS###########################
#Plotting all the candidate circles from the 
#
#
#for i in circlesR_flash[0,:]:
    # draw the outer circle
#    cv2.circle(eyeR_color,(i[0],i[1]),i[2],(255,255,0),2)
    # draw the center of the circle
#    cv2.circle(eyeR_color,(i[0],i[1]),2,(0,0,255),3)

#for j in circlesL_flash[0,:]:
    # draw the outer circle
#    cv2.circle(eyeL_color,(j[0],j[1]),j[2],(255,255,0),2)
    # draw the center of the circle
#    cv2.circle(eyeL_color,(j[0],j[1]),2,(0,0,255),3)

#for i in circlesR_iris[0,:]:
    # draw the outer circle
#    cv2.circle(eyeR_color,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
#    cv2.circle(eyeR_color,(i[0],i[1]),2,(0,0,255),3)

#for j in circlesL_iris[0,:]:
    # draw the outer circle
#    cv2.circle(eyeL_color,(j[0],j[1]),j[2],(0,255,0),2)
    # draw the center of the circle
#    cv2.circle(eyeL_color,(j[0],j[1]),2,(0,0,255),3)


#Displaying the cropped eye images with the circles for iris and flash drawn 
#cv2.imshow('eye_crop1',eyeR_color)
#cv2.imwrite('detected_right_eye.jpg',eyeR_color)
#cv2.imshow('eye_crop2',eyeL_color)
#cv2.imwrite('detected_left_eye.jpg',eyeL_color)
#cv2.waitKey(0) & 0xff
#cv2.destroyAllWindows()

#Coordinates of the flash in the left eye
fLx = np.int(circlesL_flash_sel[0])
fLy = np.int(circlesL_flash_sel[1])

#Coordinates of the flash in the right eye
fRx = np.int(circlesR_flash_sel[0])
fRy = np.int(circlesR_flash_sel[1])

#Coordinates of the iris in the left eye
ILx = np.int(circlesL_iris_sel[0])
ILy = np.int(circlesL_iris_sel[1])

#Coordinates of the iris in the right eye
IRx = np.int(circlesR_iris_sel[0])
IRy = np.int(circlesR_iris_sel[1])

#Flags for eye anomaly detection
flagL = False
flagR = False

#Threshold for eye anomaly detection
th = 0.155

print('Alignment Ratio Notation: ( Horizontal Alignment, Vertical Alignment)')

#Testing the 4 differnt cases of strabismus in the left eye
leftx_alignment = (fLx - ILx)/circlesL_iris_sel[2]
lefty_alignment = (fLy - ILy)/circlesL_iris_sel[2]
print('Left Eye Alignment Ratio: (',(fLx - ILx)/circlesL_iris_sel[2], ',', (fLy - ILy)/circlesL_iris_sel[2], ')')

rightx_alignment = (fRx - IRx)/circlesR_iris_sel[2]
righty_alignment = (fRy - IRy)/circlesR_iris_sel[2] 
print('Right Eye Alignment Ratio: (',(fRx - IRx)/circlesR_iris_sel[2], ', ', (fRy - IRy)/circlesR_iris_sel[2], ')')

if leftx_alignment < -th:
    print("Exotropia of the left eye")
    flagL = True
if leftx_alignment > th:
    print("Esotropia of the left eye")
    flagL = True
if lefty_alignment < -th:
    print("Hypotropia of the left eye")
    flagL = True
if lefty_alignment > th:
    print("Hypertropia of the left eye")
    flagL = True
if not(flagL):
   

    print("Left eye is normal")

#Testing the 4 differnt cases of strabismus in the right eye
if rightx_alignment < -th:
    print("Esotropia of the right eye")
    flagR = True
if rightx_alignment > th:
    print("Exotropia of the right eye")
    flagR = True
if righty_alignment < -th:
    print("Hypotropia of the right eye")
    flagR = True
if righty_alignment > th:
    print("Hypertropia of the right eye")
    flagR = True
if not(flagR):

    print("Right eye is normal")


#Resizing the image to display in OpenCv (large image aren't displayed properly)
img = imutils.resize(img_original, width=700)

#Displaying the original image with the different detected features overlaid 
#cv2.imshow('face',img)
#cv2.imwrite('face_detected.jpg', img)
#cv2.waitKey(0) & 0xff
#cv2.destroyAllWindows()