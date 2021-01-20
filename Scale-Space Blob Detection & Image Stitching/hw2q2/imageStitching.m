%Stiching Pairs of Images
%Author: Pranav Rawal
%Student #: 001316396
%Email: rawalp@mcmaster.ca

%%
%Clear all previous variables from the workspace and clear the command
%window
clear;
clc;
%% Reading and Converting Images

%Read the image pair
image_left=imread('uttower_left.jpg');
image_right=imread('uttower_right.jpg');
 
%Converting images to GrayScale
image_left=rgb2gray(image_left);
image_right=rgb2gray(image_right);

%Converting images to double
image_left=im2double(image_left);
image_right=im2double(image_right);

% image_left = histeq(image_left);
% image_right = histeq(image_right);
%% Feature point detection using Harris Detector
%sigma = 2
%threshold = 0.05
%radius = 2

[~, y1, x1] = harris(image_left, 2, 0.05,2); 
[~, y2, x2] = harris(image_right, 2, 0.05,2);

%% SIFT descriptors for the Harris Features

sift_circles1 = [x1, y1, 5*ones(length(y1),1)];
sift_left = find_sift(image_left,sift_circles1,1.5);

sift_circles2 =  [x2, y2, 5*ones(length(y2),1)];
sift_right = find_sift(image_right,sift_circles2, 1.5);

%% Putative Matches

%Computing the Euclidean distance between all the SIFT descriptors of the
%left and right image
descriptor_dist=dist2(sift_left,sift_right);

th = 0.05; %setting the upper threshold for the distances

%Selecting matches with distance lower than threshold
[matches_left,matches_rigth]=find(descriptor_dist < th);

%Creating matches matrix which contain coordinates of matches from image 1 and image 2
matched_features=[ x1(matches_left) y1(matches_left) x2(matches_rigth) y2(matches_rigth)];

%initializing the putative matches for each image      
img_left = ([matched_features(:,1) matched_features(:,2) ones(size(matched_features,1),1)])';
img_right = ([matched_features(:,3) matched_features(:,4) ones(size(matched_features,1),1)])';

%% Display Putative Matches 

figure(1)
imshow([image_left image_right]); hold on;
plot(matched_features(:,1), matched_features(:,2), 'dy','LineWidth',1);
plot(matched_features(:,3)+size(image_left,2), matched_features(:,4), 'dy','LineWidth',1);
line([matched_features(:,1) matched_features(:,3) + size(image_left,2)]', matched_features(:,[2 4])', 'Color', 'y','LineWidth',1);
title(sprintf('%d Putative Matches', size(matched_features,1)));
hold off;

%% RANSAC Intialization
%Computing the number of RANSAC trials
k = 8; %number of random points to be chosen during RANSAC trials
P = 0.99; %total probability of success after N number of RANSAC trials
p = 0.6;  %probability of a single match being valid

N = ceil(log(1-P)/(log(1-p^k)));

%Initializing values and lists needed for RANSAC trials
inliers_th = 5; 
inliers_list = zeros(N,1);
avg_inlierSSD_list = zeros(N,1);
H_list = zeros(3,3,N);

%% Running RANSAC and finding homography matrix
for j=1:1000
        %Selecting k random points
        samples=(randperm(size(matched_features,1),k))';
        
        %coordinates of matches from left image
        x_kL = matched_features(samples,1);
        y_kL = matched_features(samples,2);

        %Corresponding coordinates of matches from right image
        x_kR = matched_features(samples,3);
        y_kR = matched_features(samples,4);

        %initializing the A matrix used for computing the homography matrix
        %corresponding to the k random point selected
        A=zeros(2*k,9);
        
        %Constructing the A matrix
        A1 = [x_kL,y_kL,ones(8,1),zeros(8,1),zeros(8,1),zeros(8,1),-x_kL.*x_kR,-x_kR.*y_kL,-x_kR];
        A2 = [zeros(8,1),zeros(8,1),zeros(8,1),x_kL,y_kL,ones(8,1),-x_kL.*y_kR,-y_kR.*y_kL,-y_kR];
        A = [A1,A2]';
        A  = reshape(A(:),9,[])'; 
        

        %Solving the A matrix to get the homography matrix using SVD
        [U,S,V]=svd(A);
        h=V(:,end);
        H=reshape(h,3,3)';
        
        %converting to homogeneous coordinates
        H = H/H(3,3);
        
        %Estimating the projected points of right image using the left
        %image and the estimated homography matrix
        img_right1 = H*img_left;
        img_right1 = img_right1./img_right1(3,:);
        
        %Computing the squared sum difference between the estimated and
        %actual right image points
        diff = img_right1 - img_right;
        ssd = sum(diff.^2);
        
        %Classifying inliers based on the inlier threshold 
        inliers_ind = find(ssd < inliers_th);
        
        %Computing the number of inliers in the current RANSAC trial
        inliers = length(inliers_ind);
        
        %Computing the average squared sum difference for the inliers in
        %the current RANSAC trial
        avg_inlierSSD = sum(ssd(inliers_ind))./inliers;
        
        %Concatenating the number of inliers in current trial to
        %inliers_list, which keeps track of number of inliers for every
        %RANSAC trial
        inliers_list(j,1)=inliers;
        avg_inlierSSD_list(j,1)=avg_inlierSSD;
        
        if j==1
            H_list=H;  %Creating homography list for each iteration   
            
        else
            H_list=cat(3,H_list,H); %Creating homography list for each iteration
            
        end
        
        
end

%% Choosing the best Homography Matrix

%Identifying the max number of inliers for all RANSAC trials
max_inlier=max(inliers_list);   

%Gathering all the RANSAC trials with the max number of inliers
max_inliers_index = find(inliers_list == max_inlier);

%Identifying the RANSAC trial with both the max number of inliers and the
%minimum average SSD for those inliers
[min_avgSSD,min_avgSSD_ind] = min(avg_inlierSSD_list(max_inliers_index));
optimal_index = max_inliers_index(min_avgSSD_ind);

%Selecting the homography matrix with the max number of inliers and minimum
%avg SSD for inliers
H = (H_list(:,:,optimal_index))';

%Estimating the projected points of right image using the left
%image and the best homography matrix
img_right2 = H'*img_left;
img_right2 = img_right2./img_right2(3,:);

%Computing the squared sum difference between the estimated and
%actual right image points
ssd = sum((img_right2 - img_right).^2);

%Separating the inliers and the outliers from the putative matches from
%earlier
inliers_ind = find(ssd < inliers_th);
outlier_ind = find(ssd >= inliers_th);
inliers = length(inliers_ind);
outliers = length(outlier_ind);


% Display Inlier and Outlier Matches for each image
figure(3)
imshow([image_left image_right]); hold on;
plot(matched_features(inliers_ind,1), matched_features(inliers_ind,2), 'dg','LineWidth',1.5);
plot(matched_features(outlier_ind,1), matched_features(outlier_ind,2), 'dr','LineWidth',1.5);
plot(matched_features(inliers_ind,3)+size(image_left,2), matched_features(inliers_ind,4), 'dg','LineWidth',1.5);
plot(matched_features(outlier_ind,3)+size(image_left,2), matched_features(outlier_ind,4), 'dr','LineWidth',1.5);
line([matched_features(inliers_ind,1) matched_features(inliers_ind,3) + size(image_left,2)]', matched_features(inliers_ind,[2 4])', 'Color', 'g','LineWidth',1.5);
line([matched_features(outlier_ind,1) matched_features(outlier_ind,3) + size(image_left,2)]', matched_features(outlier_ind,[2 4])', 'Color', 'r','LineWidth',1.5);
xlabel(sprintf('Average SSD is %f ',min_avgSSD));
title(sprintf('%d Inlier Matches and %d Outliers', inliers,outliers));hold off;

%% Transforming the Images based on the best homography matrix 
%Computing the transformed images based on the best homography matrix

%Creating a 2D projective transformation object used to transform
%the image based on the best homography matrix H
T = projective2d(H);

%Transforming the left image based on the transform object from last step
%RL is the refencing object which contains the X&Y limits of the
%transformed image left
%The transformed left image is not of any importance to us right now since
%the X&Y limits do not account for the right image
[~,RL] = imwarp(image_left,T,'nearest');

%Extracting the limits of the x and y values of the transformed image from
%the reference object
x_range = [floor(RL.XWorldLimits(1)),ceil(RL.XWorldLimits(2))];
y_range = [floor(RL.YWorldLimits(1)),ceil(RL.YWorldLimits(2))];

%Modifying the reference object's X&Y limits to incorporate the width and
%height of the right image
RL.XWorldLimits = [min(1,x_range(1)),max(size(image_right,2),x_range(2))];
RL.YWorldLimits = [min(1,y_range(1)),max(size(image_right,1),y_range(2))];

%Transforming the left image again according to the new X&Y limits
left_trans = imwarp(image_left,T,'nearest','OutputView',RL);

figure(4)
imshow(left_trans); hold on;
title('UT Tower Left Transformed Image');
hold off; 

%the second image is transformed using an identity matrix (resulting in no
%transformation) to get the transformation object from projective2d
%function which is then used as an input for the imwarp function
%which will provide an output image which is computed for the same range
%of values (xrange and yrange) as the the first transformed image, left
right_trans= imwarp(image_right,projective2d(eye(3)),'nearest','OutputView',RL);

figure(5)
imshow(right_trans); hold on;
title('UT Tower Right Transformed Image');
hold off; 

%% Stitching and Displaying the Images together

%initializing the final_image as the transformed left image
final_image = left_trans;

%Computing the indices where the left image has intensity of 0. These
%indices will represent regions of the right image which do not overlap
%with the left image and regions where both the left and right images have
%intensity values of 0.
zero_ind = find(final_image==0);

%Computing the indices where the two images overlap and are not equal to 0
overlap_ind = find((final_image ~= 0) & (right_trans ~=0));

%Setting the values for the final image (left) to the values of the right 
%for all pixels where the left image has an intensity of 0. This operation
%will copy all of the pixels values from the right image which do not 
%overlap with the left image and also the regions of the right image where
%the pixel intensities are 0.
final_image(zero_ind) = right_trans(zero_ind);

%Setting the values for the final image to the average of the pixel 
%intensities for the right and left images for regions of overlap 
final_image(overlap_ind) = (left_trans(overlap_ind))/2 + (right_trans(overlap_ind))/2;

%Final stiching image
figure(6)
imshow(final_image); hold on
title('Final Stiched Image');
hold off;

