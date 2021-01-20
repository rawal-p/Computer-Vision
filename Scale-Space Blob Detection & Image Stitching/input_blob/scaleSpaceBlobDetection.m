%Scale-Space Blob Detection 
%Author: Pranav Rawal
%Student #: 001316396
%Email: rawalp@mcmaster.ca

%%
%Clear all previous variables from the workspace and clear the command
%window
clear;
clc;

%% 
% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end

imagefiles = dir('*.jpg');      
nfiles = length(imagefiles);    % Number of files found


%Initializing the scaling factor, k, specific to each of the images being
%procesed
k = [2^(0.25),2^(0.2),2^(0.15),2^(0.3),2^(0.3),2^(0.25),2^(0.15),2^(0.25)];

%Initializing the standard deviation values used for generating the LoG
%mask specific to each of the images being processed
sig = [2, 1.2, 1.3, 4.5, 2.5, 2, 2.5, 3];

%Initializing the threshold value for classifying blob detection in each
%scale space specific to each of the images being processed
th = [0.1, 0.045, 0.024, 0.07, 0.01, 0.05, 0.035, 0.01];

%Values for k, sig, and th were determined experimentally

%processing each image in the directory where this script is located
for ii = 1:nfiles
    
    %Reading and converting the image to double precision and grayscale
    currentfilename = imagefiles(ii).name;
    currentimage = imread(currentfilename);
    currentimage = im2double(currentimage);
    Img=rgb2gray(currentimage);
   
    [nrow,ncol] = size(Img);
    
    m=1; %varible for keeping track of the scale space level

    %Initializing the sigmaScale array which holds the values of the
    %standard deviation for the LoG mask for each level of the scale space,
    %consisting of 15 levels in the scale
    k_ii = k(ii);
    k_n = k_ii.^(0:14);
    sig_ii = sig(ii);
    sigmaScale = (sig_ii.*ones(1,15)).*k_n;
    
    for sigma = sigmaScale
       
       %Width of the mask is calculated taking into account that 99.7% of 
       %the data in gaussian distribuit is within 3 standard deviation from 
       %the mean and the width needs to be an odd number
       width = 2*ceil(sigma*3)+1;
       
       %Creating the LoG mask
       filter = fspecial('log', width, sigma);
       
       %scale-normalized LoG
       filter = (sigma^2).*filter;
       
       %Filtering the image using the scale-normalized LoG
       Img_f=imfilter(Img,filter, 'same', 'replicate');
       
       %Squaring the image
       Img_f = Img_f.*Img_f; %Square fo the Laplacian
       
       %Saving the square of Laplacian response for current level of scale
       %space
       if m==1
           scale_space = Img_f;
       else
           scale_space = cat(3,scale_space,Img_f);
       end
       
       %incrementing scale-space counter
       m = m+1;

    end
    %At this point scale_space contains the squared LoG response for all
    %the values of standard deviation defined in sigmaScale. scale_space is
    %a N x M x 15 matrix, where N and M are the dimensions of the original
    %image
    
    
    %Nonmaximum Suppression at scale-space level
    for i=1:15
        
        %replacing the elements of scale_space at each sigma level with the
        %highest value element in the 3x3 neighbourhood of each element.
        %Giving us the maximas for the slice
        localMax = ordfilt2(scale_space(:,:,i),9,ones(3,3)) ; 
        
        %Saving the maximas of each of the slices
        if i==1
            localMax_2d=localMax;
        else
            localMax_2d=cat(3,localMax_2d,localMax);
        end

    end

    %Computing the maximum element in the 3D scale space
    scale_space_max = max(localMax_2d,[],3);
    
    %Replacing all non maximum values with zeros at each level of the scale
    %space
    scale_space_max= (scale_space_max==localMax_2d).*localMax_2d;

    %Extracting the coordinates of the maximas and computing the radius of
    %the blob for each sigma (radius = sqrt(2)*sigma)--> from lectures
    for i=1 : 15
        
        %computing the radius of blobs being detected for each level of the
        %scale space
        radius = sigmaScale(i).*(2^(1/2));
        
        %Determine if each element in the slice is equal to the maximum for
        %all of the scale levels
        slice_max = (scale_space(:,:,i)==scale_space_max(:,:,i));
        
        %Determine if each element is above the threshold value
        threshold_requirement = (scale_space(:,:,i)>th(ii));
        
        %Classifying a detection based on the maxima and thresholding
        %requirements
        detection = slice_max & threshold_requirement;
        
        %Extracting the indices for pixels classified as detections
        [m,n] = find(detection); 

        %Saving the coordinates for each of the detections and their
        %corresponding radii 
        if i==1
            M=m;
            N=n;
            I=i;
            I=repmat(i,size(m,1),1);
            rad=radius;        
            rad=repmat(radius,size(m,1),1);
        else
            rad2=repmat(radius,size(m,1),1);
            rad=cat(1,rad,rad2);
            I2=repmat(i,size(m,1),1);
            I=cat(1,I,I2);
            M=cat(1,M,m);
            N=cat(1,N,n);
        end
    end
%% Displaying the Detected Blobs
    
    %Converting the detection coordinates from subscripts to linear
    %indices format to enable the calculation of the scale_space values for
    %each of the detections
    ind = sub2ind(size(scale_space),M,N,I);
    
    %Constructing matrix J consisting of scale_space values for each of the
    %detections, the linear indices for each of the detections, and the
    %radius of detections
    J = [scale_space(ind),ind,rad];
    
    %Sorting matrix J based on the scale_space values for each of the
    %detections in descending order. Forming the matrix J is crucial 
    %because performing the sortrows function on matrix J will maintain the
    %relationship between the indices and radii of the detections even
    %after sorting
    J = sortrows(J,'descend');

    
    if ii == 5;
        J = J(1:625,:);     
    elseif ii == 6;%Keeping the 300 detections for the spottedEagleRay.jpg
        J = J(1:300,:);
    else %Keeping the 100 detections for the other pics
        J = J(1:100,:);
    end

    
    %Converting the sorted linear indices back to subscript format to plot
    %the circles onto the original image
    [row,col,~] = ind2sub([nrow,ncol,15],J(:,2));
    rad = J(:,3);

    %Display original image and plotting the detected circles
    figure(ii)
    imshow(Img); hold on;

    viscircles([col row],rad,'Color','r','LineWidth',1.5);
    title(sprintf('Top %d detections for %s', size(row,1),currentfilename));

    hold off;
    
end

