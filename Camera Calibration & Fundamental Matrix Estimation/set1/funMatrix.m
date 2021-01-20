clear 
clc

%% Importing the 2D and 3D features
q1 = importdata('pt_2D_1.txt');
q2 = importdata('pt_2D_2.txt');
im1=imread('image1.jpg');
im2=imread('image2.jpg');


%% Swapping the columns of the the datasets
%coordinates
temp = q1(:,1);
q1(:,1) = q1(:,2);
q1(:,2) = temp;

temp = q2(:,1);
q2(:,1) = q2(:,2);
q2(:,2) = temp;

%% Converting to homogeneous coordinates
[rows,cols] = size(q1);
oneVector = ones(rows,1);
q1_h = [q1 oneVector];
q2_h = [q2 oneVector];

%% Normalizing Data and creating the Transformation Matrix for Normalization
T = normalizeTransformation(q1);
T_prime = normalizeTransformation(q2);
q1_n = (T*q1_h')';
q2_n = (T_prime*q2_h')';

%% Constructing the A matrix
% A1 = [q1_h q1_h q1_h];
% A2 = [q2_h(:,1) q2_h(:,1) q2_h(:,1) q2_h(:,2) q2_h(:,2) q2_h(:,2) q2_h(:,3) q2_h(:,3) q2_h(:,3)];
% A = A1.*A2;


A1 = [q1_n q1_n q1_n];
A2 = [q2_n(:,1) q2_n(:,1) q2_n(:,1) q2_n(:,2) q2_n(:,2) q2_n(:,2) q2_n(:,3) q2_n(:,3) q2_n(:,3)];
A = A1.*A2;

%% Compute SVD of A matrix to get the Fundamental Matrix (not rank 2)
[U,D,V] = svd(A);
F = V(:,end);

F = reshape(F,[3,3])';

%% Compute SVD of F matrix to get the new Fundamental Matrix and impose the
%  rank to constraint
[U,D,V] = svd(F);
D(3,3) = 0;
F_new = U*D*V';

%% Denormalizing the rank 2 contrained Fundamental Matrix
F_new_denormalized = (T_prime)'*F_new*T;

%% Computing the epipolar lines 

% l2 = F_new*q1_h';
% l1 = F_new'*q2_h';

l2 = F_new_denormalized*q1_h';
l1 = F_new_denormalized'*q2_h';
l1 = l1';
l2 = l2';

X1 = linspaceNDim(q1(:,1) - 29, q1(:,1)+30,60);
X2 = linspaceNDim(q2(:,1) - 29, q2(:,1)+30,60);
y1 = (-l1(:,1).*X1 - l1(:,3))./l1(:,2);
y2 = (-l2(:,1).*X2 - l2(:,3))./l2(:,2);

%% Plotting the images, image points, and epipolar lines
figure(2)
imshow(im1)
hold on;
plot(X1',y1', 'r', 'LineWidth', 1);
scatter(q1(:,1),q1(:,2),20,'white','filled')
hold off;

figure(3)
imshow(im2)
hold on;
plot(X2',y2', 'r','LineWidth', 1);
scatter(q2(:,1),q2(:,2),20,'white','filled')
hold off;

%% Computing the Average distance from epipolar lines to the corresponding image points

D1 = abs(l1(:,1).*q1(:,1) + l1(:,2).*q1(:,2) + l1(:,3))./((l1(:,1).^2) + (l1(:,2).^2)).^(1/2);
D2 = abs(l2(:,1).*q2(:,1) + l2(:,2).*q2(:,2) + l2(:,3))./((l2(:,1).^2) + (l2(:,2).^2)).^(1/2);

d1 = mean(D1)
d2 = mean(D2)