clear
clc

%Importing the 2D and 3D features
q = importdata('Features2D.txt'); 
Q = importdata('Features3D.txt');

P = LinearCalib(Q,q);

%extracting the calibration matrix K, containing the intrinsic parameters
%of the camera
P_bar = P(:,1:3);
B = chol(P_bar*P_bar');
K = B./B(3,3)

%extracting the extrinsic parameters, matrix R and vector t
A = inv(K)*P;
A = A(:,1:3);
lambda = nthroot(1/det(A),3);

R = lambda*A
t = lambda*inv(K)*P;
t = t(:,4);
t = -R'*t

