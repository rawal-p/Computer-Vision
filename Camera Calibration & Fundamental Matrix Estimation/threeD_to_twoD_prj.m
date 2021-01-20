clear
clc

%Importing the 2D and 3D features
q = importdata('Features2D.txt'); 
Q = importdata('Features3D.txt');

P = LinearCalib(Q,q);

prj_points2D = CameraProject(Q,P);

err = (abs(prj_points2D - q)./q)*100;
err_max = max(err)

threeD = importdata('3Dpoints.txt');

prj_points2D_new = CameraProject(threeD,P)