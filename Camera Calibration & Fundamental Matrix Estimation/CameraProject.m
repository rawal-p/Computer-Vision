function[PrjPoints2D] = CameraProject(Points3D,CamMatrix)

%q_prj --> (3x37)
q_prj = CamMatrix*Points3D';

%a,b,c --> (1x37)
a = q_prj(1,:);  %scaled projected u values
b = q_prj(2,:);  %scaled projected v values
c = q_prj(3,:);  %scaling factor

%u,v --> (1x37)
u = a./c;   %normalized u values 
v = b./c;   %normalized v values

%PrjPoints2D --> (2x37)
PrjPoints2D = [u;v]';

end
