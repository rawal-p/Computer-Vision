function[CamMatrix] = LinearCalib(Points3D,Points2D)

%Constructing Zero matrix same dimension as Points3D
[Rows,Columns] = size(Points3D);
Z = zeros(Rows,Columns);

%A1 is a (2n x 4) matrix representing the first 4 columns of matrix A
%Interlacing the Q matrix with rows of zeros starting from the second row
A1 = [Points3D,Z]'; 
A1 = reshape(A1(:),Columns,[])'; 

%A2 is a (2n x 4) matrix representing columns 5-8 matrix A
%Interlacing the Q matrix with rows of zeros starting from the first row
A2 = [Z,Points3D]'; 
A2 = reshape(A2(:),Columns,[])'; 

%A3 is a (2n x 4) matrix representing columns 9-12 matrix A
%extracting the u and v values from the q matrix
q_u = Points2D(:,1);
q_v = Points2D(:,2);

%pointwise multiplication of each row of matrix Q with the associated
%values of column vectors q_u and q_v to get matrices A3_u and A3_v
A3_u = -(q_u.*Points3D);
A3_v = -(q_v.*Points3D);

%Interlacing the rows matrices A3_u and A3_v, starting with the firt row 
%of A3_u to get matrix A3
A3 = [A3_u,A3_v]'; 
A3 = reshape(A3(:),Columns,[])'; 

%Concatenating A1, A2, A3
A = [A1,A2,A3];

%Preforming SVD on matrix A to get the projection matrix 
[U,D,V] = svd(A,0);
CamMatrix = V(:,end);

%converting P from (12x1) to (3x4)
CamMatrix = reshape(CamMatrix,[4,3])';

end





