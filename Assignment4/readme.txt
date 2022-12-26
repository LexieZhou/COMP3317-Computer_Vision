COMP3317 Computer Vision
Assignment 4 - Camera calibration

1. Goal
 • estimating a planar projective transformation for each of the planes of a calibration grid
 • generating 2D-3D correspondences for corners on a calibration grid
 • estimating a projection matrix for a camera
 • decomposing a projection matrix into a camera calibration matrix and a matrix composed of the rigid body motion
 • estimating an essential matrix for a pair of cameras

2. Implemented Features
（1）calibrate2D()
Estimate a plane-to-plane projectivity for each of the two planes of the calibration grid using the 2D-3D point pairs picked by the user.
 - X-Z plane 
 • build initial matrix of bxz and Axz
 • In a for loop, obtain xi, zi, ui, vi from ref3D and ref2D
 • form the matrix Axz and bxz
 • use numpy function lstsq to form the equation of Axz@pxz = bxz and obtain matrix pxz
 • append 1 to the end of matrix pxz, get Hxz_
 • reshape Hxz_ into (3,3), get Hxz

 - Y-Z plane 
 • build initial matrix of byz and Ayz
 • In a for loop, obtain yii, zii, uii, vii from ref3D and ref2D
 • form the matrix Ayz and byz
 • use numpy function lstsq to form the equation of Ayz@pyz = byz and obtain matrix pyz
 • append 1 to the end of matrix pyz, get Hyz_
 • reshape Hyz_ into (3,3), get Hyz

（2）gen_correspondences()
Use the estimated plane-to-plane projectivities to assign 3D coordinates to all the detected corners on the calibration grid.
 • build initial matrix of co3Dxz_ and co3Dyz_ 
 • In the for loop, use coX, coY to define the 2D position
 • In the for loop, co3Dxz and co3Dyz hold 3D coordinates of 160 corners in 2 calibration plane accordingly, and store them into co3Dxz_ and co3Dyz_
 • Define x,y,z 3D coordinates of all 160 corners in 2 calibration plane using for loop
 • ref3D is the combination of co3Dxz_ and co3Dyz_

 - X-Z plane 
 • Build two initial list bxz_2D for holding the 2D coordinates of the projections
 • Find out the corners within the calibration plane 1
 • Build the matrix Pxz by Pxz = np.array([ref3D[k][0],ref3D[k][2],1])
 • find project corners on the calibration plane 1, compute bxz by bxz = Hxz@Pxz and append it to bxz_2D
 • transform bxz_2D into bxz_2D_array

 - Y-Z plane 
 • Build two initial list byz_2D for holding the 2D coordinates of the projections
 • Find out the corners within the calibration plane 2
 • Build the matrix Pyz by Pyz = np.array([ref3D[k][1],ref3D[k][2],1])
 • find project corners on the calibration plane 2, compute byz by byz = Hyz@Pyz and append it to byz_2D
 • transform byz_2D into byz_2D_array 

 • b_2D is the combination of bxz_2D_array and byz_2D_array
 • obtain ref2D by calling the function find_nearest_corner

（3）calibrate3D()
Estimate a 3×4 camera projection matrix P from all the detected corners on the calibration grid using linear least squares.
 • build initial matrix of b and A
 • In a for loop, obtain x, y, z, u, v from ref3D and ref2D
 • form the matrix A and b
 • use numpy function lstsq to form the equation of A@p = b and obtain matrix p
 • append 1 to the end of matrix p, get P
 • reshape P into (3,4)

（4）decompose_P()
Use QR decomposition to decompose the camera projection matrix P into the product of a 3×3 camera calibration matrix K and a 3×4 matrix [R T] composed of the rigid body motion of the camera.
Use QR decomposition to decompose the camera projection matrix P into the product of a 3×3 camera calibration matrix K and a 3×4 matrix [R T] composed of the rigid body motion of the camera.
 • store the third column into P3
 • extract the first three columns of P by deleting the last column of P
 • perform QR decomposition on the inverse P by using qr function, obtain Q and R
 • obtain K as the inverse of R
 • obtain R as the tranpose of Q
 • normalize K by dividing the whole matrix by k22
 • If the element k00 is negative, multiply the 1st column of K and the 1st row of R by -1 respectively to make it positive
 • If the element k11 is negative, multiply the 2nd column of K and the 2nd row of R by -1 respectively to make it positive
 • obtain T from P3 by using the equation T = 1/k22 * inv(K) @ P3, reshape T
 • form RT by R and T

Assignment 4 - Epipolar constraint
1. Goal
 • estimating an essential matrix for a pair of cameras

2. Implemented Features
（1）compose_E()
Estimate an essential matrix from the camera projection matrices of a pair of cameras.
 • get R1 and R2 by extracting the first three column of RT1 and RT2
 • obtain relative rotation R by R = R2@inv(R1)
 • get T1 and T2 by extracting the third column of RT1 and RT2
 • obtain relative translation T by T = T2-R@T1
 • form Tx using T[0],T[1] and T[2]
 • compose E by using the equation E = Tx@R
 


