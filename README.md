# DLT3D
(A) Matlab code : 
Plane: Random points from a plane used and samples from a real world object. 
Shpere: Random points from a sphere used and samples from a real world object. 

In both cases, DLT for 3D points was implemented in order to find the M matrix (3*4).

(B) Python Code:
Shpere: Random points from a sphere used and random points from a real world object. 

Thne DLT and SVD were implemented in order to find M vector(12*1).


K matrix is a well known matrix (Kint, camera parameters). 
K= [a b c
    0 d e
    0 0 f]
In modern cameras there is no skew, so b=0 and f=1. Thus, I asked the user for a-f inputs. Using these inputs I have calculated R (rotation matrix) and T (translation matrix).
