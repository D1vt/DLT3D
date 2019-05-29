"""
Created on Thu Apr  5 15:41:33 2019
 /
@author: diana
"""
import numpy as np
import scipy.linalg
import random
from vision.camera import Camera
from python.sphere import Sphere
from numpy import linalg as LA
from scipy.linalg import expm, inv
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def normalize(points, size=2):
    """
 Normalize image points, source:
 https://inside.mines.edu/~whoff/courses/EENG512/lectures/18-LinearPoseEstimation.pdf
    """
    pointsnorm = np.dot(inv(cam.K), points)
    return pointsnorm


def DLT3D(self, worldpoints, imagepoints, normalization=False):
    """
    This function calculates the final H matrix, using DLT
    algorithm for 3D points
    """
    # if odd row :0,0,0,0,xi,yi,zi,1,-vixi,-viyi,-vizi,-vi
    # if even row : Χι,Υι,Ζι,1,0,0,0,0,-uixi,-uiyi,-uizi,-ui
    if not normalization:
        imagepoints = normalize(imagepoints)
    A = np.array([[worldpoints[0, 0], worldpoints[1, 0], worldpoints[2, 0], 1., 0., 0., 0., 0., -imagepoints[0, 0]*worldpoints[0, 0], -imagepoints[0, 0]*worldpoints[1, 0], -imagepoints[0, 0]*worldpoints[2, 0], -imagepoints[0, 0]],
               [0., 0., 0., 0., worldpoints[0, 0], worldpoints[1, 0], worldpoints[2, 0], 1., -imagepoints[1,0]*worldpoints[0,0],-imagepoints[1,0]*worldpoints[1,0],-imagepoints[1,0]*worldpoints[2,0],-imagepoints[1,0]], 
               [worldpoints[0, 1],worldpoints[1,1],worldpoints[2,1],1.,0.,0.,0.,0., -imagepoints[0, 1]*worldpoints[0,1],-imagepoints[0,1]*worldpoints[1,1],-imagepoints[0,1]*worldpoints[2,1],-imagepoints[0,1]],                              
               [0.,0.,0.,0.,worldpoints[0,1],worldpoints[1,1],worldpoints[2,1],1., -imagepoints[1, 1]*worldpoints[0,1],-imagepoints[1,1]*worldpoints[1,1],-imagepoints[1,1]*worldpoints[2,1],-imagepoints[1,1]],
               [worldpoints[0,2],worldpoints[1,2],worldpoints[2,2],1.,0., 0., 0., 0., -imagepoints[0, 2]*worldpoints[0,2],-imagepoints[0,2]*worldpoints[1,2],-imagepoints[0,2]*worldpoints[2,2],-imagepoints[0,2]],
               [0.,0.,0.,0.,worldpoints[0,2],worldpoints[1,2],worldpoints[2, 2], 1.,-imagepoints[1, 2]*worldpoints[0,2],-imagepoints[1,2]*worldpoints[1,2],-imagepoints[1,2]*worldpoints[2,2],-imagepoints[1,2]],
               [worldpoints[0,3],worldpoints[1,3],worldpoints[2,3],1.,0.,0.,0.,0.,-imagepoints[0, 3]*worldpoints[0,3],-imagepoints[0,3]*worldpoints[1,3],-imagepoints[0,3]*worldpoints[2,3],-imagepoints[0,3]],
               [0.,0.,0.,0.,worldpoints[0,3],worldpoints[1,3],worldpoints[2,3],1.,-imagepoints[1, 3]*worldpoints[0,3],-imagepoints[1,3]*worldpoints[1,3],-imagepoints[1,3]*worldpoints[2,3],-imagepoints[1,3]],
               [worldpoints[0, 4], worldpoints[1, 4], worldpoints[2, 4], 1., 0., 0., 0., 0., -imagepoints[0,4]*worldpoints[0,4],-imagepoints[0,4]*worldpoints[1,4],-imagepoints[0,4]*worldpoints[2,4],-imagepoints[0,4]],
               [0., 0., 0., 0., worldpoints[0, 4], worldpoints[1, 4], worldpoints[2, 4], 1., -imagepoints[1,4]*worldpoints[0,4],-imagepoints[1,4]*worldpoints[1,4],-imagepoints[1,4]*worldpoints[2,4],-imagepoints[1,4]],
               [worldpoints[0, 5], worldpoints[1, 5], worldpoints[2, 5],1., 0., 0., 0., 0., -imagepoints[0,5]*worldpoints[0,5],-imagepoints[0,5]*worldpoints[1,5],-imagepoints[0,5]*worldpoints[2,5],-imagepoints[0,5]],
               [0., 0., 0., 0., worldpoints[0, 5], worldpoints[1, 5], worldpoints[2, 5], 1., -imagepoints[1,5]*worldpoints[0,5],-imagepoints[1,5]*worldpoints[1,5],-imagepoints[1,5]*worldpoints[2,5],-imagepoints[1,5]]])
    U1, s, Vh = np.linalg.svd(A)
    Vh = np.transpose(Vh)
    H = Vh[:, 11].reshape(3, 4)
    if not normalization:
        H = np.dot(self.K, H)
    return H

def estimate_R_withQR(H):
    """
    A function to estimate the rotation matrix R, using the H matrix, that was
    calculated from DLT algorithm.
    To do that we use the QR Decomposition. Source: p. 31
    https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws13-14/3DCV_lec05_parameterEstimation.pdf
    """
    Q, B = scipy.linalg.qr((H[:3, :3]))
    for i in range(3):
        if B[i, i] < 0.:
            B[i, :] = - B[i, :]
            Q[:, i] = - Q[:, i]
    Rot = Q
    if np.linalg.det(Rot) < 0.:
        Rot = -Rot
    # Final estimated pose
    Rot1 = np.transpose(Rot)
    # print Rot1
    est_R = np.array([[Rot1[0, 0], Rot1[0, 1], Rot1[0, 2], 0.],
                      [Rot1[1, 0], Rot1[1, 1], Rot1[1, 2], 0.],
                      [Rot1[2, 0], Rot1[2, 1], Rot1[2, 2], 0.],
                      [0., 0., 0., 1.]])
    return est_R


def estimate_t(self, cam_center):
    """
    A function to estimate the translation t, using the H matrix,
    that was found from DLT algorithm
    Source: Multiple View Geometry Richard Hartley and Andrew Zisserman,
    p.15 decompose P to K,R,t matrices
    """
    est_trans = np.array([[1., 0., 0., abs(cam_center[0])],
                          [0., 1., 0., abs(cam_center[1])],
                          [0., 0., 1., abs(cam_center[2])],
                          [0., 0., 0., 1.]])

    return est_trans


def camera_center(H):
    """
    A function to estimate the camera center, using the H matrix, that was
    found from DLT algorithm. Source:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud810/slides/Unit-3/3C-L3.pdf
    """
    Q = np.array([[H[0, 0], H[0, 1], H[0, 2]],
                  [H[1, 0], H[1, 1], H[1, 2]],
                  [H[2, 0], H[2, 1], H[2, 2]]])
    b = np.array([[H[0, 3]],
                  [H[1, 3]],
                  [H[2, 3]]])
    Q = -inv(Q)
    cam_center = np.dot(Q, b)
    return cam_center


def DLTproject(H, X, quant_error=False):
        """
        Project points in X (6*n array) and normalize coordinates,
        using H that was calculated from DLT.
        """
        x = np.dot(H, X)
        for i in range(x.shape[1]):
            x[:, i] /= x[2, i]
        if(quant_error):
            x = np.around(x, decimals=0)
        return x


def reprojection_error(imagepoints, DLTimage, points):
    """
        Find reprojection error,
        caused by DLT. Source:
        https://en.wikipedia.org/wiki/Reprojection_error
    """
    repr_error = 0.
    for size in range(points):
        distance = math.sqrt((imagepoints[0, size]-DLTimage[0, size]) *
                             (imagepoints[0, size]-DLTimage[0, size])
                             + (imagepoints[1, size]-DLTimage[1, size]) *
                             (imagepoints[1, size]-DLTimage[1, size]))
        repr_error = distance+repr_error
    return repr_error


def error_R(self, estimated_R):
    """
    Find R error, caused from DLT.Source:
    https://inside.mines.edu/~whoff/courses/EENG512/lectures/18-LinearPoseEstimation.pdf
    """
    error_r = np.dot(estimated_R[:3, :3], inv(self.R[:3, :3]))
    ang = math.acos(((error_r[0, 0]+error_r[1, 1]+error_r[2, 2])-1.)/2.)
    # print ang * 180/math.pi
    return np.rad2deg(ang)

   
def error_t(self, estimated_t):
    """
    Find t error, caused from DLT.Source:
    https://inside.mines.edu/~whoff/courses/EENG512/lectures/18-LinearPoseEstimation.pdf
    """
    ertx = ((estimated_t[0, 3]-self.t[0, 3])*(estimated_t[0, 3]-self.t[0, 3]))
    erty = ((estimated_t[1, 3]-self.t[1, 3])*(estimated_t[1, 3]-self.t[1, 3]))
    ertz = ((estimated_t[2, 3]-self.t[2, 3])*(estimated_t[2, 3]-self.t[2, 3]))
    normt = math.sqrt((self.t[0, 3]*self.t[0, 3])+(self.t[1, 3]*self.t[1, 3])
                      + (self.t[2, 3]*self.t[2, 3]))
    total_error = math.sqrt(ertx+erty+ertz)/normt
    total_error = total_error * 100.
    return total_error

def add_noise(imagepoints, sd=4., mean=0., size=10000):
    """
    Adding noise to each u,v of the image (random gaussian noise)
    """
    if size == 0:
        imagenoise = np.full((3, 6), 1.0)
        imagenoise[0, 0] = imagepoints[0, 0] + np.random.normal(mean, sd)
        imagenoise[1, 0] = imagepoints[1, 0] + np.random.normal(mean, sd)
        imagenoise[0, 1] = imagepoints[0, 1] + np.random.normal(mean, sd)
        imagenoise[1, 1] = imagepoints[1, 1] + np.random.normal(mean, sd)
        imagenoise[0, 2] = imagepoints[0, 2] + np.random.normal(mean, sd)
        imagenoise[1, 2] = imagepoints[1, 2] + np.random.normal(mean, sd)
        imagenoise[0, 3] = imagepoints[0, 3] + np.random.normal(mean, sd)
        imagenoise[1, 3] = imagepoints[1, 3] + np.random.normal(mean, sd)
        imagenoise[0, 4] = imagepoints[0, 4] + np.random.normal(mean, sd)
        imagenoise[1, 4] = imagepoints[1, 4] + np.random.normal(mean, sd)
        imagenoise[0, 5] = imagepoints[0, 5] + np.random.normal(mean, sd)
        imagenoise[1, 5] = imagepoints[1, 5] + np.random.normal(mean, sd)
        return imagenoise
    else:
        px1 = imagepoints[0, 0] + np.random.normal(mean, sd, size)
        py1 = imagepoints[1, 0] + np.random.normal(mean, sd, size)
        px2 = imagepoints[0, 1] + np.random.normal(mean, sd, size)
        py2 = imagepoints[1, 1] + np.random.normal(mean, sd, size)
        px3 = imagepoints[0, 2] + np.random.normal(mean, sd, size)
        py3 = imagepoints[1, 2] + np.random.normal(mean, sd, size)
        px4 = imagepoints[0, 3] + np.random.normal(mean, sd, size)
        py4 = imagepoints[1, 3] + np.random.normal(mean, sd, size)
        px5 = imagepoints[0, 4] + np.random.normal(mean, sd, size)
        py5 = imagepoints[1, 4] + np.random.normal(mean, sd, size)
        px6 = imagepoints[0, 5] + np.random.normal(mean, sd, size)
        py6 = imagepoints[1, 5] + np.random.normal(mean, sd, size)
        imageNoise = np.array([[px1, px2, px3, px4, px5, px6],
                               [py1, py2, py3, py4, py5, py6]])
        return imageNoise        
    
    
# ---------- test -------------------------------------------
cam = Camera()


sph = Sphere()


cam.set_K(fx=800., fy=800., cx=640., cy=480.)
cam.set_width_heigth(1280, 960)
imagePoints = np.full((2, 6), 0.0)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
cam.set_t(0.0, -0.0, 0.5, frame='world')
worldpoints = sph.random_points(6, 0.3)
imagePoints = cam.project(worldpoints, False)    

H = DLT3D(cam,worldpoints, imagePoints, True)
DLTimage = DLTproject(H, worldpoints)
