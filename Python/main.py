#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:41:33 2019
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
from mayavi import mlab
import python.dlt_and_errors as dlt
mlab.clf()      


def check_valid_radius(self, number, imagepoints):
    """
    Check whether the projected points are inside the imageframe or not
    """
    wrongradius = False
    minimumx = 100000000000
    minimumy = minimumx
    maximumx = -10000
    maximumy = maximumx
    for i in range(number):
        if(imagepoints[0, i] < minimumx):
            minimumx = imagepoints[0, i]
        if(imagepoints[0, i] > maximumx):
            maximumx = imagepoints[0, i]
        if(imagepoints[1, i] < minimumy):
            minimumy = imagepoints[1, i]
        if(imagepoints[1, i] > maximumy):
            maximumy = imagepoints[1, i]
        if(minimumx * maximumx < 0):
            diffx = abs(maximumx) + abs(minimumx)
        else:
            diffx = abs(abs(maximumx) - abs(minimumx))
        if(minimumy * maximumy < 0):
            diffy = abs(maximumy)+abs(minimumy)
        else:
            diffy = abs(abs(maximumy)-abs(minimumy))
        if ((diffx <= self.img_width and diffy <= self.img_height) or (diffx <= self.img_height and diffy <= self.img_width)):
            wrongradius = False
        else:
            wrongradius = True
        return wrongradius

   
# ------------------------------------test cases------------------------------


#cam = Camera()


#sph = Sphere()


#cam.set_K(fx=800., fy=800., cx=640., cy=480.)
#cam.set_width_heigth(1280, 960)
#imagePoints = np.full((2, 6), 0.0)
#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
#cam.set_t(0.0, -0.0, 0.5, frame='world')
# worldpoints = sph.random_points(6,0.3)
# H=dlt.DLT3D(cam,worldpoints,imagePoints,True)
# DLTimage=dlt.DLTproject(H,worldpoints)

# -Find best points, that gives the minimum error to DLT-----------------------
#sph.spherepoints = None
#sph.spherepoints = points_config_randomtest(1000)
#print sph.spherepoints
#cams = [cam]
#spheres = [sph]
#plot3D(cams, spheres)
#Hbest=dlt.DLT3D(cam, sph.spherepoints, imagePoints,True)
#cam_center = camera_center(Hbest)
#estimated_t = dlt.estimate_t(cam, cam_center)
#Rotation = dlt.estimate_R_withQR(bestH)
#err_R = dlt.error_R(cam, Rotation)
#err_t = dlt.error_t(cam, estimated_t)

# ---------------------------------------Plot the best points configuration---
#sph.plot_sphere_and_points(sph.spherepoints)
