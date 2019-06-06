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
import python.R_t_spherical_coordinates.py as rtspherical
mlab.clf()      
import csv
with open('error_r_sph.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_t_sph.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_reproject_sph.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_r_prism.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_t_prism.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_reproject_prism.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')    
with open('error_cov_prism.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')   
with open('error_cov_sph.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')         

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

def rms_error(self):
    i = 0
    rms_sph = np.array([[0],
                        [0]])
    rms_prism = np.array([[0],
                          [0]])
    mean_t_prism = 0.
    mean_t_sphere = 0.
    mean_r_prism = 0.
    mean_r_sphere = 0.
    mean_repro_prism = 0.
    mean_repro_sphere = 0.
    for i in range(100):
     sph_points = self.random_points(6, 0.07071)
     prism_points =self.random_prism(0.07071)
     imagesphpoints = cam.project(sph_points)
     imagesphpoints = dlt.stand_add_noise(imagesphpoints)
     imageprismpoints = cam.project(prism_points)
     imageprismpoints = dlt.stand_add_noise(imageprismpoints)
     H_sph = dlt.DLT3D(cam, sph_points, imagesphpoints, normalization=False)
     H_prism = dlt.DLT3D(cam, prism_points, imageprismpoints, normalization=False)
     sph_reproject = dlt.DLTproject(H_sph, sph_points, quant_error=False)
     prism_reproject = dlt.DLTproject(H_prism, prism_points, quant_error=False)
     reproject_sph = dlt.reprojection_error(imagesphpoints, sph_reproject, 6)
     reproject_prism = dlt.reprojection_error(imageprismpoints, prism_reproject, 6)
     estim_R_sph = dlt.estimate_R_withQR(H_sph)
     estim_center_sph = dlt.camera_center(H_sph)
     estim_t_sph = dlt.estimate_t(cam, estim_center_sph)
     t_error_sph =  dlt.error_t(cam, estim_t_sph)
     r_error_sph =  dlt.error_R(cam, estim_R_sph)
     estim_R_prism =  dlt.estimate_R_withQR(H_prism)
     estim_center_prism =  dlt.camera_center(H_prism)
     estim_t_prism =  dlt.estimate_t(cam, estim_center_prism)
     t_error_prism =  dlt.error_t(cam, estim_t_prism)
     r_error_prism =  dlt.error_R(cam, estim_R_prism)
     mean_t_prism = mean_t_prism + t_error_prism
     mean_t_sphere = mean_t_sphere + t_error_sph
     mean_r_prism =  mean_r_prism + r_error_prism
     mean_r_sphere = mean_r_sphere + r_error_sph
     mean_repro_prism = mean_repro_prism + reproject_prism
     mean_repro_sphere = mean_repro_sphere + reproject_sph
     #covar_sph = np.cov(H_sph)
     conditioncov_sph = LA.cond(H_sph)
    # covar_prism = np.cov(H_prism)
     conditioncov_prism = LA.cond(H_prism)
     with open('error_t_sph.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(t_error_sph), i])
     with open('error_cov_sph.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(conditioncov_sph), i])
     with open('error_cov_prism.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(conditioncov_prism), i])
   
     with open('error_r_sph.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(r_error_sph), i])
     with open('error_reproject_sph.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(reproject_sph), i])
     with open('error_t_prism.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(t_error_prism), i])
   
     with open('error_r_prism.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(r_error_prism), i])
     with open('error_reproject_prism.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(reproject_prism), i])       
     for k in range(6): 
         rms_prism[0] = rms_prism[0] + (prism_points[0,k]-prism_reproject[0,k])* (prism_points[0,k]-prism_reproject[0,k])
         rms_prism[1] = rms_prism[1] + (prism_points[1,k]-prism_reproject[1,k])* (prism_points[1,k]-prism_reproject[1,k])
         rms_sph[0] = rms_sph[0] + (sph_points[0,k]-sph_reproject[0,k])*(sph_points[0,k]-sph_reproject[0,k])
         rms_sph[1] = rms_sph[1] + (sph_points[1,k]-sph_reproject[1,k])*(sph_points[1,k]-sph_reproject[1,k])
    sphere = math.sqrt(rms_sph[0]*rms_sph[0]+rms_sph[1]*rms_sph[1])
    prism = math.sqrt(rms_prism[0]*rms_prism[0]+rms_prism[1]*rms_prism[1])
    rms_sph =  math.sqrt(sphere/100.)
    rms_prism = math.sqrt(prism/100.) 
    print rms_prism
    print rms_sph
    return mean_t_prism/100., mean_t_sphere/100., mean_r_prism/100., mean_r_sphere/100., mean_repro_prism/100., mean_repro_sphere/100.   
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

# Test case using a,b,r spherical coordinates---------------------------------
#a, b, r = rtspherical.a_b_r_spherical(cam)
#worldpoints = sph.random_points(6, 0.3)
#covmatrix = rtspherical.covariance_matrix_p(cam, np.transpose(worldpoints),
                    #            imagePoints, a, b, r)
#rtspherical.rbest=rtspherical.calculate_best_r(cam,np.transpose(worldpoints),imagePoints,b,a)
#rtspherical.a_angle_deriv(cam, np.transpose(worldpoints), imagePoints, b, r)
#rtsphericalbest_a(cam, np.transpose(worldpoints), imagePoints, b, r)
# rtspherical.calculate_best_a(cam,np.transpose(worldpoints),imagePoints,b,r)
# rtspherical.bestangledist(cam)
