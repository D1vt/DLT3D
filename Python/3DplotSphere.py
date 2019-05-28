#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:18:43 2019

@author: diana
"""

# -*- coding: utf-8 -*-

from scipy.linalg import expm, rq, det, inv
import matplotlib.pyplot as plt
from math import atan
import numpy as np
import math
import random
from vision.camera import Camera
from Python.Class_Sphere import Sphere
from mayavi import mlab


  
def display_figure():
    """Display the Mayavi Opengl figure"""
    mlab.show()


def plot3D_cam(cam, axis_scale=0.2):
    """Plots the camera axis in a given position and orientation in 3D space
    Parameters
    ----------
    cam : :obj:`Camera`
            Object of the type Camera, with a proper Rt matrix.
    axis_scale : int, optional
            The Scale of the axis in 3D space.
    Returns
    -------
    None
    """
    # Coordinate Frame of camera
    cam_axis_x = np.array([1, 0, 0, 1]).T
    cam_axis_y = np.array([0, 1, 0, 1]).T
    cam_axis_z = np.array([0, 0, 1, 1]).T

    cam_axis_x = np.dot(cam.R.T, cam_axis_x)
    cam_axis_y = np.dot(cam.R.T, cam_axis_y)
    cam_axis_z = np.dot(cam.R.T, cam_axis_z)

    cam_world = cam.get_world_position()

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2],
                  cam_axis_x[0], cam_axis_x[1], cam_axis_x[2],
                  line_width=3, scale_factor=axis_scale,
                  color=(1-axis_scale, 0, 0))

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2],
                  cam_axis_y[0], cam_axis_y[1], cam_axis_y[2],
                  line_width=3, scale_factor=axis_scale,
                  color=(0, 1-axis_scale, 0))

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2],
                  cam_axis_z[0], cam_axis_z[1], cam_axis_z[2],
                  line_width=3, scale_factor=axis_scale,
                  color=(0, 0, 1-axis_scale))


def plotPoints3D(fiducial_space, scale=0.01):
    """Plots a set of points from a fiducial space on 3D Space
    Parameters
    ----------
    fiducial_space : object
            This can be a Plane() object or similar fiducial space objects
    scale : float, optional
            Scale of each one of the plotted points
    Returns
    -------
    None
    """

    fiducial_points = fiducial_space.get_sphere_points()

    mlab.points3d(fiducial_points[0], fiducial_points[1], fiducial_points[2],
                  scale_factor=scale, color=fiducial_space.get_color())


def plot3D(cams, planes):
    """Plots a set of cameras and a set of fiducial planes on the 3D Space
    Parameters
    ----------
    cams : list
            List of objects of the type Camera each one with a proper Rt matrix
    planes : list
            List of objects of the type Plane
    Returns
    -------
    None
    """
    # mlab.figure(figure=None, bgcolor=(0.1,0.5,0.5), fgcolor=None,
    #              engine=None, size=(400, 350))
    axis_scale = 0.05
    for cam in cams:
        plot3D_cam(cam, axis_scale)
    for fiducial_space in planes:
        # Plot plane points in 3D
        plotPoints3D(fiducial_space)
display_figure()    
cam = Camera()
cam.set_K(fx=100, fy=100, cx=640, cy=480)
cam.set_width_heigth(1280, 960)

""" Initial camera pose looking straight down into the plane model """
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
cam.set_t(0.0, 0.0, 1.5, frame='world')

""" Plane for the control points """
sph = Sphere(2)
sph.set_color((1, 1, 0))
sph.random_points(3000, 0.3, 0.0001)

""" Plot camera axis and plane """
cams = [cam]
planes = [sph]
plot3D(cams, planes)
