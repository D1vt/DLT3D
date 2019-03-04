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
from mayavi import mlab

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self):
        """ Initialize P = K[R|t] camera model. """
        self.P = np.eye(3,4)
        self.K = np.eye(3, dtype=np.float32) # calibration matrix
        self.R = np.eye(4, dtype=np.float32) # rotation
        self.t = np.eye(4, dtype=np.float32) # translation
        self.Rt = np.eye(4, dtype=np.float32)
        self.fx = 1.
        self.fy = 1.
        self.cx = 0.
        self.cy = 0.
        self.img_width = 1280
        self.img_height = 960

    def clone_withPose(self, tvec, rmat):
        new_cam = Camera()
        new_cam.K = self.K
        new_cam.set_R_mat(rmat)
        new_cam.set_t(tvec[0], tvec[1],  tvec[2])
        new_cam.set_P()
        new_cam.img_height = self.img_height
        new_cam.img_width = self.img_width
        return new_cam

    def clone(self):
        new_cam = Camera()
        new_cam.P = self.P
        new_cam.K = self.K
        new_cam.R = self.R
        new_cam.t = self.t
        new_cam.Rt = self.Rt
        new_cam.fx = self.fx
        new_cam.fy = self.fy
        new_cam.cx = self.cx
        new_cam.cy = self.cy
        new_cam.img_height = self.img_height
        new_cam.img_width = self.img_width
        return new_cam


    def set_P(self):
        # P = K[R|t]
        # P is a 3x4 Projection Matrix (from 3d euclidean to image)
        #self.Rt = hstack((self.R, self.t))
        self.P = np.dot(self.K, self.Rt[:3,:4])

    def set_K(self, fx = 1, fy = 1, cx = 0,cy = 0):
        # K is the 3x3 Camera matrix
        # fx, fy are focal lenghts expressed in pixel units
        # cx, cy is a principal point usually at image center
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.mat([[fx, 0, cx],
                      [0,fy,cy],
                      [0,0,1.]], dtype=np.float32)
        self.set_P()

    def set_width_heigth(self,width, heigth):
        self.img_width = width
        self.img_height = heigth

    def update_Rt(self):
        self.Rt = np.dot(self.t,self.R)
        self.set_P()

    def set_R_axisAngle(self,x,y,z, alpha):
        """  Creates a 3D [R|t] matrix for rotation
        around the axis of the vector defined by (x,y,z)
        and an alpha angle."""
        #Normalize the rotation axis a
        a = np.array([x,y,z])
        a = a / np.linalg.norm(a)

        #Build the skew symetric
        a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(4)
        R[:3,:3] = expm(a_skew*alpha)
        self.R = R
        self.update_Rt()

    def set_R_mat(self,R):
        self.R = R
        self.update_Rt()


    def set_t(self, x,y,z, frame = 'camera'):
        #self.t = array([[x],[y],[z]])
        self.t = np.eye(4)
        if frame=='world':
          cam_world = np.array([x,y,z,1]).T
          cam_t = np.dot(self.R,-cam_world)
          self.t[:3,3] = cam_t[:3]
        else:
          self.t[:3,3] = np.array([x,y,z])
        self.update_Rt()

    def get_normalized_pixel_coordinates(self, X):
        """
        These are in normalised pixel coordinates. That is, the effects of the
        camera's intrinsic matrix and lens distortion are corrected, so that
        the Q projects with a perfect pinhole model.
        """
        return np.dot(inv(self.K), X)

    def addnoise_imagePoints(self, imagePoints, mean = 0, sd = 2):
        """ Add Gaussian noise to image points
        imagePoints: 3xn points in homogeneous pixel coordinates
        mean: zero mean
        sd: pixels of standard deviation
        """
        imagePoints = np.copy(imagePoints)
        if sd > 0:
            gaussian_noise = np.random.normal(mean,sd,(2,imagePoints.shape[1]))
            imagePoints[:2,:] = imagePoints[:2,:] + gaussian_noise
        return imagePoints

    def get_tvec(self):
        tvec = self.t[:,3]
        return tvec

    def get_world_position(self):
        t = np.dot(inv(self.Rt), np.array([0,0,0,1]))
        return t


    def project(self,X, quant_error=False):
        """  Project points in X (4*n array) and normalize coordinates. """
        self.set_P()
        x = np.dot(self.P,X)
        for i in range(x.shape[1]):
          x[:,i] /= x[2,i]
        if(quant_error):
            x = np.around(x, decimals=0)
        return x

    def rotation_matrix(a, alpha):
      """  Creates a 3D [R|t] matrix for rotation
      around the axis of the vector a by an alpha angle."""
      #Normalize the rotation axis a
      a = a / np.linalg.norm(a)

      #Build the skew symetric
      a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
      R = np.eye(4)
      R[:3,:3] = expm(a_skew*alpha)
      return R

    def translation_matrix(t):
      """  Creates a 3D [R|t] matrix with a translation t
      and an identity rotation """
      R = np.eye(4)
      R[:3,3] = np.array([t[0],t[1],t[2]])
      return R

    def plot_image(self, imgpoints, points_color = 'blue'):
        # show Image
        # plot projection
        plt.figure("Camera Projection")
        plt.plot(imgpoints[0],imgpoints[1],'.',color = points_color)
        #we add a key point to help us see orientation of the points
        plt.plot(imgpoints[0,0],imgpoints[1,0],'.',color = 'blue')
        plt.xlim(0,self.img_width)
        plt.ylim(0,self.img_height)
        plt.gca().invert_yaxis()
        plt.show()

    def plot_plane(self, plane):
        if plane.type == 'rectangular':
            corners = plane.get_corners()
            img_corners = np.array(self.project(corners))
            img_corners =np.c_[img_corners,img_corners[:,0]]
            plt.plot(img_corners[0],img_corners[1])
        elif plane.type == 'circular':
            c = plane.circle
            c_projected = c.project(self.homography_from_Rt())
            c_projected.contour(grid_size=100)

    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
        # factor first 3*3 part
        K,R = rq(self.P[:,:3])
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if det(T) < 0:
            T[1,1] *= -1
        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(inv(self.K),self.P[:,3])
        return self.K, self.R, self.t

    def fov(self):
        """ Calculate field of view angles (grads) from camera matrix """
        fovx = np.rad2deg(2 * atan(self.img_width / (2. * self.fx)))
        fovy = np.rad2deg(2 * atan(self.img_height / (2. * self.fy)))
        return fovx, fovy

    def move(self, x,y,z):
        Rt = np.identity(4);
        Rt[:3,3] = np.array([x,y,z])
        self.P = np.dot(self.K, self.Rt)
        
    def rotate_camera(self, axis, angle):
        """ rotate camera around a given axis in CAMERA coordinate, please use following Rt"""
        R = self.rotation_matrix(axis, angle)
        newR = np.dot(R,self.R)
        self.Rt = np.dot(self.t, newR)
        self.R[:3,:3] = self.Rt[:3,:3]
        self.t[:3,3] = self.Rt[:3,3]
        self.set_P()

    def rotate(self, axis, angle):
        """ rotate camera around a given axis in world coordinates"""
        R = self.rotation_matrix(axis, angle)
        self.Rt = np.dot(R, self.Rt)
        self.R[:3,:3] = self.Rt[:3,:3]
        self.t[:3,3] = self.Rt[:3,3]
        # DO NOT forget to set new P
        self.set_P()

    def rotate_x(self,angle):
        self.rotate(np.array([1,0,0],dtype=np.float32), angle)

    def rotate_y(self,angle):
        self.rotate(np.array([0,1,0],dtype=np.float32), angle)

    def rotate_z(self,angle):
        self.rotate(np.array([0,0,1],dtype=np.float32), angle)

    def look_at(self, world_position):
      #%%
      world_position = self.get_world_position()[:3]
      eye = world_position
      target = np.array([0,0,0])
      up = np.array([0,1,0])

      zaxis = (target-eye)/np.linalg.norm(target-eye)
      xaxis = (np.cross(up,zaxis))/np.linalg.norm(np.cross(up,zaxis))
      yaxis = np.cross(zaxis, xaxis)

      R = np.eye(4)
      R = np.array([[xaxis[0], yaxis[0], zaxis[0], 0],
                   [xaxis[1], yaxis[1], zaxis[1], 0],
                   [xaxis[2], yaxis[2], zaxis[1], 0],
                   [       0,        0,        0, 1]]
          )

      R = np.array([[xaxis[0], xaxis[1], xaxis[2], 0],
                   [yaxis[0], yaxis[1], yaxis[2], 0],
                   [zaxis[0], zaxis[1], zaxis[2], 0],
                   [       0,        0,        0, 1]])


      t = np.eye(4, dtype=np.float32) # translation
      t[:3,3] = -eye

      self.R = R


      self.Rt = np.dot(R,t)
      self.t = np.eye(4, dtype=np.float32)
      self.t[:3,3] = self.Rt[:3,3]

    def homography_from_Rt(self):
      rt_reduced = self.Rt[:3,[0,1,3]]
      H = np.dot(self.K,rt_reduced)
      if H[2,2] != 0.:
        H = H/H[2,2]
      return H
  
class Sphere(object):
    """ Class for representing a 3D grid sphere based on radius r, and angles phi,theta."""
    def __init__(self ,radius=1.,origin=np.array([0., 0., 0.])):
        self.origin= origin
        self.radius= radius
        self.color = (1,0,0)
        self.sphere_points = None
        self.angle = 0.
        self.R = np.eye(4)
        self.type = 'sphere'
        
    def clone(self):
        new_sphere = Sphere()
        new_sphere.origin = self.origin
        new_sphere.radius=self.radius
        new_sphere.color = self.color
        new_sphere.angle = self.angle
        new_sphere.R = self.R
        return new_sphere   
    
    def random_points(self,p=6, r=3,min_dist=0.01):
        """
        p: ammount of points on sphere
        r: radius of sphere
        min_dist: minimum distance between each point
        """
        worldpoints=np.full((4,p),1.0)
        self.radius=r
        for k in range(0,p):
            theta=random.uniform(0.,math.pi)
            phi=random.uniform(0.,2*math.pi)
            radius=random.uniform(0,r)
            while(radius==0):
              radius=random.uniform(0,r)
            worldpoints[0,k]=radius*(math.sin(theta)*math.cos(phi))
            worldpoints[1,k]=radius*(math.sin(theta)*math.sin(phi))
            worldpoints[2,k]=radius*(math.cos(theta))
            if(k>1):
               for j in range(k-1,0,-1):
                distancex=(worldpoints[0,k]-worldpoints[0,j])*(worldpoints[0,k]-worldpoints[0,j])
                distancey=(worldpoints[1,k]-worldpoints[1,j])*(worldpoints[1,k]-worldpoints[1,j])
                distancez=(worldpoints[2,k]-worldpoints[2,j])*(worldpoints[2,k]-worldpoints[2,j])
                distance=math.sqrt(distancex+distancey+distancez)
                if (distance<min_dist):
                      theta=random.uniform(0.,math.pi)
                      phi=random.uniform(0.,2*math.pi)
                      radius=random.uniform(0,r)
                      while(radius==0):
                        radius=random.uniform(0,r)
                      worldpoints[0,k]=radius*(math.sin(theta)*math.cos(phi))
                      worldpoints[1,k]=radius*(math.sin(theta)*math.sin(phi))
                      worldpoints[2,k]=radius*(math.cos(theta))
                      j=k-1
        self.sphere_points=worldpoints
        return worldpoints
              
    def set_origin(self, origin):
        self.origin = origin
        
    def get_sphere_points(self):
       return np.copy(self.sphere_points) 
   
    def get_color(self):
       return self.color
   
    def set_color(self,color):
       self.color = color
   
    def plot_points(self):
        # show Image
        # plot projection
        plt.figure("Sphere points")
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        ax.scatter(self.sphere_points[:3,0],self.sphere_points[:3,1],self.sphere_points[:3,2],s=70, c='r') 
        ax.scatter(self.sphere_points[:3,3],self.sphere_points[:3,4],self.sphere_points[:3,5],s=70, c='r') 
        plt.show()
        
    def plot_sphere(self):
        phisim = np.linspace((-math.pi)/2.,(math.pi/2.))
        thetasim = np.linspace(0, 2 * np.pi)
        x = np.outer(np.sin(thetasim), np.cos(phisim))
        y = np.outer(np.sin(thetasim), np.sin(phisim))
        z = np.outer(np.cos(thetasim), np.ones_like(phisim))
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        ax.plot_wireframe(self.radius*x, self.radius*y, self.radius*z, color='g')    
  

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

""" Initial camera pose looking stratight down into the plane model """
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
cam.set_t(0.0, 0.0, 1.5, frame='world')

""" Plane for the control points """
sph = Sphere(2)
sph.set_color((1, 1, 0))
sph.random_points(5000, 1, 0.0001)

""" Plot camera axis and plane """
cams = [cam]
planes = [sph]
plot3D(cams, planes)
