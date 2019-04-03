#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 00:00:30 2019

@author: diana
"""

import autograd.numpy as np
from autograd import grad
from vision.camera import Camera
from python.sphere import Sphere
from autograd.misc.optimizers import adam
from optimize.utils import flatten_points, unflatten_points, normalise_points

class OptimalPointsSim(object):
  """ Class that defines and optimization to obtain optimal control points
  configurations for homography and plannar pose estimation. """

  def __init__(self):
    """ Definition of a simulated camera """
    self.cam = Camera()
    self.cam.set_K(fx = 100,fy = 100,cx = 640,cy = 480)
    self.cam.set_width_heigth(1280,960)

    """ Initial camera pose looking stratight down into the plane model """
    self.cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
    self.cam.set_t(0.0,0.0,1.5, frame='world')

    """ Plane for the control points """
    self.sph = Sphere(radius=0.5)
    self.sph.random_points(p =6, r = 0.5, min_dist = 0.001)

  def run(self):
    self.objectPoints = self.sph.get_sphere_points()
    self.init_params = flatten_points(self.objectPoints, type='object')

    self.objective1 = lambda params: matrix_condition_number_autograd(params, self.cam.P, normalize = False)
    self.objective2 = lambda params,iter: matrix_condition_number_autograd(params, self.cam.P, normalize = True)

    print("Optimizing condition number...")
    objective_grad = grad(self.objective2)
    self.optimized_params = adam(objective_grad, self.init_params, step_size=0.001,
                                num_iters=200, callback = self.plot_points)
    
  def plot_points(self,params, iter, gradient):
        phisim = np.linspace((-math.pi)/2., (math.pi/2.))
        thetasim = np.linspace(0, 2 * np.pi)
        print params
        pointscoord=np.full((3,6),0.0)
        
        for i in range (6):
            
           pointscoord[0,i]=params[i]
           pointscoord[1,i]=params[i+1]
           pointscoord[2,i]=params[i+2]

        x = np.outer(np.sin(thetasim), np.cos(phisim))
        y = np.outer(np.sin(thetasim), np.sin(phisim))
        z = np.outer(np.cos(thetasim), np.ones_like(phisim))
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_wireframe(sph.radius*x, sph.radius*y,
                          sph.radius*z, color='g')
        ax.scatter(pointscoord[:3, 0], pointscoord[:3, 1],
                   pointscoord[:3, 2], c='r')
        ax.scatter(pointscoord[:3, 3], pointscoord[:3, 4],
                   pointscoord[:3, 5], c='r')
        plt.show()
        

def hom_3d_to_2d(pts):
    pts = pts[[0,1,3],:]
    return pts

def hom_2d_to_3d(pts):
    pts = np.insert(pts,2,np.zeros(pts.shape[1]),0)
    return pts

def calculate_AN_matrix_autograd(params,P,normalize=False):
  P = np.array(P)
  n_points = params.shape[0]/3

  object_pts = unflatten_points(params, type='object')
  image_pts = np.array(np.dot(P,object_pts))
  image_pts = image_pts/image_pts[2,:]

  if normalize:
   object_pts_norm,T1 = normalise_points(object_pts)
   image_pts_norm,T2 = normalise_points(image_pts)
  else:
   object_pts_norm = object_pts[[0,1,3],:]
   image_pts_norm = image_pts
   
  AN = np.array([]).reshape([0,12])
  for i in range(n_points):
      x = object_pts_norm[0,i]
      y = object_pts_norm[1,i]
      z= object_pts_norm[2,i]
      u = image_pts_norm[0,i]
      v = image_pts_norm[1,i]

      row1 = np.array([x, y, z, 1., 0., 0., 0., 0., -u*x, -u*y, -u*z, -u])
      row2 = np.array([0., 0., 0., 0., x, y, z, 1., -v*x, -v*y, -u*z, -v])

      AN = np.vstack([AN, row1])
      AN = np.vstack([AN, row2])
  return AN


def matrix_condition_number_autograd(params,P,normalize = False):
  AN = calculate_AN_matrix_autograd(params,P, normalize)
  U, s, V = np.linalg.svd(AN,full_matrices=False)
  greatest_singular_value = s[0]
  smallest_singular_value = s[11]
  return greatest_singular_value/smallest_singular_value


sim = OptimalPointsSim()
sim.run()
