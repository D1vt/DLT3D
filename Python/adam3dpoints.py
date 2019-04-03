#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 00:00:30 2019

@author: diana
"""

import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

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
    self.optimized_params = adam(objective_grad, self.init_params, step_size=0.01,
                                num_iters=100, callback = self.plot_points)
    
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
        #ax.plot_wireframe(sph.radius*x, sph.radius*y,
         #                 sph.radius*z, color='g')
        ax.scatter(pointscoord[:3, 0], pointscoord[:3, 1],
                   pointscoord[:3, 2], c='r')
        ax.scatter(pointscoord[:3, 3], pointscoord[:3, 4],
                   pointscoord[:3, 5], c='r')
        plt.show()
        


def flatten_points(points, type = 'image'):
  """
  Points:
    Array of n points in homogeneous coordinates
    example image points: np.array([[u1,v1,1], [u2,v2,1] ,....,[un,vn,1] ])
  Type:
    image: 2d image points in homogeneous coordinates [u,v,1]
    object: 3d world points in homogeneous coordinates [x,y,z,1]
    object_plane: 3d world points on a plane in homogeneous coordinates [x,y,0,1]
  """
  if type == 'image':
    # normalize and remove the homogeneous term
    n_points = np.copy(points/points[2,:])
    n_points = n_points[:2,:]

  if type == 'object':
    # normalize and remove the homogeneous term
    n_points = np.copy(points/points[3,:])
    n_points = n_points[:3,:]
  out = n_points.flatten('F')

  return out


def unflatten_points(points, type = 'image'):
  """
  Points:
    Array vector of flatened n points.
    example image points: np.array([u1,v1, u2,v2,....,un,vn])
  Type: output format
    image: 2d image points in homogeneous coordinates [u,v,1]
    object: 3d world points in homogeneous coordinates [x,y,z,1]
    object_plane: 3d world points on a plane in homogeneous coordinates [x,y,0,1]
  """
  if type == 'image':
    out = points.reshape((2,-1),order = 'F')
    out = np.vstack([out, np.ones(out.shape[1])])

  if type == 'object':
    out = points.reshape((3,-1),order = 'F')
    out = np.vstack([out, np.ones(out.shape[1])])

  return out


def normalise_points(pts):
    """
    Function translates and normalises a set of 2D or 3d homogeneous points
    so that their centroid is at the origin and their mean distance from
    the origin is sqrt(2).  This process typically improves the
    conditioning of any equations used to solve homographies, fundamental
    matrices etc.
    Inputs:
    pts: 3xN array of 2D homogeneous coordinates
    Returns:
    newpts: 3xN array of transformed 2D homogeneous coordinates.  The
            scaling parameter is normalised to 1 unless the point is at
            infinity.
    T: The 3x3 transformation matrix, newpts = T*pts
    """
    if pts.shape[0] == 4:
        pts = hom_3d_to_2d(pts)

    if pts.shape[0] != 3 and pts.shape[0] != 4  :
        print "Shape error"


    finiteind = np.nonzero(abs(pts[2,:]) > np.spacing(1))

    if len(finiteind[0]) != pts.shape[1]:
        print('Some points are at infinity')

    dist = []
    pts = pts/pts[2,:]
    for i in finiteind:
        #Replaced below for autograd
#        pts[0,i] = pts[0,i]/pts[2,i]
#        pts[1,i] = pts[1,i]/pts[2,i]
#        pts[2,i] = 1;

        c = np.mean(pts[0:2,i].T, axis=0).T

        newp1 = pts[0,i]-c[0]
        newp2 = pts[1,i]-c[1]

        dist.append(np.sqrt(newp1**2 + newp2**2))

    dist = np.array(dist)

    meandist = np.mean(dist)

    scale = np.sqrt(2)/meandist

    T = np.array([[scale, 0, -scale*c[0]], [0, scale, -scale*c[1]], [0, 0, 1]])

    newpts = np.dot(T,pts)


    return newpts, T

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

  #object_pts_norm = object_pts_norm/object_pts_norm[3,:]
  #image_pts_norm = image_pts_norm/image_pts_norm[2,:]
   
  AN = np.array([]).reshape([0,12])
  #A = np.array([])
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
#  rcond=1e-5
#  if s[-1] > rcond:
#    smalles_singular_value = s[-1]
#  else:
#    smalles_singular_value = s[-2]
  smallest_singular_value = s[11]
  return greatest_singular_value/smallest_singular_value


sim = OptimalPointsSim()
sim.run()
