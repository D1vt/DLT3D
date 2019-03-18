#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:42:40 2019

@author: diana
"""

import numpy as np
import random
import math

from matplotlib import pyplot as plt
#from vision.rt_matrix import *
#import matplotlib.pyplot as plt

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
            worldpoints[0,k]=radius*(math.sin(theta)*math.cos(phi))+self.origin[0]
            worldpoints[1,k]=radius*(math.sin(theta)*math.sin(phi))+self.origin[1]
            worldpoints[2,k]=radius*(math.cos(theta))+self.origin[2]
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
                      worldpoints[0,k]=radius*(math.sin(theta)*math.cos(phi))+self.origin[0]
                      worldpoints[1,k]=radius*(math.sin(theta)*math.sin(phi))+self.origin[1]
                      worldpoints[2,k]=radius*(math.cos(theta))+self.origin[2]
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

S=Sphere()
worldpoints = S.random_points(6,2) 
S.plot_sphere()
Sph = Sphere(radius=0.5)
Sph.set_color((1, 1, 0))
Sph.random_points(6,2, 0.001)       
