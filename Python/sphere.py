#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:42:40 2019

@author: diana
"""

import numpy as np
import random
import math
from matplotlib import pyplot as plt

class Sphere(object):
    """
    Class for representing a 3D grid sphere based on radius r,
    and angles phi,theta.
    """
    def __init__(self, radius=0.5, origin=np.array([0., 0., 0.])):
        self.origin = origin
        self.radius = radius
        self.color = (1, 0, 0)
        self.sphere_points = None
        self.angle = 0.
        self.R = np.eye(4)
        self.type = 'sphere'

    def clone(self):
        new_sphere = Sphere()
        new_sphere.origin = self.origin
        new_sphere.radius = self.radius
        new_sphere.color = self.color
        new_sphere.sphere_points = self.sphere_points
        new_sphere.angle = self.angle
        new_sphere.R = self.R
        new_sphere.type = self.type
        return new_sphere

     def random_points(self, p=6, r=0.3, min_dist=0.016, periphery=False):
        """
        p: ammount of points on sphere
        r: radius of sphere in meters
        min_dist: minimum distance between each point
        periphery: True = all the tandomly selected points are belonging to the periphery of the sphere
                   False = the points are randomly selected from the sphere plane
        """
        worldpoints = np.full((4, p), 1.0)
        self.radius = r
        for k in range(0, p):
            theta = random.uniform(0., math.pi)
            phi = random.uniform(0., 2*math.pi)
            if (periphery):
                radius = r
            else:
                radius = random.uniform(0, r)
                while(radius == 0):
                    radius = random.uniform(0, r)
            worldpoints[0, k] = (radius*(math.sin(theta)*math.cos(phi))
                                 + self.origin[0])
            worldpoints[1, k] = (radius*(math.sin(theta)*math.sin(phi))
                                 + self.origin[1])
            worldpoints[2, k] = radius*(math.cos(theta))+self.origin[2]
            if(k > 1):
                for j in range(k-1, 0, -1):
                    distancex = ((worldpoints[0, k]-worldpoints[0, j])
                                 * (worldpoints[0, k]-worldpoints[0, j]))
                    distancey = ((worldpoints[1, k]-worldpoints[1, j])
                                 * (worldpoints[1, k]-worldpoints[1, j]))
                    distancez = ((worldpoints[2, k]-worldpoints[2, j])
                                 * (worldpoints[2, k]-worldpoints[2, j]))
                    distance = math.sqrt(distancex+distancey+distancez)
                if (distance < min_dist):
                    theta = random.uniform(0., math.pi)
                    phi = random.uniform(0., 2*math.pi)
                    radius = random.uniform(0, r)
                    # radius = r
                    while(radius == 0):
                        radius = random.uniform(0, r)
                    worldpoints[0, k] = radius*(math.sin(theta)*math.cos(phi))
                    worldpoints[1, k] = radius*(math.sin(theta)*math.sin(phi))
                    worldpoints[2, k] = radius*(math.cos(theta))
                    j = k-1
        self.sphere_points = worldpoints
        return worldpoints
    
    def random_prism(self, r):
            """
            In the thesis the distance between 2 markers is set as 10 cm = 0.1 m
            This function creates only sets of 6 points configurations (prisms) 
            """
            prismpoints = np.full((4, 6), 1.0)
            theta = random.uniform(0., math.pi)
            phi = random.uniform(0., 2*math.pi)
            radius = r
            prismpoints[0, 0] = (radius*(math.sin(theta)*math.cos(phi))
                                 + self.origin[0])
            prismpoints[1, 0] = (radius*(math.sin(theta)*math.sin(phi))
                                 + self.origin[1])
            prismpoints[2, 0] = radius*(math.cos(theta))+self.origin[2]
            prismpoints[0, 1] = prismpoints[0, 0]
            prismpoints[1, 1] = prismpoints[1, 0]-0.10 
            prismpoints[2, 1] = prismpoints[2, 0]
            prismpoints[0, 2] = prismpoints[0, 0]-0.10
            prismpoints[1, 2] = prismpoints[1, 0]
            prismpoints[2, 2] = prismpoints[2, 0]
            prismpoints[0, 3] = prismpoints[0, 0]-0.1
            prismpoints[1, 3] = prismpoints[1, 0]-0.1
            prismpoints[2, 3] = prismpoints[2, 0] +0.05
            prismpoints[0, 4] = prismpoints[0, 0]- 0.05
            prismpoints[1, 4] = prismpoints[1, 0] -0.1
            prismpoints[2, 4] = prismpoints[2, 0] + 0.05
            prismpoints[0, 5] = - prismpoints[0, 0]- 0.05
            prismpoints[1, 5] = - prismpoints[1, 0] -0.1
            prismpoints[2, 5] = prismpoints[2, 0] + 0.05
            """
            the below comments are needed if we want to plot the prism
            """
            #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            #ax.scatter(prismpoints[:3, 0], prismpoints[:3, 1],
             #          prismpoints[:3, 2], s=80, c='r')
            #ax.scatter(prismpoints[:3, 3],prismpoints[:3, 4], prismpoints[:3, 5],
             #          s=80, c='r')
            #ax.set_xlabel('x label')
            #ax.set_ylabel('y label')
            #ax.set_zlabel('z label')
            #plt.show()  
            return prismpoints

    def set_origin(self, origin):
        self.origin = origin

    def get_sphere_points(self):
        return np.copy(self.sphere_points)

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def plot_points(self):
        # show Image
        # plot projection
        plt.figure("Sphere points")
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.scatter(self.sphere_points[:3, 0], self.sphere_points[:3, 1],
                   self.sphere_points[:3, 2], s=70, c='r')
        ax.scatter(self.sphere_points[:3, 3], self.sphere_points[:3, 4],
                   self.sphere_points[:3, 5], s=70, c='r')
        plt.show()

    def plot_sphere(self):
        phisim = np.linspace((-math.pi)/2., (math.pi/2.))
        thetasim = np.linspace(0, 2 * np.pi)
        x = np.outer(np.sin(thetasim), np.cos(phisim))
        y = np.outer(np.sin(thetasim), np.sin(phisim))
        z = np.outer(np.cos(thetasim), np.ones_like(phisim))
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_wireframe(self.radius*x, self.radius*y,
                          self.radius*z, color='g')

    def plot_sphere_and_points(self, pointscoord):
        phisim = np.linspace((-math.pi)/2., (math.pi/2.))
        thetasim = np.linspace(0, 2 * np.pi)
        x = np.outer(np.sin(thetasim), np.cos(phisim))
        y = np.outer(np.sin(thetasim), np.sin(phisim))
        z = np.outer(np.cos(thetasim), np.ones_like(phisim))
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_wireframe(self.radius*x, self.radius*y,
                          self.radius*z, color='g')
        ax.scatter(pointscoord[:3, 0], pointscoord[:3, 1],
                   pointscoord[:3, 2], s=80, c='r')
        ax.scatter(pointscoord[:3, 3], pointscoord[:3, 4],
                   pointscoord[:3, 5], s=80, c='r')
        plt.show()

# Test ------------------------------------------------------------------------


S = Sphere()


worldpoints = S.random_points(6, 2)
S.plot_sphere()
Sph = Sphere(radius=0.5)
Sph.set_color((1, 1, 0))
Sph.random_points(6, 2, 0.001)

