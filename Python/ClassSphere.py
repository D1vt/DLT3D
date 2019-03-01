Created on Fri Feb 28 21:42:40 2019

@author: diana
"""

import numpy as np
import random
import math

class Sphere(object):
    """ Class for representing a 3D grid sphere based on radius r, and angles phi,theta."""
    def __init__(self, origin=np.array([0., 0., 0.])):
        self.origin= origin
        self.sphere_points = None
        self.type = 'sphere'
    
    def get_points(self,p=6, r=3,min_dist=0.01):
        """
        p: ammount of points on sphere
        r: radius of sphere
        min_dist: minimum distance between each point
        """
        wrldpoints=np.full((4,p),1.0)
        
        for k in range(0,p):
            theta=random.uniform(0.,math.pi)
            phi=random.uniform(0.,2*math.pi)
            radius=random.uniform(0,r)
            while(radius==0):
              radius=random.uniform(0,r)
            wrldpoints[0,k]=radius*(math.sin(theta)*math.cos(phi))
            wrldpoints[1,k]=radius*(math.sin(theta)*math.sin(phi))
            wrldpoints[2,k]=radius*(math.cos(theta))
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
        return wrldpoints
              
    def set_origin(self, origin):
        self.origin = origin

    def get_sphere_points(self):
       return np.copy(self.sphere_points)         
   
S=Sphere()
worldpoints = S.get_points(6,2) 
        
