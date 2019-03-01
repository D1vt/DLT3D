# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 18:42:35 2019
/
@author: diana
"""

import numpy as np 
from random import randrange
from numpy import linalg as LA
from scipy.linalg import expm, inv
import math



def R_Euler(a,b,c):
    
    #Euler as defined in page 51 of thesis Tracking Errors in Laparoscopic Surgeries
    a=math.atan(cam.R[2,1]/cam.R[2,2])
    b=-math.asin(cam.R[2,0])
    c=math.atan(cam.R[1,0]/cam.R[0,0])
    #print a,b,c
    R_Eu_a=np.array([[1.,0.,0.],
                     [0.,math.cos(a),-math.sin(a)],
                     [0.,math.sin(a),math.cos(a)]])
    R_Eu_b=np.array([[math.cos(b),0.,math.sin(b)],
                      [0.,1.,0.],
                      [-math.sin(b),0.,math.cos(b)]])
    R_Eu_c=np.array([[math.cos(c),-math.sin(c),0.],
                      [math.sin(c),math.cos(c),0.],
                      [0.,0.,1.]])
    R_Eu1=np.dot(R_Eu_c,R_Eu_b)
    R_Eu=np.dot(R_Eu1,R_Eu_a)
    
    return R_Eu

