#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:41:33 2019
/
@author: diana
@class Camera & set_P, set_K, set_R_axisAngle, set_t, project developed by raultron
"""
import numpy as np 
import scipy.linalg
from random import randrange
import random
from numpy import linalg as LA
from scipy.linalg import expm, inv
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import csv
with open('dataA.csv', 'w') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=' ')
with open('dataR.csv', 'w') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=' ')
with open('dataerror.csv', 'w') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=' ')          


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

    def set_P(self):
        # P = K[R|t]
        # P is a 3x4 Projection Matrix (from 3d euclidean to image)
        #self.Rt = np.hstack((self.R, self.t))
        self.P = np.dot(self.K, self.Rt[:3,:4])
    
    def set_K(self, fx = 1, fy = 1, cx = 0,cy = 0):
        # K is the 3x3 Camera matrix
        # fx, fy are focal lenghts expressed in pixel units
        # cx, cy is a principal point usually at image center
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.mat([[fx, 0., cx],
                      [0.,fy,cy],
                      [0.,0.,1.]], dtype=np.float32)
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
        a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
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
          cam_world = np.array([x,y,z,1.]).T
          cam_t = np.dot(self.R,-cam_world)
          self.t[:3,3] = cam_t[:3]
        else:
          self.t[:3,3] = np.array([x,y,z])
        self.update_Rt()



    def get_world_position(self):
        t = np.dot(inv(cam.Rt), np.array([0.,0.,0.,1.]))
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

   
def spherepoints(points,r):
    worldpoints=np.full((4,points),1.0)
    for k in range(points):
     theta=random.uniform(0.,math.pi)
     phi=random.uniform(0.,2*math.pi)
     radius=random.uniform(0,r)
     while(radius==0):
         radius=random.uniform(0,r)
     worldpoints[0,k]=radius*(math.sin(theta)*math.cos(phi))
     worldpoints[1,k]=radius*(math.sin(theta)*math.sin(phi))
     worldpoints[2,k]=radius*(math.cos(theta))
    #phisim = np.linspace((-math.pi)/2.,(math.pi/2.))
    #thetasim = np.linspace(0, 2 * np.pi)
    #x = np.outer(radius*np.sin(thetasim), radius*np.cos(phisim))
    #y = np.outer(radius*np.sin(thetasim), radius*np.sin(phisim))
    #z = np.outer(radius*np.cos(thetasim), np.ones_like(phisim))
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(radius*x, radius*y, radius*z, color='g')
    ax.scatter(worldpoints[:3,1],worldpoints[:3,2],worldpoints[:3,3],s=70, c='r') 
    ax.scatter(worldpoints[:3,0],worldpoints[:3,4],worldpoints[:3,5],s=70, c='r') 
    plt.show()
    #worldpoints=np.vstack((worldpoints,[1.0,1.0,1.0,1.0,1.0,1.0]))
    return worldpoints

def normalize(points,size=2):
    #https://inside.mines.edu/~whoff/courses/EENG512/lectures/18-LinearPoseEstimation.pdf
     pointsnorm=np.dot(inv(cam.K),points)
     return pointsnorm  

def DLT3D(self,worldpoints, imagepoints, normalization=False):
    #if odd row 0,0,0,0,xi,yi,zi,1,-vixi,-viyi,-vizi,-vi
    #if even row : Χι,Υι,Ζι,1,0,0,0,0,-uixi,-uiyi,-uizi,-ui
    if (normalization==False):
       imagepoints=normalize(imagepoints)
       
    
        
    A=np.array([[worldpoints[0,0],worldpoints[1,0],worldpoints[2,0],1.,0.,0.,0.,0.,-imagepoints[0,0]*worldpoints[0,0],-imagepoints[0,0]*worldpoints[1,0],-imagepoints[0,0]*worldpoints[2,0],-imagepoints[0,0]],
               [0.,0.,0.,0.,worldpoints[0,0],worldpoints[1,0],worldpoints[2,0],1.,-imagepoints[1,0]*worldpoints[0,0],-imagepoints[1,0]*worldpoints[1,0],-imagepoints[1,0]*worldpoints[2,0],-imagepoints[1,0]], 
               [worldpoints[0,1],worldpoints[1,1],worldpoints[2,1],1.,0.,0.,0.,0.,-imagepoints[0,1]*worldpoints[0,1],-imagepoints[0,1]*worldpoints[1,1],-imagepoints[0,1]*worldpoints[2,1],-imagepoints[0,1]],                              
               [0.,0.,0.,0.,worldpoints[0,1],worldpoints[1,1],worldpoints[2,1],1.,-imagepoints[1,1]*worldpoints[0,1],-imagepoints[1,1]*worldpoints[1,1],-imagepoints[1,1]*worldpoints[2,1],-imagepoints[1,1]],
               [worldpoints[0,2],worldpoints[1,2],worldpoints[2,2],1.,0.,0.,0.,0.,-imagepoints[0,2]*worldpoints[0,2],-imagepoints[0,2]*worldpoints[1,2],-imagepoints[0,2]*worldpoints[2,2],-imagepoints[0,2]],
               [0.,0.,0.,0.,worldpoints[0,2],worldpoints[1,2],worldpoints[2,2],1.,-imagepoints[1,2]*worldpoints[0,2],-imagepoints[1,2]*worldpoints[1,2],-imagepoints[1,2]*worldpoints[2,2],-imagepoints[1,2]],
               [worldpoints[0,3],worldpoints[1,3],worldpoints[2,3],1.,0.,0.,0.,0.,-imagepoints[0,3]*worldpoints[0,3],-imagepoints[0,3]*worldpoints[1,3],-imagepoints[0,3]*worldpoints[2,3],-imagepoints[0,3]],
               [0.,0.,0.,0.,worldpoints[0,3],worldpoints[1,3],worldpoints[2,3],1.,-imagepoints[1,3]*worldpoints[0,3],-imagepoints[1,3]*worldpoints[1,3],-imagepoints[1,3]*worldpoints[2,3],-imagepoints[1,3]],
               [worldpoints[0,4],worldpoints[1,4],worldpoints[2,4],1.,0.,0.,0.,0.,-imagepoints[0,4]*worldpoints[0,4],-imagepoints[0,4]*worldpoints[1,4],-imagepoints[0,4]*worldpoints[2,4],-imagepoints[0,4]],
               [0.,0.,0.,0.,worldpoints[0,4],worldpoints[1,4],worldpoints[2,4],1.,-imagepoints[1,4]*worldpoints[0,4],-imagepoints[1,4]*worldpoints[1,4],-imagepoints[1,4]*worldpoints[2,4],-imagepoints[1,4]],
               [worldpoints[0,5],worldpoints[1,5],worldpoints[2,5],1.,0.,0.,0.,0.,-imagepoints[0,5]*worldpoints[0,5],-imagepoints[0,5]*worldpoints[1,5],-imagepoints[0,5]*worldpoints[2,5],-imagepoints[0,5]],
               [0.,0.,0.,0.,worldpoints[0,5],worldpoints[1,5],worldpoints[2,5],1.,-imagepoints[1,5]*worldpoints[0,5],-imagepoints[1,5]*worldpoints[1,5],-imagepoints[1,5]*worldpoints[2,5],-imagepoints[1,5]]])
   
     
    U1, s, Vh = np.linalg.svd(A)
   
    Vh=np.transpose(Vh)
    H=Vh[:,11].reshape(3,4)  
       
    if (normalization==False):
        H=np.dot(self.K,H)
   
    return H      



#a function for the estimation of R,t using the V matrix from the DLT algorithm  
def estimate_t(self,cam_center):
    #source: Multiple View Geometry Richard Hartley and Andrew Zisserman, p.15 decompose P to K,R,t matrices
    est_trans=np.array([[1.,0.,0.,abs(cam_center[0])],
                    [0.,1.,0.,abs(cam_center[1])],
                    [0.,0.,1.,abs(cam_center[2])],
                    [0.,0.,0.,1.]])

    return est_trans
       

#find camera center source: https://s3.amazonaws.com/content.udacity-data.com/courses/ud810/slides/Unit-3/3C-L3.pdf
def camera_center(H):
    Q=np.array([[H[0,0],H[0,1],H[0,2]],
               [H[1,0],H[1,1],H[1,2]],
               [H[2,0],H[2,1],H[2,2]]])
    b=np.array([[H[0,3]],
                [H[1,3]],
                [H[2,3]]])
    Q=-inv(Q)
  
    cam_center=np.dot(Q,b)
   
    return cam_center


    #adding noise to each u,v of the image (random gaussian noise)
def add_noise(imagepoints,sd=4.,mean=0., size=10000):
   if size==0:
    imagenoise=np.full((3,6),1.0)
    
    imagenoise[0,0]=imagepoints[0,0] +np.random.normal(mean,sd)
    imagenoise[1,0]=imagepoints[1,0] +np.random.normal(mean,sd)
    imagenoise[0,1]= imagepoints[0,1] +np.random.normal(mean,sd)
    imagenoise[1,1]=imagepoints[1,1] +np.random.normal(mean,sd)
    imagenoise[0,2]=imagepoints[0,2] +np.random.normal(mean,sd)
    imagenoise[1,2]=imagepoints[1,2] +np.random.normal(mean,sd)
    imagenoise[0,3]=imagepoints[0,3] +np.random.normal(mean,sd)
    imagenoise[1,3]=imagepoints[1,3] +np.random.normal(mean,sd)
    imagenoise[0,4]=imagepoints[0,4] +np.random.normal(mean,sd)
    imagenoise[1,4]=imagepoints[1,4] +np.random.normal(mean,sd)
    imagenoise[0,5]=imagepoints[0,5] +np.random.normal(mean,sd)
    imagenoise[1,5]=imagepoints[1,5] +np.random.normal(mean,sd)
    return imagenoise
   else: 
    px1=imagepoints[0,0] +np.random.normal(mean,sd,size)
    py1=imagepoints[1,0] +np.random.normal(mean,sd,size)
    px2=imagepoints[0,1] +np.random.normal(mean,sd,size)
    py2=imagepoints[1,1] +np.random.normal(mean,sd,size)
    px3=imagepoints[0,2] +np.random.normal(mean,sd,size)
    py3=imagepoints[1,2] +np.random.normal(mean,sd,size)
    px4=imagepoints[0,3] +np.random.normal(mean,sd,size)
    py4=imagepoints[1,3] +np.random.normal(mean,sd,size)
    px5=imagepoints[0,4] +np.random.normal(mean,sd,size)
    py5=imagepoints[1,4] +np.random.normal(mean,sd,size)
    px6=imagepoints[0,5] +np.random.normal(mean,sd,size)
    py6=imagepoints[1,5] +np.random.normal(mean,sd,size)
    imageNoise=np.array([[px1,px2,px3,px4,px5,px6],
                        [py1,py2,py3,py4,py5,py6]])
    return imageNoise        

def stand_add_noise(imagepoints,noise):
   
    imagenoise=np.full((3,6),1.0)
    
    imagenoise[0,0]=imagepoints[0,0] +noise[0,0]
    imagenoise[1,0]=imagepoints[1,0] +noise[1,0]
    imagenoise[0,1]= imagepoints[0,1] +noise[0,1]
    imagenoise[1,1]=imagepoints[1,1] +noise[1,1]
    imagenoise[0,2]=imagepoints[0,2] +noise[0,2]
    imagenoise[1,2]=imagepoints[1,2] +noise[1,2]
    imagenoise[0,3]=imagepoints[0,3] +noise[0,3]
    imagenoise[1,3]=imagepoints[1,3] +noise[1,3]
    imagenoise[0,4]=imagepoints[0,4] +noise[0,4]
    imagenoise[1,4]=imagepoints[1,4] +noise[1,4]
    imagenoise[0,5]=imagepoints[0,5] +noise[0,5]
    imagenoise[1,5]=imagepoints[1,5] +noise[1,5]
    return imagenoise
        
    
def DLTproject(H,X, quant_error=False):
        """  Project points in X (4*n array) and normalize coordinates. """
        #set_P(cam)
        x = np.dot(H,X)
        for i in range(x.shape[1]):
          x[:,i] /= x[2,i]
        if(quant_error):
            x = np.around(x, decimals=0)
        return x        
    
  ### here we solve the problem using spherical coordinates and Euler angles

#calculate the a,b,c 
def a_b_r_spherical(self):
    world_position = self.get_world_position()  
    x = world_position[0]
    y = world_position[1]
    z = world_position[2]
    r = np.sqrt(x * x + y * y + z * z,dtype=np.float32)
    if r ==0:
        a = np.deg2rad(0.0)
    else:
        a = math.acos(z / r)
    if x == 0:
        b = np.deg2rad(90.0)
    else:
       b = math.atan(y / x)
    return a,b,r

#a function to calculate cov matrix . The equation for the final cov matrix (given in Thesis Tracking Errors in AR (eq. 5.8) = inv(np.dot(np.dot(JacT,image6points), Jac)), where JacT is the transpose matrix of Jacobian matrix (in spherical coordinates) 
def covariance_matrix_p(self,worldpoints,imagepoints,a,b,c):
    #calculate the a,b,c in spherical coord. 
   
    Jac,JacT=jacobian_matrix(self,a,b,c,worldpoints)
    
    #adding noise to imagePoints
    points=add_noise(imagepoints)
    
    
#Calculate the cov. matrix for each point in order to create the Σvi (eq. 5.8)
    cov_mat_p1 = np.cov(points[0,0],points[1,0])
    cov_mat_p2 = np.cov(points[0,1],points[1,1])
    cov_mat_p3 = np.cov(points[0,2],points[1,2])
    cov_mat_p4 = np.cov(points[0,3],points[1,3])
    cov_mat_p5 = np.cov(points[0,4],points[1,4])
    cov_mat_p6 = np.cov(points[0,5],points[1,5])
    
    
    #create the 12*12 cov matrix, that has 0 everywhere except in the main diagonal
    image6points = np.block([[cov_mat_p1, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                                      [np.zeros((2, 2)), cov_mat_p2, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                                       [np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p3, np.zeros((2, 2)),np.zeros((2, 2)),np.zeros((2, 2))],
                                       [np.zeros((2, 2)), np.zeros((2, 2)),np.zeros((2, 2)),cov_mat_p4, np.zeros((2, 2)), np.zeros((2, 2))],
                                       [np.zeros((2, 2)), np.zeros((2, 2)),np.zeros((2, 2)),np.zeros((2, 2)), cov_mat_p5, np.zeros((2, 2))],
                                       [np.zeros((2, 2)), np.zeros((2, 2)),np.zeros((2, 2)),np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p6]])
    image6points=np.diag(np.diag( image6points))
    #the eq. 5.8 uses the inv. cov. matrix of those points
    image6points=inv(image6points)
   
    #find the final 3*3 matrix
    cov_mat = inv(np.dot(np.dot(JacT,image6points), Jac))
    #print image6points
    return cov_mat

#here I calculate the R from Euler equations and also the following: dR/da , dR/db , dR/dc as I am going to use them to calculate the Jacobian Matrix 
def R_Euler():
   
    
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


 
#I am using this function to find the u&v of each point (p. 81-82 in "Tracking Errors in AI")      
def jacobian_uv(X,P,P_da,P_db,P_dc):
     Y=np.array([[X[0]],
                [X[1]],
                [X[2]],
                [1.]])
     
     #image_points=np.dot(P,Y)
     
     image_point_da = np.dot(P_da,Y)
     image_point_db = np.dot(P_db,Y)
     image_point_dc = np.dot(P_dc,Y)
     u_da=  image_point_da[0] #/ image_points[2]
     v_da=  image_point_da[1] #/image_points[2]
     u_db=image_point_db[0] #/image_points[2]
     v_db=image_point_db[1] #/image_points[2]
     u_dc=image_point_dc[0] #/image_points[2]
     v_dc=image_point_dc[1] #/image_points[2]
     return float(u_da),float(u_db),float(u_dc),float(v_da),float(v_db),float(v_dc) #2*3
 
        
#here I calculate Jacobian matrix and transpose of the Jacobian matrix. As I have to calculate the Jacobian Matrix I need to calculate the following: dR/da, dR/db,dR/dc , dt/da , dt/db, dt/dc 
def jacobian_matrix(self,a,b,r,worldpoints):
    
     R_Sphere = np.array([[math.cos(b)*math.cos(a), -math.sin(b), math.cos(b)*math.sin(a)],
                       [math.sin(b)*math.cos(a), math.cos(b), math.sin(b)*math.sin(a)],
                       [math.sin(a), 0., math.cos(a)]])
     
     dR_da=np.array([[math.cos(b)*(-math.sin(a)),0., math.cos(b)*math.cos(a)], #,c*(-math.cos(a))*math.cos(b),
                          [math.sin(b)*(-math.sin(a)), 0., math.sin(b)*math.cos(a)], #c*math.sin(b)*math.cos(a),
                          [(math.cos(a)), 0.,(-math.sin(a))] ]) #,c*(-math.sin(a))]])
     dR_db=np.array([[(-math.sin(b))*math.cos(a), -math.cos(b), (-math.sin(b))*math.sin(a)],
                          [math.cos(b)*math.cos(a), (-math.sin(b)),(math.cos(b))*math.sin(a)],
                          [0., 0., 0.]])
     dR_dr=np.full((3,3),0.) 
     R_Eu=R_Euler()
     R_Sphere_da=np.dot(dR_da,R_Eu[:3,:3])
     R_Sphere_db=np.dot(dR_db,R_Eu[:3,:3])
     R_Sphere_dr=np.dot(dR_dr,R_Eu[:3,:3])
     R_Sphere = np.dot(R_Sphere, R_Eu[:3, :3])
     t_Sphere= np.array([[r*math.sin(a)*math.cos(b)],
            [r*math.sin(b)*math.sin(a)],
             [r*math.cos(a)]])
     t_da=np.array([[r*math.cos(a)*math.cos(b)],
            [r*math.sin(b)*math.cos(a)],
             [r*(-math.sin(a))]])
     t_db=np.array([[r*math.sin(a)*(-math.sin(b))],
            [r*math.cos(b)*math.sin(a)],
             [0.]])
     t_dr=np.array([[math.sin(a)*math.cos(b)],
            [math.sin(b)*math.sin(a)],
             [math.cos(a)]])
     t_Sphere_da=np.dot(R_Sphere_da, -t_Sphere)+np.dot(R_Sphere, -t_da)
     t_Sphere_db=np.dot(R_Sphere_db, -t_Sphere)+np.dot(R_Sphere, -t_db)
     t_Sphere_dr=np.dot(R_Sphere_dr, -t_Sphere)+np.dot(R_Sphere, -t_dr)
     t_Sphere=np.dot(R_Sphere,-t_Sphere)
     #then I am using them to find the following: dRt/da , dRt/db, dRt/dc where a,b,c were calculated in a previous function
     dRt_da=np.hstack((R_Sphere_da, t_Sphere_da))
     dRt_db=np.hstack((R_Sphere_db, t_Sphere_db))
     dRt_dr=np.hstack((R_Sphere_dr, t_Sphere_dr))
     Rt_sphere=np.hstack((R_Sphere, t_Sphere))
     #print Rt_sphere
     #I am creating the non linear camera equations to find each point (u,v) page 81 in : "Tracking Errors in AR"
     P_da=np.dot(self.K,dRt_da)
     P_db=np.dot(self.K,dRt_db)
     P_dr=np.dot(self.K,dRt_dr)
     P=np.dot(self.K,Rt_sphere)
      
     #then for each one of the world points I find the u,v and use them to calculate the Jacobian Matrix. So I need to calculate the followin for each point: du/da, dv/da, du/db,dv/db,du/dc,dv/dc
     u1a,u1b,u1c,v1a,v1b,v1c=jacobian_uv(worldpoints[0,:],P,P_da,P_db,P_dr)
     u2a,u2b,u2c,v2a,v2b,v2c=jacobian_uv(worldpoints[1,:],P,P_da,P_db,P_dr)
     u3a,u3b,u3c,v3a,v3b,v3c=jacobian_uv(worldpoints[2,:],P,P_da,P_db,P_dr)
     u4a,u4b,u4c,v4a,v4b,v4c=jacobian_uv(worldpoints[3,:],P,P_da,P_db,P_dr)
     u5a,u5b,u5c,v5a,v5b,v5c=jacobian_uv(worldpoints[4,:],P,P_da,P_db,P_dr)
     u6a,u6b,u6c,v6a,v6b,v6c=jacobian_uv(worldpoints[5,:],P,P_da,P_db,P_dr)
    
     
     #3*12 Jacobian.Transpose
    # JacT=np.array([[u1a,u2a,u3a,u4a,u5a,u6a,v1a,v2a,v3a,v4a,v5a,v6a],
                #   [u1b,u2b,u3b,u4b,u5b,u6b,v1b,v2b,v3b,v4b,v5b,v6b],
                 #  [u1c,u2c,u3c,u4c,u5c,u6c,v1c,v2c,v3c,v4c,v5c,v6c]])/
     
    
     #12*3 Jacobian Matrix
     Jac=np.array([[u1a,u1b,u1c],
                    [u2a,u2b,u2c],
                    [u3a,u3b,u3c],
                    [u4a,u4b,u4c],
                    [u5a,u5b,u5c],
                    [u6a,u6b,u6c],
                    [v1a,v1b,v1c],
                    [v2a,v2b,v2c],
                    [v3a,v3b,v3c],
                    [v4a,v4b,v4c],
                    [v5a,v5b,v5c],
                    [v6a,v6b,v6c]])
     JacT=np.transpose(Jac)
     #print JacT 
    
     return Jac,JacT
 
def calculate_best_a(self,worldpoints,imagepoints,b,r):
   #1.57079632679
    best=2*math.pi
    worst=best
    mincond=1000000000.
    maxcond=-1
    
 #a is limited from -pi/2 to p/2. Minimum cond number of the cov matrix == best a angle and Maximum cond number of the cov matrix==worst a angle
    for a in np.arange(-90., 95.,5.):
 
        covmat=(covariance_matrix_p(self,worldpoints,imagepoints,np.rad2deg(a),b,r))
        cond=LA.cond(covmat)
        with open('dataA.csv', 'ab') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=' ')
          filewriter.writerow([cond , a])
          
        if cond<=mincond:
            mincond=cond
            best=a
        if cond>=maxcond:
            maxcond=cond
            worst=a
    x = []
    y = []
    with open('dataA.csv','r') as csvfile:
         plots = csv.reader(csvfile, delimiter=' ')
         for column in plots:
          x.append((float(column[1])))
          y.append(float(column[0]))

         plt.plot(x,y, label='Loaded from file!')
         plt.xlabel('a angle')
         plt.ylabel('condition number')
         plt.title('Relationship between a angle &Condition number of cov. matrix')
         plt.legend()
         plt.show()
    
    return worst,best


def calculate_best_r(self,worldpoints,imagepoints,b,a):
   #1.57079632679
    best=-1.
    mincond=1000000000.
    covmat=(covariance_matrix_p(self,worldpoints,imagepoints,a,b,100))
    #condmax=LA.cond(covmat)
 #r>0 so I tested many cases after 1 worst
    for r in np.arange(0.0,3.2,0.2):
        covmat=(covariance_matrix_p(self,worldpoints,imagepoints,a,b,r))
        cond=LA.cond(covmat)
        with open('dataR.csv', 'ab') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=' ')
          #filewriter.writerow([float(cond)/condmax , r/100.])
          filewriter.writerow([float(cond) , r])
      
        if cond<=mincond:
            mincond=cond
            
            best=r
    x = []
    y = []
    with open('dataR.csv','r') as csvfile:
         plots = csv.reader(csvfile, delimiter=' ')
         for column in plots:
          x.append((float(column[1])))
          y.append((float(column[0])))

         plt.plot(x,y, label='Loaded from file!')
         plt.xlabel('r distance')
         plt.ylabel('condition number')
         plt.title('Relationship between distance&Condition number of cov. matrix')
         plt.legend()
         plt.show()


        
    return best
#https://en.wikipedia.org/wiki/Reprojection_error
def reprojection_error(imagepoints,DLTimage,points):
    repr_error=0.
    for size in range(points):
     distance=math.sqrt((imagepoints[0,size]-DLTimage[0,size])*(imagepoints[0,size]-DLTimage[0,size])+(imagepoints[1,size]-DLTimage[1,size])*(imagepoints[1,size]-DLTimage[1,size]))
     repr_error=distance+repr_error
    return repr_error


#find R,t error: https://inside.mines.edu/~whoff/courses/EENG512/lectures/18-LinearPoseEstimation.pdf
def mean_error_R(self,estimated_R):
    er=abs((self.R[:3,:3])-(estimated_R[:3,:3]))
    sumerr=sum(er)
    meanerr=(1./9.)*(sumerr[0]+sumerr[1]+sumerr[2])
    return meanerr

def mean_error_t(self,estimated_t):
    er=abs((self.t[:3,3])-(estimated_t[:3,3]))
    #squareerr=(self.t[:3,3])-(estimated_t[:3,3])*(self.t[:3,3])-(estimated_t[:3,3])
    #print er
    sumerr=sum(er)
    meanert=(1./3.)*(sumerr)
    return meanert
    #return meanerr
    
def estimateRwithQRfact(H):
#p. 31 https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws13-14/3DCV_lec05_parameterEstimation.pdf   
   
    Q,B=scipy.linalg.qr((H[:3,:3]))
    #print B
    for i in range(3):
     if B[i,i]<0.:
        B[i,:] = -B[i,:]   
        Q[:,i] = -Q[:,i]
    Rot=Q
    if np.linalg.det(Rot)<0. :  
      Rot= - Rot
    #Final estimated pose
    Rot1=  np.transpose(Rot)
    print Rot1
    est_R=np.array([[Rot1[0,0],Rot1[0,1],Rot1[0,2],0.],
                   [Rot1[1,0],Rot1[1,1],Rot1[1,2],0.],
                   [Rot1[2,0],Rot1[2,1],Rot1[2,2],0.],
                   [0.,0.,0.,1.]])
    return est_R


#values
cam=Camera()
cam.set_K(fx = 800.,fy = 800.,cx = 640.,cy = 480.)
cam.set_width_heigth(960,960)
imagePoints=np.full((2,6),0.0)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
cam.set_t(0.0,-0.0,0.5, frame='world')


mincondH=1000000000.
error=1000000000000.
noise = np.random.normal(0,1,(2,6))
print noise

for p in range(10):
     worldpoints = spherepoints(6,3)
     imagePoints= cam.project(worldpoints, False)
     #H=DLT3D(cam,worldpoints,imagePoints,True)
     #DLTimage=DLTproject(H,worldpoints)
     
     #imagepoints_noise=add_noise(imagePoints,noise,0.,0)
     imagepoints_noise=stand_add_noise(imagePoints,noise)
     H=DLT3D(cam,worldpoints, imagepoints_noise, False)
     DLTimage=DLTproject(H,worldpoints)
     error_withnoise=reprojection_error(imagePoints,DLTimage,6)
     covH=np.cov(H)
     condH=LA.cond(covH)
     if condH<mincondH:
         mincondH=condH
         bestpoints=worldpoints
     if error_withnoise<error:
         error=error_withnoise
         bestnoerror=worldpoints
         bestH=H
     
     print p    
     with open('dataerror.csv', 'ab') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=' ')
          filewriter.writerow([error_withnoise , worldpoints])
 

#x = []
#y = []
#with open('dataerror.csv','r') as csvfile:
 #        plots = csv.reader(csvfile, delimiter=' ')
  #       for column in plots:
   #       x.append((float(column[1])))
    #      y.append(float(column[0]))

     #    plt.plot(x,y, label='Loaded from file!')
      #   plt.xlabel('noise added')
       #  plt.ylabel('error_withnoise')
        # plt.title('reprojectionerror as we increase noise')
         #plt.legend()
         #plt.show()

#NoNoiseTest
#


#NoiseTest

cam_center=camera_center(bestH)
estimated_t=estimate_t(cam,cam_center)
Rotation=estimateRwithQRfact(bestH)
fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(radius*x, radius*y, radius*z, color='g')
ax.scatter(bestnoerror[:3,1],bestnoerror[:3,2],bestnoerror[:3,3],s=70, c='r') 
ax.scatter(bestnoerror[:3,0],bestnoerror[:3,4],bestnoerror[:3,5],s=70, c='r') 
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #ax.plot_wireframe(radius*x, radius*y, radius*z, color='g')
ax.scatter(bestpoints[:3,0],bestpoints[:3,1],bestpoints[:3,2],s=70, c='b') 
ax.scatter(bestpoints[:3,3],bestpoints[:3,4],bestpoints[:3,5],s=70, c='b') 
plt.show()

a,b,r=a_b_r_spherical(cam)
#a=0.
#b=0.
#r=0.
#covmatrix=covariance_matrix_p(cam,np.transpose(worldpoints),imagePoints,a,b,r)
#rbest=calculate_best_r(cam,np.transpose(worldpoints),imagePoints,b,a)
#worst,best=calculate_best_a(cam,np.transpose(worldpoints),imagePoints,b,rbest)

#
 #if error<minerror:
  # minerror=error
   #bestpointconfig=worldpoints
   
#print cam.P
print cam.t

err_R=mean_error_R(cam, Rotation)
err_t=mean_error_t(cam,estimated_t)

#print cam.P
 
