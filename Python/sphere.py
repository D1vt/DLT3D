#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:41:33 2019
/
@author: diana
@set_P, set_K, set_R_axisAngle, set_t developed by raultron
"""
import numpy as np 
from random import randrange
from numpy import linalg as LA
from scipy.linalg import expm, inv
import math

def set_P(K,Rt):
        # P = K[R|t]
        # P is a 3x4 Projection Matrix (from 3d euclidean to image)
        #self.Rt = hstack((self.R, self.t))
        P = np.dot(K,Rt[:3,:4])
        return P
    
def set_K(fx = 1, fy = 1, cx = 0,cy = 0):
        # K is the 3x3 Camera matrix
        # fx, fy are focal lenghts expressed in pixel units
        # cx, cy is a principal point usually at image center
        K = np.mat([[fx, 0, cx],
                      [0,fy,cy],
                      [0,0,1.]], dtype=np.float32)   
        return K



def project(worldpoints, P, imagePoints, quant_error=False):
        """  Project points in x (6*n array) & convert to homogeneous"""
        x = np.dot(P,worldpoints)
        for i in range(x.shape[1]):
          x[:,i] /= x[2,i]
        if(quant_error):
            x = np.around(x, decimals=0)
        return x
   

def set_R_axisAngle(x,y,z, alpha):
        """  Creates a 3D [R|t] matrix for rotation
        around the axis of the vector defined by (x,y,z)
        and an alpha angle."""
        #Normalize the rotation axis a
        a = np.array([x,y,z])
        a = a / np.linalg.norm(a)
        a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(4)
        R[:3,:3] = expm(a_skew*alpha)
        return R
  
def set_t(x,y,z,R, frame = 'camera'):
        #self.t = array([[x],[y],[z]])
        t = np.eye(4)
        if frame=='world':
          cam_world = np.array([x,y,z,1]).T
          cam_t = np.dot(R,-cam_world)
          t[:3,3] = cam_t[:3]
        else:
          t[:3,3] = np.array([x,y,z])
        return t

def update_Rt(R,t):
    Rt=np.dot(R,t)
    return Rt


def spherepoints(points):
    worldpoints = np.random.randn(points, 3)
    worldpoints /= np.linalg.norm(worldpoints, axis=0)
    worldpoints=np.transpose(worldpoints)
    worldpoints=np.vstack((worldpoints,[1.0,1.0,1.0,1.0,1.0,1.0]))
    return worldpoints

def normalize(imagepoints):
    normp= np.linalg.norm(imagepoints)
    imagepoints=imagepoints/normp
    return imagepoints               

def DLT3D(worldpoints, imagepoints, K, normalization=False):
    #if odd row 0,0,0,0,xi,yi,zi,1,-vixi,-viyi,-vizi,-vi
    #if even row : Χι,Υι,Ζι,1,0,0,0,0,-uixi,-uiyi,-uizi,-ui
    if(normalization):
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
               [0.,0.,0.,0.,worldpoints[0,5],worldpoints[1,5],worldpoints[2,5],1.,-imagepoints[1,5]*worldpoints[1,5],-imagepoints[1,5]*worldpoints[1,5],-imagepoints[1,5]*worldpoints[2,5],-imagepoints[1,5]]])
   
     U, s, V = np.linalg.svd(A, full_matrices=True)
     M=V[:,11]
     trans[2]=M[11]/K[2,2]
     trans[1]=(M[7]-K[1,2]*trans[2])/K[1,1]
     trans[0]=(M[3]-K[0,1]*trans[1]-K[0,2]*trans[2])/K[0,0]            
     for i in range(3):
          Rot[2,i]=M[i+8]/K[2,2]  #fr3i=m3i
          Rot[1,i]=(M[i+4]- Rot[2,i]*K[1,2])/K[1,1];  #dr2i+er3i=m2,i
          Rot[0,i]= (M[i]-K[0,2]*Rot[2,i]-K[0,1]*Rot[1,i])/K[0,0]; #ar1i+br2i+cr3i=m1i
     H=np.array([[M[0],M[1],M[2],M[3]],
                [M[4],M[5],M[6],M[7]],
                [M[8],M[9],M[10],M[11]],
                [0.,0.,0.,1.]])
     return trans,Rot,H       
    else:
        pnorm=normalize(imagepoints)
        return DLT3D(worldpoints,pnorm, K, True)
    
    #adding noise to each u,v of the image (random gaussian noise)
def add_noise(imagepoints,sd=4.,mean=0., size=10000):
    px1=np.random.normal(mean,sd,size)+ imagepoints[0,0]
    py1=np.random.normal(mean,sd,size)+ imagepoints[1,0]
    px2=np.random.normal(mean,sd,size)+ imagepoints[0,1]
    py2=np.random.normal(mean,sd,size)+ imagepoints[1,1]
    px3=np.random.normal(mean,sd,size)+ imagepoints[0,2]
    py3=np.random.normal(mean,sd,size)+ imagepoints[1,2]
    px4=np.random.normal(mean,sd,size)+ imagepoints[0,3]
    py4=np.random.normal(mean,sd,size)+ imagepoints[1,3]
    px5=np.random.normal(mean,sd,size)+ imagepoints[0,4]
    py5=np.random.normal(mean,sd,size)+ imagepoints[1,4]
    px6=np.random.normal(mean,sd,size)+ imagepoints[0,5]
    py6=np.random.normal(mean,sd,size)+ imagepoints[1,5]
    return np.array([[px1,px2,px3,px4,px5,px6],
                    [py1,py2,py3,py4,py5,py6]])         
    
    
  ### here we solve the problem using spherical coordinates and Euler angles

#calculate the a,b,c 
def a_b_c_from_Euler(Rt):
    world_position = get_world_position(Rt)  
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
def covariance_matrix_p(K,Rot,Rt,worldpoints,imagepoints):
    #calculate the a,b,c in spherical coord. 
    a,b,c=a_b_c_from_Euler(Rt)
    Jac,JacT=jacobian_matrix(a,b,c,K,worldpoints)
    
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
    return cov_mat,image6points

#here I calculate the R from Euler equations and also the following: dR/da , dR/db , dR/dc as I am going to use them to calculate the Jacobian Matrix 
def R_Euler(a,b,c):
    R_Eu= np.array([[math.cos(a)*math.cos(c) - math.cos(b)*math.sin(a)*math.sin(c), -math.cos(a)*math.sin(c)-math.cos(b)*math.cos(c)*math.sin(a), math.sin(a)*math.sin(b)],
                       [math.cos(c)*math.sin(a)+math.cos(a)*math.cos(b)*math.sin(c), math.cos(a)*math.cos(b)*math.cos(c)-math.sin(a)*math.sin(c), -math.cos(a)*math.sin(b)],
                       [math.sin(b)*math.sin(c), math.cos(c)*math.sin(b), math.cos(b)]])
    dR_Euler_da=np.array([[(-math.sin(a))*math.cos(c) - math.cos(b)*math.cos(a)*math.sin(c), (math.sin(a))*math.sin(c)-math.cos(b)*math.cos(c)*math.cos(a), math.cos(a)*math.sin(b)],
                       [math.cos(c)*math.cos(a)+(-math.sin(a))*math.cos(b)*math.sin(c), (-math.sin(a))*math.cos(b)*math.cos(c)-math.cos(a)*math.sin(c), math.sin(a)*math.sin(b)],
                       [0., 0.,0.]])
    dR_Euler_db=np.array([[math.sin(b)*math.sin(a)*math.sin(c), math.sin(b)*math.cos(c)*math.sin(a), math.sin(a)*math.cos(b)],
                       [math.cos(a)*(-math.sin(b))*math.sin(c), math.cos(a)*(-math.sin(b))*math.cos(c), -math.cos(a)*math.cos(b)],
                       [math.cos(b)*math.sin(c), math.cos(c)*math.cos(b), (-math.sin(b))]])
    dR_Euler_dc=np.array([[math.cos(a)*(-math.sin(c)) - math.cos(b)*math.sin(a)*math.cos(c), -math.cos(a)*math.cos(c)-math.cos(b)*(-math.sin(c))*math.sin(a),0.],
                       [(-math.sin(c))*math.sin(a)+math.cos(a)*math.cos(b)*math.cos(c), math.cos(a)*math.cos(b)*(-math.sin(c))-math.sin(a)*math.cos(c), 0.],
                       [math.sin(b)*math.cos(c), (-math.sin(c))*math.sin(b), 0.]])
    return R_Eu,dR_Euler_da,dR_Euler_db,dR_Euler_dc

def get_world_position(Rt):
        w_pos = np.dot(inv(Rt), np.array([0,0,0,1]))
        return w_pos
 
#I am using this function to find the u&v of each point (p. 81-82 in "Tracking Errors in AI")      
def jacobian_uv(X,P_da,P_db,P_dc):
     Y=np.array([[X[0]],
                [X[1]],
                [X[2]],
                [1]]) 
     image_point_da = np.dot(P_da,Y)
     image_point_db = np.dot(P_db,Y)
     image_point_dc = np.dot(P_dc,Y)
     u_da=  image_point_da[0]/ image_point_da[2]
     v_da=  image_point_da[1]/image_point_da[2]
     u_db=image_point_db[0]/image_point_db[2]
     v_db=image_point_db[1]/image_point_db[2]
     u_dc=image_point_dc[0]/image_point_dc[2]
     v_dc=image_point_dc[1]/image_point_dc[2]
     return float(u_da),float(u_db),float(u_dc),float(v_da),float(v_db),float(v_dc) #2*3
 
        
#here I calculate Jacobian matrix and transpose of the Jacobian matrix. As I have to calculate the Jacobian Matrix I need to calculate the following: dR/da, dR/db,dR/dc , dt/da , dt/db, dt/dc 
def jacobian_matrix(a,b,c,K,worldpoints):
    
     R_Sphere = np.array([[math.cos(b)*math.cos(a), -math.sin(b), math.cos(b)*math.sin(a)],
                       [math.sin(b)*math.cos(a), math.cos(b), math.sin(b)*math.sin(a)],
                       [math.sin(a), 0, math.cos(a)]])
     
     dR_da=np.array([ [math.cos(b)*(-math.sin(a)),0, math.cos(b)*math.cos(a)], #,c*(-math.cos(a))*math.cos(b),
                          [math.sin(b)*(-math.sin(a)), 0, math.sin(b)*math.cos(a)], #c*math.sin(b)*math.cos(a),
                          [(-math.sin(a)), 0,(-math.sin(a))] ]) #,c*(-math.sin(a))]])
     dR_db=np.array([[(-math.sin(b))*math.cos(a), -math.cos(b), (-math.sin(b))*math.sin(a)],
                          [math.cos(b)*math.cos(a), (-math.sin(b)),(-math.sin(b))*math.sin(a)],
                          [0., 0., 0.]])
     dR_dc=np.full((3,3),0.0) 
     R_Eu,dR_Euler_da,dR_Euler_db,dR_Euler_dc=R_Euler(0.0,np.deg2rad(180.0),0.0)
     R_Sphere_da=np.dot(dR_da,R_Eu)+np.dot(R_Sphere,dR_Euler_da)
     R_Sphere_db=np.dot(dR_db,R_Eu)+np.dot(R_Sphere,dR_Euler_db)
     R_Sphere_dc=np.dot(dR_dc,R_Eu)+np.dot(R_Sphere,dR_Euler_dc)
     R_Sphere = np.dot(R_Sphere, R_Eu[:3, :3])
     t_Sphere= np.array([[c*math.sin(a)*math.cos(b)],
            [c*math.sin(b)*math.sin(a)],
             [c*math.cos(a)]])
     t_da=np.array([[c*math.cos(a)*math.cos(b)],
            [c*math.sin(b)*math.cos(a)],
             [c*(-math.sin(a))]])
     t_db=np.array([[c*math.sin(a)*(-math.sin(b))],
            [c*math.cos(b)*math.sin(a)],
             [0.]])
     t_dc=np.array([[math.sin(a)*math.cos(b)],
            [math.sin(b)*math.sin(a)],
             [math.cos(a)]])
     t_Sphere_da=np.dot(R_Sphere_da, -t_Sphere)+np.dot(R_Sphere, -t_da)
     t_Sphere_db=np.dot(R_Sphere_db, -t_Sphere)+np.dot(R_Sphere, -t_db)
     t_Sphere_dc=np.dot(R_Sphere_dc, -t_Sphere)+np.dot(R_Sphere, -t_dc)
     t_Sphere=np.dot(R_Sphere,-t_Sphere)
     #then I am using them to find the following: dRt/da , dRt/db, dRt/dc where a,b,c were calculated in a previous function
     dRt_da=np.hstack((R_Sphere_da, t_Sphere_da))
     dRt_db=np.hstack((R_Sphere_db, t_Sphere_db))
     dRt_dc=np.hstack((R_Sphere_dc, t_Sphere_dc))
     
     #I am creating the non linear camera equations to find each point (u,v) page 81 in : "Tracking Errors in AR"
     P_da=np.dot(K,dRt_da)
     P_db=np.dot(K,dRt_db)
     P_dc=np.dot(K,dRt_dc)
      
     #then for each one of the world points I find the u,v and use them to calculate the Jacobian Matrix. So I need to calculate the followin for each point: du/da, dv/da, du/db,dv/db,du/dc,dv/dc
     u1a,u1b,u1c,v1a,v1b,v1c=jacobian_uv(worldpoints[0,:],P_da,P_db,P_dc)
     u2a,u2b,u2c,v2a,v2b,v2c=jacobian_uv(worldpoints[1,:],P_da,P_db,P_dc)
     u4a,u4b,u4c,v4a,v4b,v4c=jacobian_uv(worldpoints[3,:],P_da,P_db,P_dc)
     u5a,u5b,u5c,v5a,v5b,v5c=jacobian_uv(worldpoints[4,:],P_da,P_db,P_dc)
     u6a,u6b,u6c,v6a,v6b,v6c=jacobian_uv(worldpoints[5,:],P_da,P_db,P_dc)
     u3a,u3b,u3c,v3a,v3b,v3c=jacobian_uv(worldpoints[2,:],P_da,P_db,P_dc)
     
     #3*12 Jacobian.Transpose
    # JacT=np.array([[u1a,u2a,u3a,u4a,u5a,u6a,v1a,v2a,v3a,v4a,v5a,v6a],
                #   [u1b,u2b,u3b,u4b,u5b,u6b,v1b,v2b,v3b,v4b,v5b,v6b],
                 #  [u1c,u2c,u3c,u4c,u5c,u6c,v1c,v2c,v3c,v4c,v5c,v6c]])
     
    
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
        
    
   #na brw to p για να εκτιμησω το σφαλμα γιατι απο π παιρνω το τζακομπιαν
        
#values
P = np.eye(3,4)
R = np.eye(4, dtype=np.float32) # rotation
t = np.eye(4, dtype=np.float32) # translation
Rt = np.eye(4, dtype=np.float32)
imagePoints=np.full((3,6),0.0)
K=set_K(fx = 800, fy = 800, cx = 640, cy = 480)
P=set_P(K,Rt)
R=set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
Rt=update_Rt(R,t)
P=set_P(K,Rt)
t=set_t(0.0,-0.0,0.5,R, frame='world')
Rt=update_Rt(R,t)
P=set_P(K,Rt)
worldpoints = spherepoints(6)
#testpoints
#worldpoints = np.array([[-0.206674, 0.240009,-0.29203],
#                        [0.091502,-0.144823,	0.0323819],
 #                       [0.371293,	-0.458251	,0.269246],
  #                      [-0.864962	,0.222202	,0.0354375],
   #                     [-0.211795	,0.418585	,0.491078],
    #                    [-0.134308	,0.69774,	-0.773798]])


imagePoints = project(worldpoints,P,imagePoints, False)
imagepoints_noise=add_noise(imagePoints,2,0)




trans=np.full((3,1), 0.0)
Rot=np.full((3,3), 0.0)

#NoNoiseTest
trans,Rot,H=DLT3D(worldpoints, imagePoints, K, True)
cond2=LA.cond(H)

#NoiseTest
#trans,Rot,H=DLT3D(worldpoints, imagepoints_noise, K)
#condnoise=LA.cond(H)   
#covnoise=np.cov(H)
 


covmatrix,imagecov=covariance_matrix_p(K,Rot,Rt,np.transpose(worldpoints),imagePoints)
    

