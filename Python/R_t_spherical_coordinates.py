import math
from vision.camera import Camera
import numpy as np
import scipy.linalg
from python.sphere import Sphere
from numpy import linalg as LA
from scipy.linalg import expm, inv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import csv
with open('dataA.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('dataR.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('dataerror.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')


def a_b_r_spherical(self):
    """
    A function to calculate a,b,c from spherical coordinates
    """
    world_position = self.get_world_position()
    x = world_position[0]
    y = world_position[1]
    z = world_position[2]
    r = np.sqrt(x * x + y * y + z * z, dtype=np.float32)
    if r == 0:
        a = np.deg2rad(0.0)
    else:
        a = math.acos(z / r)
    if x == 0:
        b = np.deg2rad(90.0)
    else:
        b = math.atan(y / x)
return a, b, r


def covariance_matrix_p(self, worldpoints, imagepoints, a, b, c):
    """
    Find the covariance matrix from R,t expressed in spherical coordinates
    The equation for the final cov matrix (given in Thesis Tracking Errors in
    AR (eq. 5.8) = inv(np.dot(np.dot(JacT,image6points), Jac)),
    where JacT is the transpose matrix of Jacobian matrix
    (in spherical coordinates)
    """
    Jac, JacT = jacobian_matrix(self, a, b, c, worldpoints)
    # adding noise to imagePoints
    points = add_noise(imagepoints)
    # Calculate cov. matrix for each point in order to create the Σvi(eq. 5.8)
    cov_mat_p1 = np.cov(points[0, 0], points[1, 0])
    cov_mat_p2 = np.cov(points[0, 1], points[1, 1])
    cov_mat_p3 = np.cov(points[0, 2], points[1, 2])
    cov_mat_p4 = np.cov(points[0, 3], points[1, 3])
    cov_mat_p5 = np.cov(points[0, 4], points[1, 4])
    cov_mat_p6 = np.cov(points[0, 5], points[1, 5])
    # create the 12*12 cov matrix
    image6points = np.block([[cov_mat_p1, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                             [np.zeros((2, 2)), cov_mat_p2, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                             [np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p3, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                             [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p4, np.zeros((2, 2)), np.zeros((2, 2))],
                             [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p5, np.zeros((2, 2))],
                             [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p6]])
    image6points = np.diag(np.diag(image6points))
    # the eq. 5.8 uses the inv. cov. matrix of those points
    image6points = inv(image6points)
    # find the final 3*3 matrix
    cov_mat = inv(np.dot(np.dot(JacT, image6points), Jac))
    return cov_mat

def jacobian_uv(X, P, P_da, P_db, P_dc):
    """
    Find the u&v of each point as well as their derivatives p.81-82 in
    "Tracking Errors in in Augmented Reality in Laparoscopic Surgeries"
    """
    Y = np.array([[X[0]],
                  [X[1]],
                  [X[2]],
                  [1.]])
    # image_points=np.dot(P,Y)
    image_point_da = np.dot(P_da, Y)  # find da
    image_point_db = np.dot(P_db, Y)  # find db
    image_point_dc = np.dot(P_dc, Y)  # find dc
    u_da = image_point_da[0]  # find du/da
    v_da = image_point_da[1]  # find dv/da
    u_db = image_point_db[0]  # find du/db
    v_db = image_point_db[1]  # find dv/db
    u_dc = image_point_dc[0]  # find du/dc
    v_dc = image_point_dc[1]  # find dv/dc
return float(u_da), float(u_db), float(u_dc), float(v_da), float(v_db), float(v_dc)


def add_noise(imagepoints, sd=4., mean=0., size=10000):
    """
    Adding noise to each u,v of the image (random gaussian noise)
    """
    if size == 0:
        imagenoise = np.full((3, 6), 1.0)
        imagenoise[0, 0] = imagepoints[0, 0] + np.random.normal(mean, sd)
        imagenoise[1, 0] = imagepoints[1, 0] + np.random.normal(mean, sd)
        imagenoise[0, 1] = imagepoints[0, 1] + np.random.normal(mean, sd)
        imagenoise[1, 1] = imagepoints[1, 1] + np.random.normal(mean, sd)
        imagenoise[0, 2] = imagepoints[0, 2] + np.random.normal(mean, sd)
        imagenoise[1, 2] = imagepoints[1, 2] + np.random.normal(mean, sd)
        imagenoise[0, 3] = imagepoints[0, 3] + np.random.normal(mean, sd)
        imagenoise[1, 3] = imagepoints[1, 3] + np.random.normal(mean, sd)
        imagenoise[0, 4] = imagepoints[0, 4] + np.random.normal(mean, sd)
        imagenoise[1, 4] = imagepoints[1, 4] + np.random.normal(mean, sd)
        imagenoise[0, 5] = imagepoints[0, 5] + np.random.normal(mean, sd)
        imagenoise[1, 5] = imagepoints[1, 5] + np.random.normal(mean, sd)
        return imagenoise
    else:
        px1 = imagepoints[0, 0] + np.random.normal(mean, sd, size)
        py1 = imagepoints[1, 0] + np.random.normal(mean, sd, size)
        px2 = imagepoints[0, 1] + np.random.normal(mean, sd, size)
        py2 = imagepoints[1, 1] + np.random.normal(mean, sd, size)
        px3 = imagepoints[0, 2] + np.random.normal(mean, sd, size)
        py3 = imagepoints[1, 2] + np.random.normal(mean, sd, size)
        px4 = imagepoints[0, 3] + np.random.normal(mean, sd, size)
        py4 = imagepoints[1, 3] + np.random.normal(mean, sd, size)
        px5 = imagepoints[0, 4] + np.random.normal(mean, sd, size)
        py5 = imagepoints[1, 4] + np.random.normal(mean, sd, size)
        px6 = imagepoints[0, 5] + np.random.normal(mean, sd, size)
        py6 = imagepoints[1, 5] + np.random.normal(mean, sd, size)
        imageNoise = np.array([[px1, px2, px3, px4, px5, px6],
                               [py1, py2, py3, py4, py5, py6]])
        return imageNoise     

def jacobian_matrix(self, a, b, r, worldpoints):
    """
    Calculate Jacobian matrix and transpose of the Jacobian matrix.
    But first there is a need
    to calculate the following: dR/da, dR/db,dR/dr , dt/da , dt/db, dt/dr
    """
    R_Sphere = np.array([[math.cos(b)*math.cos(a), -math.sin(b), math.cos(b)*math.sin(a)],
                         [math.sin(b)*math.cos(a), math.cos(b), math.sin(b)*math.sin(a)],
                         [math.sin(a), 0., math.cos(a)]])
    # dR/da
    dR_da = np.array([[math.cos(b)*(-math.sin(a)), 0., math.cos(b)*math.cos(a)],
                      [math.sin(b)*(-math.sin(a)), 0., math.sin(b)*math.cos(a)],
                      [(math.cos(a)), 0., (-math.sin(a))]])
    # dR/db
    dR_db = np.array([[(-math.sin(b))*math.cos(a), -math.cos(b), (-math.sin(b))*math.sin(a)],
                      [math.cos(b)*math.cos(a), (-math.sin(b)), (math.cos(b))*math.sin(a)],
                      [0., 0., 0.]])
    dR_dr = np.full((3, 3), 0.)
    R_Eu = R_from_euler()
    # find the final R, t
    R_Sphere_da = np.dot(dR_da, R_Eu[:3, :3])
    R_Sphere_db = np.dot(dR_db, R_Eu[:3, :3])
    R_Sphere_dr = np.dot(dR_dr, R_Eu[:3, :3])
    R_Sphere = np.dot(R_Sphere, R_Eu[:3, :3])
    t_Sphere = np.array([[r*math.sin(a)*math.cos(b)],
                         [r*math.sin(b)*math.sin(a)],
                         [r*math.cos(a)]])
    t_da = np.array([[r*math.cos(a)*math.cos(b)],
                     [r*math.sin(b)*math.cos(a)],
                     [r*(-math.sin(a))]])
    t_db = np.array([[r*math.sin(a)*(-math.sin(b))],
                     [r*math.cos(b)*math.sin(a)],
                     [0.]])
    t_dr = np.array([[math.sin(a)*math.cos(b)],
                     [math.sin(b)*math.sin(a)],
                     [math.cos(a)]])
    t_Sphere_da = np.dot(R_Sphere_da, -t_Sphere) + np.dot(R_Sphere, -t_da)
    t_Sphere_db = np.dot(R_Sphere_db, -t_Sphere) + np.dot(R_Sphere, -t_db)
    t_Sphere_dr = np.dot(R_Sphere_dr, -t_Sphere) + np.dot(R_Sphere, -t_dr)
    t_Sphere = np.dot(R_Sphere, -t_Sphere)
    # then I am using them to find the following: dRt/da , dRt/db, dRt/dc where a,b,c were calculated in a previous function
    dRt_da = np.hstack((R_Sphere_da, t_Sphere_da))
    dRt_db = np.hstack((R_Sphere_db, t_Sphere_db))
    dRt_dr = np.hstack((R_Sphere_dr, t_Sphere_dr))
    Rt_sphere = np.hstack((R_Sphere, t_Sphere))
    # I am creating the non linear camera equations to find each point (u,v) page 81 in : "Tracking Errors in in Laparoscopic Surgeries"
    P_da = np.dot(self.K, dRt_da)
    P_db = np.dot(self.K, dRt_db)
    P_dr = np.dot(self.K, dRt_dr)
    P = np.dot(self.K, Rt_sphere)
    """
    then for each one of the world points I find the u,v and use them
    to calculate the Jacobian Matrix.
    So I need to calculate the followin for
    each point: du/da, dv/da, du/db,dv/db,du/dc,dv/dc
    """
    u1a, u1b, u1c, v1a, v1b, v1c = jacobian_uv(worldpoints[0, :], P, P_da, P_db, P_dr)
    u2a, u2b, u2c, v2a, v2b, v2c = jacobian_uv(worldpoints[1, :], P, P_da, P_db, P_dr)
    u3a, u3b, u3c, v3a, v3b, v3c = jacobian_uv(worldpoints[2, :], P, P_da, P_db, P_dr)
    u4a, u4b, u4c, v4a, v4b, v4c = jacobian_uv(worldpoints[3, :], P, P_da, P_db, P_dr)
    u5a, u5b, u5c, v5a, v5b, v5c = jacobian_uv(worldpoints[4, :], P, P_da, P_db, P_dr)
    u6a, u6b, u6c, v6a, v6b, v6c = jacobian_uv(worldpoints[5, :], P, P_da, P_db, P_dr)
    # 12*3 Jacobian Matrix
    Jac = np.array([[u1a, u1b, u1c],
                    [u2a, u2b, u2c],
                    [u3a, u3b, u3c],
                    [u4a, u4b, u4c],
                    [u5a, u5b, u5c],
                    [u6a, u6b, u6c],
                    [v1a, v1b, v1c],
                    [v2a, v2b, v2c],
                    [v3a, v3b, v3c],
                    [v4a, v4b, v4c],
                    [v5a, v5b, v5c],
                    [v6a, v6b, v6c]])
    JacT = np.transpose(Jac)
    return Jac, JacT


def R_from_euler():
    """
    Find the R using Euler Angles, as defined in p. 51 of thesis Tracking
    Errors in Augmented Reality in Laparoscopic Surgeries
    """
    a = math.atan(cam.R[2, 1]/cam.R[2, 2])
    b = -math.asin(cam.R[2, 0])
    c = math.atan(cam.R[1, 0]/cam.R[0, 0])
    # print a,b,c
    R_Eu_a = np.array([[1., 0., 0.],
                       [0., math.cos(a), -math.sin(a)],
                       [0., math.sin(a), math.cos(a)]])
    R_Eu_b = np.array([[math.cos(b), 0., math.sin(b)],
                       [0., 1., 0.],
                       [-math.sin(b), 0., math.cos(b)]])
    R_Eu_c = np.array([[math.cos(c), -math.sin(c), 0.],
                       [math.sin(c), math.cos(c), 0.],
                       [0., 0., 1.]])
    R_Eu1 = np.dot(R_Eu_c, R_Eu_b)  # Rz*Ry*Rx
    R_Eu = np.dot(R_Eu1, R_Eu_a)
    return R_Eu
    
    
    def calculate_best_a(self,worldpoints,imagepoints,b,r):
    """Find a(angle) that leads to the minimum condition number==the well conditioned matrix. 
    With a limited from -pi/2 to pi/2. 
    The minimum condition number of the cov matrix will give the best a angle, when 
    r, b are constants (origin). 
    The maximum condition number of the cov matrix will give the worst a angle, when 
    r, b are constants (origin). """
    best=2*math.pi
    worst=best
    mincond=1000000000.
    maxcond=-1

    for a in np.arange(-90., 95.,5.):
 
        covmat=(covariance_matrix_p(self,worldpoints,imagepoints,np.rad2deg(a),b,r))
        cond=LA.cond(covmat)
        with open('dataA.csv', 'ab') as csvfile: #crete a csv to save and the plot the measurments for a
          filewriter = csv.writer(csvfile, delimiter=' ')
          filewriter.writerow([cond , a])
          
        if cond<=mincond:
            mincond=cond
            best=a  #best angle
        if cond>=maxcond:
            maxcond=cond  #worst angle
            worst=a
    x = []
    y = []
    with open('dataA.csv','r') as csvfile:
         plots = csv.reader(csvfile, delimiter=' ')
         for column in plots:
          x.append((float(column[1])))
          y.append(float(column[0]))

         plt.plot(x,y, label='Loaded from file!')
         plt.xlabel('a angle(*degrees)')
         plt.ylabel('condition number')
         plt.title('Relationship between a angle & Condition number of cov. matrix')
         plt.legend()
         plt.show()
    
    return worst,best
   
   
def a_angle_deriv(self, worldpoints, imagepoints, b, r):
        """
        Calculate best a angle of camera using the derivative of covariance
        matrices and then plot my results
        """
        for a in np.arange(-90., 95., 5.):
            covmat = (covariance_matrix_p(self, worldpoints, imagepoints, np.rad2deg(a), b, r))
            deriv = (np.gradient(covmat))
            # deriv=np.gradient(deriv_covariance)
            print deriv, np.linalg.norm(deriv)
            with open('dataA.csv', 'ab') as csvfile:  # create a csv to save and the plot the measurments
                filewriter = csv.writer(csvfile, delimiter=' ')
                # filewriter.writerow([float(cond)/condmax , r/100.])
                filewriter.writerow([float(np.linalg.norm(deriv)), a])
        x = []
        y = []
        with open('dataA.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=' ')
            for column in plots:
                x.append((float(column[1])))
                y.append((float(column[0])))
        plt.plot(x, y, label='Loaded from file!')
        plt.xlabel('a angle(degrees)')
        plt.ylabel('norm of deriv of covariance')
        plt.title('Relationship between a angle & covariance matrix')
        plt.legend()
        plt.show()  


def calculate_best_r(self, worldpoints, imagepoints, b, a):
    """
    Find r(radius) that lead to the
    minimum condition number==the
    well conditioned matrix. With r>0
    """
    best = -1.
    mincond = 1000000000.
    covmat = (covariance_matrix_p(self, worldpoints, imagepoints, a, b, 100))
    # r>0 so I tested many cases after 1 worst
    for r in np.arange(0.0, 3.2, 0.2):
        covmat = (covariance_matrix_p(self, worldpoints, imagepoints, a, b, r))
        cond = LA.cond(covmat)
        with open('dataR.csv', 'ab') as csvfile:  # create a csv to save and the plot the measurments
            filewriter = csv.writer(csvfile, delimiter=' ')
            # filewriter.writerow([float(cond)/condmax , r/100.])
            filewriter.writerow([float(cond), r])
        if cond <= mincond:
            mincond = cond
            best = r
    x = []
    y = []
    with open('dataR.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for column in plots:
            x.append((float(column[1])))
            y.append((float(column[0])))
        plt.plot(x, y, label='Loaded from file!')
        plt.xlabel('r distance(*m)')
        plt.ylabel('condition number')
        plt.title('Relationship between distance&Cond. number of cov.matrix')
        plt.legend()
        plt.show()
    return best

# ------------------------------------test cases------------------------------
#sph = Sphere()

#cam = Camera()
#cam.set_K(fx=800., fy=800., cx=640., cy=480.)
#cam.set_width_heigth(1280, 960)
#imagePoints = np.full((2, 6), 0.0)
#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
#cam.set_t(0.0, -0.0, 0.5, frame='world')
# Test case using a,b,r spherical coordinates---------------------------------
#worldpoints = sph.random_points(6,0.3)
#a, b, r = a_b_r_spherical(cam)
#covmatrix = covariance_matrix_p(cam, np.transpose(worldpoints), imagePoints, a, b, r)
#rbest=calculate_best_r(cam,np.transpose(worldpoints),imagePoints,b,a)
#a_angle_deriv(cam, np.transpose(worldpoints), imagePoints, b, r)
# calculate_best_a(cam,np.transpose(worldpoints),imagePoints,b,r)
