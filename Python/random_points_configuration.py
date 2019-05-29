import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from vision.camera import Camera
from python.sphere import Sphere
from numpy import linalg as LA
import random
import python.dlt_and_errors as dlt


def stand_add_noise(imagepoints, points=6):
    imagenoise = np.full((3, points), 1.0)
    for i in range(2000):
        imagenoise = imagenoise + add_noise(imagepoints, sd=8., mean=2.,
                                           )
    imagenoise = imagenoise/2000.
    return imagenoise

def add_noise(imagepoints, sd=4., mean=0.):
    """
    Adding noise to each u,v of the image (random gaussian noise)
    """
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

def points_config_randomtest(notest=1):
    """
    Select the six points configuration that leads to the minimum condition number after
    a specific number of tests (notests)
    """
    
    # mincondH = 1000000000.
    error = 100000000000000.
    # add same noise to each random configuration of worldpoints
    noise = np.random.normal(0, 1, (2, 6))
    # choose how many random test we will do
    for p in range(notest):
        worldpoints = sph.random_points(6, 0.3)
        if p==0:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.scatter(worldpoints[:3, 0], worldpoints[:3, 1],
                       worldpoints[:3, 2], s=50, c='r')
            ax.scatter(worldpoints[:3, 3], worldpoints[:3, 4],
                       worldpoints[:3, 5], s=50, c='r')
            ax.set_xlabel('x label')
            ax.set_ylabel('y label')
            ax.set_zlabel('z label')
            plt.show()
            sph.plot_sphere_and_points(worldpoints)
        imagePoints = cam.project(worldpoints, False)
        # H = DLT3D(cam, worldpoints, imagePoints, True)
        # DLTimage=DLTproject(H,worldpoints)
        imagepoints_noise = stand_add_noise(imagePoints)
        H = dlt.DLT3D(cam, worldpoints, imagepoints_noise, False)
        DLTimage = dlt.DLTproject(H, worldpoints)
        condit=LA.cond(H)
        #error_withnoise = dlt.reprojection_error(imagePoints, DLTimage, 6)
        if condit<error:
        #if error_withnoise < error:
            # error = error_withnoise
            error = condit
            bestnoerror = worldpoints
            
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(bestnoerror[:3, 0], bestnoerror[:3, 1],
                       bestnoerror[:3, 2], s=50, c='g')
    ax.scatter(bestnoerror[:3, 3], bestnoerror[:3, 4],
                       bestnoerror[:3, 5], s=50, c='g')
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')
    plt.show()
    sph.plot_sphere_and_points(bestnoerror)
        # print p     
    return bestnoerror
    
    
# ------------------------------------test cases------------------------------


cam = Camera()


sph = Sphere()


cam.set_K(fx=800., fy=800., cx=640., cy=480.)
cam.set_width_heigth(1280, 960)
imagePoints = np.full((2, 6), 0.0)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
cam.set_t(0.0, -0.0, 0.5, frame='world')
