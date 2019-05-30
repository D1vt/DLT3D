from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np


def get_semi_axes_abc(cl):
    """
    Get a,b,c of the ellipsoid, they are half the length of the principal axes.
    """
    # Confidence level, we consider only n = 3
    if cl == 0.25:
        z_square = 1.21253
    elif cl == 0.5:
        z_square = 2.36597
    elif cl == 0.75:
        z_square = 4.10834
    elif cl == 0.95:
        z_square = 7.81473
    elif cl == 0.97:
        z_square = 8.94729
    elif cl == 0.99:
        z_square = 11.3449
    else:
        z_square = 0
        print "Error: Wrong confidence level!!!"
    return z_square


def plotEllipsoid(eigenvalues, center = np.transpose([0,0,0]), ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
        """Plot an ellipsoid"""
        z_square = get_semi_axes_abc(0.75)
        l1 = z_square*math.sqrt(eigenvalues[0]) 
        l2 = z_square*math.sqrt(eigenvalues[1]) 
        l3 = z_square*math.sqrt(eigenvalues[2]) 
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        
        # cartesian coordinates that correspond to the spherical angles:
        x = l1 * np.outer(np.cos(u), np.sin(v))
        y =l2 * np.outer(np.sin(u), np.sin(v))
        z = l3 * np.outer(np.ones_like(u), np.cos(v))
    
        if plotAxes:
            axes = np.array([[l1,0.0,0.0],
                             [0.0,l2,0.0],
                             [0.0,0.0,l3]])
           
            # plot axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color=cageColor)
        # plot ellipsoid
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
        if make_ax:
            plt.show()
            plt.close(fig)
            del fig

            
# Test case using a,b,r spherical coordinates---------------------------------
a, b, r = a_b_r_spherical(cam)
worldpoints = sph.random_points(6, 0.3)
#test for the first pose in front of the marker
covmatrix = covariance_matrix_p(cam, np.transpose(worldpoints),
                                imagePoints, a, b, r)
eigenvalues=LA.eigvals(covmatrix)
plotEllipsoid(eigenvalues)

#test for alpha=pi/2
a = math.pi/2
covmatrix = covariance_matrix_p(cam, np.transpose(worldpoints),
                                imagePoints, a, b, r)   
eigenvalues=LA.eigvals(covmatrix)
plotEllipsoid(eigenvalues)  
#test for a=0
a = 0.
covmatrix = covariance_matrix_p(cam, np.transpose(worldpoints),
                                imagePoints, a, b, r)   
eigenvalues=LA.eigvals(covmatrix)
plotEllipsoid(eigenvalues)   
#test for a=-pi/2
a = -math.pi/2
covmatrix = covariance_matrix_p(cam, np.transpose(worldpoints),
                                imagePoints, a, b, r)   
eigenvalues=LA.eigvals(covmatrix)
plotEllipsoid(eigenvalues)             
