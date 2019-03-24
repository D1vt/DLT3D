"""
Created on Thu Mar 21 16:20:27 2019

@author: diana, raulacuna
"""
import autograd.numpy as np
from autograd import grad

class Gradient(object):
    def __init__(self):
        self.dx1 = None
        self.dy1 = None
        self.dz1 = None
        self.dx2 = None
        self.dy2 = None
        self.dz2 = None
        self.dx3 = None
        self.dy3 = None
        self.dz3 = None
        self.dx4 = None
        self.dy4 = None
        self.dz4 = None
        self.dx5 = None
        self.dy5 = None
        self.dz5 = None
        self.dx6 = None
        self.dy6 = None
        self.dz6 = None

        self.dx1_eval = 0
        self.dy1_eval = 0
        self.dz1_eval = 0

        self.dx2_eval = 0
        self.dy2_eval = 0
        self.dz2_eval = 0

        self.dx3_eval = 0
        self.dy3_eval = 0
        self.dz3_eval = 0

        self.dx4_eval = 0
        self.dy4_eval = 0
        self.dz4_eval = 0

        self.dx5_eval = 0
        self.dy5_eval = 0
        self.dz5_eval = 0

        self.dx6_eval = 0
        self.dy6_eval = 0
        self.dz6_eval = 0

        self.dx1_eval_old = 0
        self.dy1_eval_old = 0
        self.dz1_eval_old = 0

        self.dx2_eval_old = 0
        self.dy2_eval_old = 0
        self.dz2_eval_old = 0

        self.dx3_eval_old = 0
        self.dy3_eval_old = 0
        self.dz3_eval_old = 0

        self.dx4_eval_old = 0
        self.dy4_eval_old = 0
        self.dz4_eval_old = 0

        self.dx5_eval_old = 0
        self.dy5_eval_old = 0
        self.dz5_eval_old = 0

        self.dx6_eval_old = 0
        self.dy6_eval_old = 0
        self.dz6_eval_old = 0

        self.n = None  # step in gradient descent

        self.n_x1 = self.n
        self.n_x2 = self.n
        self.n_x3 = self.n
        self.n_x4 = self.n
        self.n_x5 = self.n
        self.n_x6 = self.n

        self.n_y1 = self.n
        self.n_y2 = self.n
        self.n_y3 = self.n
        self.n_y4 = self.n
        self.n_y5 = self.n
        self.n_y6 = self.n

        self.n_z1 = self.n
        self.n_z2 = self.n
        self.n_z3 = self.n
        self.n_z4 = self.n
        self.n_z5 = self.n
        self.n_z6 = self.n

    def set_n(self, n):
        self.n = 0.0001  # step in gradient descent
        self.n_pos = 0.02*n  # for SuperSAB
        self.n_neg = 0.03*n  # for SuperSAB
        self.n_x1 = n
        self.n_x2 = n
        self.n_x3 = n
        self.n_x4 = n
        self.n_x5 = n
        self.n_x6 = n

        self.n_y1 = n
        self.n_y2 = n
        self.n_y3 = n
        self.n_y4 = n
        self.n_y5 = n
        self.n_y6 = n

        self.n_z1 = n
        self.n_z2 = n
        self.n_z3 = n
        self.n_z4 = n
        self.n_z5 = n
        self.n_z6 = n


def calculate_A_matrix_autograd(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4,
                                x5, y5, z5, x6, y6, z6, P):
    """ Calculate the A matrix for the DLT algorithm:  A.H = 0
    all coordinates are in object sphere
    """
    X1 = np.array([[x1], [y1], [z1], [1.]]).reshape(4, 1)
    X2 = np.array([[x2], [y2], [z2], [1.]]).reshape(4, 1)
    X3 = np.array([[x3], [y3], [z3], [1.]]).reshape(4, 1)
    X4 = np.array([[x4], [y4], [z4], [1.]]).reshape(4, 1)
    X5 = np.array([[x5], [y5], [z5], [1.]]).reshape(4, 1)
    X6 = np.array([[x6], [y6], [z6], [1.]]).reshape(4, 1)

    U1 = np.array(np.dot(P, X1)).reshape(3, 1)
    U2 = np.array(np.dot(P, X2)).reshape(3, 1)
    U3 = np.array(np.dot(P, X3)).reshape(3, 1)
    U4 = np.array(np.dot(P, X4)).reshape(3, 1)
    U5 = np.array(np.dot(P, X5)).reshape(3, 1)
    U6 = np.array(np.dot(P, X6)).reshape(3, 1)

    world_pts = np.hstack([X1, X2, X3, X4, X5, X6])
    image_pts = np.hstack([U1, U2, U3, U4, U5, U6])

    x1 = world_pts[0, 0]/world_pts[3, 0]
    y1 = world_pts[1, 0]/world_pts[3, 0]
    z1 = world_pts[2, 0]/world_pts[3, 0]

    x2 = world_pts[0, 1]/world_pts[3, 1]
    y2 = world_pts[1, 1]/world_pts[3, 1]
    z2 = world_pts[2, 1]/world_pts[3, 1]

    x3 = world_pts[0, 2]/world_pts[3, 2]
    y3 = world_pts[1, 2]/world_pts[3, 2]
    z3 = world_pts[2, 2]/world_pts[3, 2]

    x4 = world_pts[0, 3]/world_pts[3, 3]
    y4 = world_pts[1, 3]/world_pts[3, 3]
    z4 = world_pts[2, 3]/world_pts[3, 3]

    x5 = world_pts[0, 4]/world_pts[3, 4]
    y5 = world_pts[1, 4]/world_pts[3, 4]
    z5 = world_pts[2, 4]/world_pts[3, 4]

    x6 = world_pts[0, 5]/world_pts[3, 5]
    y6 = world_pts[1, 5]/world_pts[3, 5]
    z6 = world_pts[2, 5]/world_pts[3, 5]

    u1 = image_pts[0, 0]/image_pts[2, 0]
    v1 = image_pts[1, 0]/image_pts[2, 0]

    u2 = image_pts[0, 1]/image_pts[2, 1]
    v2 = image_pts[1, 1]/image_pts[2, 1]

    u3 = image_pts[0, 2]/image_pts[2, 2]
    v3 = image_pts[1, 2]/image_pts[2, 2]

    u4 = image_pts[0, 3]/image_pts[2, 3]
    v4 = image_pts[1, 3]/image_pts[2, 3]

    u5 = image_pts[0, 4]/image_pts[2, 4]
    v5 = image_pts[1, 4]/image_pts[2, 4]

    u6 = image_pts[0, 5]/image_pts[2, 5]
    v6 = image_pts[1, 5]/image_pts[2, 5]

    A = np.array([[x1, y1, z1, 1., 0., 0., 0., 0., -u1*x1, -u1*y1, -u1*z1, -u1],
                  [0., 0., 0., 0., x1, y1, z1, 1., -v1*x1, -v1*y1, -u1*z1, -v1],

                  [x2, y2, z2, 1., 0., 0., 0., 0., -u2*x2, -u2*y2, -u2*z2, -u2],
                  [0., 0., 0., 0., x2, y2, z2, 1., -v2*x2, -v2*y2, -u2*z2, -v2],

                  [x3, y3, z3, 1., 0., 0., 0., 0., -u3*x3, -u3*y3, -u3*z3, -u3],
                  [0., 0., 0., 0., x3, y3, z3, 1., -v3*x3, -v3*y3, -u3*z3, -v3],

                  [x4, y4, z4, 1., 0., 0., 0., 0., -u4*x4, -u4*y4, -u4*z4, -u4],
                  [0., 0., 0., 0., x4, y4, z4, 1., -v4*x4, -v4*y4, -u4*z4, -v4],
                  
                  [x5, y5, z5, 1., 0., 0., 0., 0., -u5*x5, -u5*y5, -u5*z5, -u5],
                  [0., 0., 0., 0., x5, y5, z5, 1., -v5*x5, -v5*y5, -u5*z5, -v5],

                  [x6, y6, z6, 1., 0., 0., 0., 0., -u6*x6, -u6*y6, -u6*z6, -u6],
                  [0., 0., 0., 0., x6, y6, z6, 1., -v6*x6, -v6*y6, -u6*z6, -v6],
                 ])
    return A


def matrix_condition_number_autograd(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured):
    A = calculate_A_matrix_autograd(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4,
                                    z4, x5, y5, z5, x6, y6, z6, P)

    U, s, V = np.linalg.svd(A, full_matrices=False)

    greatest_singular_value = s[0]
    smallest_singular_value = s[11]
    return greatest_singular_value/smallest_singular_value


def repro_error_autograd(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4,
                         x5, y5, z5, x6, y6, z6, P, image_pts_measured):
    X1 = np.array([[x1], [y1], [z1], [1.]]).reshape(4, 1)
    X2 = np.array([[x2], [y2], [z2], [1.]]).reshape(4, 1)
    X3 = np.array([[x3], [y3], [z3], [1.]]).reshape(4, 1)
    X4 = np.array([[x4], [y4], [z4], [1.]]).reshape(4, 1)
    X5 = np.array([[x5], [y5], [z5], [1.]]).reshape(4, 1)
    X6 = np.array([[x6], [y6], [z6], [1.]]).reshape(4, 1)

    U1 = np.array(np.dot(P, X1)).reshape(3, 1)
    U2 = np.array(np.dot(P, X2)).reshape(3, 1)
    U3 = np.array(np.dot(P, X3)).reshape(3, 1)
    U4 = np.array(np.dot(P, X4)).reshape(3, 1)
    U5 = np.array(np.dot(P, X5)).reshape(3, 1)
    U6 = np.array(np.dot(P, X6)).reshape(3, 1)
    U1 = U1/U1[2, 0]
    U2 = U2/U2[2, 0]
    U3 = U3/U3[2, 0]
    U4 = U4/U4[2, 0]
    U5 = U5/U4[2, 0]
    U6 = U6/U4[2, 0]

    # world_pts = np.hstack([X1, X2, X3, X4, X5, X6])
    image_pts_repro = np.hstack([U1, U2, U3, U4, U5, U6])

    x = image_pts_measured[:2, :]-image_pts_repro[:2, :]
    repro = np.sum(x**2)**(1./2)
    return repro


def create_gradient(metric='condition_number', n=0.000001):
    """"
    metric: 'condition_number' (default)
    """
    if metric == 'condition_number':
        metric_function = matrix_condition_number_autograd
    elif metric == 'repro_error':
        metric_function = repro_error_autograd

    gradient = Gradient()
    gradient.set_n(n)
    gradient.dx1 = grad(metric_function, 0)
    gradient.dy1 = grad(metric_function, 1)
    gradient.dz1 = grad(metric_function, 2)

    gradient.dx2 = grad(metric_function, 3)
    gradient.dy2 = grad(metric_function, 4)
    gradient.dz2 = grad(metric_function, 5)

    gradient.dx3 = grad(metric_function, 6)
    gradient.dy3 = grad(metric_function, 7)
    gradient.dz3 = grad(metric_function, 8)

    gradient.dx4 = grad(metric_function, 9)
    gradient.dy4 = grad(metric_function, 10)
    gradient.dz4 = grad(metric_function, 11)

    gradient.dx5 = grad(metric_function, 12)
    gradient.dy5 = grad(metric_function, 13)
    gradient.dz5 = grad(metric_function, 14)

    gradient.dx6 = grad(metric_function, 15)
    gradient.dy6 = grad(metric_function, 16)
    gradient.dz6 = grad(metric_function, 17)
    return gradient
    
    
    def extract_objectpoints_vars(objectPoints):
    x1 = objectPoints[0, 0]
    y1 = objectPoints[1, 0]
    z1 = objectPoints[2, 0]

    x2 = objectPoints[0, 1]
    y2 = objectPoints[1, 1]
    z2 = objectPoints[2, 1]

    x3 = objectPoints[0, 2]
    y3 = objectPoints[1, 2]
    z3 = objectPoints[2, 2]

    x4 = objectPoints[0, 3]
    y4 = objectPoints[1, 3]
    z4 = objectPoints[2, 3]

    x5 = objectPoints[0, 4]
    y5 = objectPoints[1, 4]
    z5 = objectPoints[2, 4]

    x6 = objectPoints[0, 5]
    y6 = objectPoints[1, 5]
    z6 = objectPoints[2, 5]

    return [x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5,
            x6, y6, z6]
