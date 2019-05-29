"""
Created on Thu Mar 21 16:20:27 2019

@author: diana, raulacuna
"""
import autograd.numpy as np
from autograd import grad
from vision.camera import Camera
from python.sphere import Sphere
import python.dlt_and_errors as dLt
import random
from numpy import linalg as LA
import csv
with open('error_r.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_t.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')
with open('error_reproject.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ') 
with open('condition.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=' ')

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

def evaluate_gradient(gradient, objectPoints, P, image_pts_measured):
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6 = extract_objectpoints_vars(objectPoints)

    gradient.dx1_eval_old = gradient.dx1_eval
    gradient.dy1_eval_old = gradient.dy1_eval
    gradient.dz1_eval_old = gradient.dz1_eval

    gradient.dx2_eval_old = gradient.dx2_eval
    gradient.dy2_eval_old = gradient.dy2_eval
    gradient.dz2_eval_old = gradient.dz2_eval

    gradient.dx3_eval_old = gradient.dx3_eval
    gradient.dy3_eval_old = gradient.dy3_eval
    gradient.dz3_eval_old = gradient.dz3_eval

    gradient.dx4_eval_old = gradient.dx4_eval
    gradient.dy4_eval_old = gradient.dy4_eval
    gradient.dz4_eval_old = gradient.dz4_eval

    gradient.dx5_eval_old = gradient.dx5_eval
    gradient.dy5_eval_old = gradient.dy5_eval
    gradient.dz5_eval_old = gradient.dz5_eval

    gradient.dx6_eval_old = gradient.dx6_eval
    gradient.dy6_eval_old = gradient.dy6_eval
    gradient.dz6_eval_old = gradient.dz6_eval

    gradient.dx1_eval = gradient.dx1(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_x1
    gradient.dy1_eval = gradient.dy1(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                     x4, y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_y1
    gradient.dz1_eval = gradient.dz1(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_z1

    gradient.dx2_eval = gradient.dx2(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_x2
    gradient.dy2_eval = gradient.dy2(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                     x4, y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_y2
    gradient.dz2_eval = gradient.dz2(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_z2

    gradient.dx3_eval = gradient.dx3(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_x3
    gradient.dy3_eval = gradient.dy3(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                     x4, y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_y3
    gradient.dz3_eval = gradient.dz3(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_z3

    gradient.dx4_eval = gradient.dx4(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_x4
    gradient.dy4_eval = gradient.dy4(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                     x4, y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_y4
    gradient.dz4_eval = gradient.dz4(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_z4

    gradient.dx5_eval = gradient.dx5(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_x5
    gradient.dy5_eval = gradient.dy5(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                     x4, y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_y5
    gradient.dz5_eval = gradient.dz5(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_z5

    gradient.dx6_eval = gradient.dx6(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_x6
    gradient.dy6_eval = gradient.dy6(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                     x4, y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_y6
    gradient.dz6_eval = gradient.dz6(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                                     y4, z4, x5, y5, z5, x6, y6, z6, P,
                                     image_pts_measured)*gradient.n_z6

    gradient.n_x1 = supersab(gradient.n_x1, gradient.dx1_eval,
                             gradient.dx1_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_x2 = supersab(gradient.n_x2, gradient.dx2_eval,
                             gradient.dx2_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_x3 = supersab(gradient.n_x3, gradient.dx3_eval,
                             gradient.dx3_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_x4 = supersab(gradient.n_x4, gradient.dx4_eval,
                             gradient.dx4_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_x5 = supersab(gradient.n_x5, gradient.dx5_eval,
                             gradient.dx5_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_x6 = supersab(gradient.n_x6, gradient.dx6_eval,
                             gradient.dx6_eval_old, gradient.n_pos,
                             gradient.n_neg)

    gradient.n_y1 = supersab(gradient.n_y1, gradient.dy1_eval,
                             gradient.dy1_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_y2 = supersab(gradient.n_y2, gradient.dy2_eval,
                             gradient.dy2_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_y3 = supersab(gradient.n_y3, gradient.dy3_eval,
                             gradient.dy3_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_y4 = supersab(gradient.n_y4, gradient.dy4_eval,
                             gradient.dy4_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_y5 = supersab(gradient.n_y5, gradient.dy5_eval,
                             gradient.dy5_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_y6 = supersab(gradient.n_y6, gradient.dy6_eval,
                             gradient.dy6_eval_old, gradient.n_pos,
                             gradient.n_neg)

    gradient.n_z1 = supersab(gradient.n_z1, gradient.dz1_eval,
                             gradient.dz1_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_z2 = supersab(gradient.n_z2, gradient.dz2_eval,
                             gradient.dz2_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_z3 = supersab(gradient.n_z3, gradient.dz3_eval,
                             gradient.dz3_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_z4 = supersab(gradient.n_z4, gradient.dz4_eval,
                             gradient.dz4_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_z5 = supersab(gradient.n_z5, gradient.dz5_eval,
                             gradient.dz5_eval_old, gradient.n_pos,
                             gradient.n_neg)
    gradient.n_z6 = supersab(gradient.n_z6, gradient.dz6_eval,
                             gradient.dz6_eval_old, gradient.n_pos,
                             gradient.n_neg)
    # # Limit
    limit = 0.05
    gradient.dx1_eval = np.clip(gradient.dx1_eval, -limit, limit)
    gradient.dy1_eval = np.clip(gradient.dy1_eval, -limit, limit)
    gradient.dz1_eval = np.clip(gradient.dz1_eval, -limit, limit)

    gradient.dx2_eval = np.clip(gradient.dx2_eval, -limit, limit)
    gradient.dy2_eval = np.clip(gradient.dy2_eval, -limit, limit)
    gradient.dz2_eval = np.clip(gradient.dz2_eval, -limit, limit)

    gradient.dx3_eval = np.clip(gradient.dx3_eval, -limit, limit)
    gradient.dy3_eval = np.clip(gradient.dy3_eval, -limit, limit)
    gradient.dz3_eval = np.clip(gradient.dy3_eval, -limit, limit)

    gradient.dx4_eval = np.clip(gradient.dx4_eval, -limit, limit)
    gradient.dy4_eval = np.clip(gradient.dy4_eval, -limit, limit)
    gradient.dz4_eval = np.clip(gradient.dz4_eval, -limit, limit)

    gradient.dx5_eval = np.clip(gradient.dx5_eval, -limit, limit)
    gradient.dy5_eval = np.clip(gradient.dy5_eval, -limit, limit)
    gradient.dz5_eval = np.clip(gradient.dz5_eval, -limit, limit)

    gradient.dx6_eval = np.clip(gradient.dx6_eval, -limit, limit)
    gradient.dy6_eval = np.clip(gradient.dy6_eval, -limit, limit)
    gradient.dz6_eval = np.clip(gradient.dz6_eval, -limit, limit)

    return gradient

def supersab(n, gradient_eval_current, gradient_eval_old, n_pos, n_neg):
    if np.sign(gradient_eval_current*gradient_eval_old) > 0:
        n = n + n_pos
    else:
        n = n*n_neg
    return n


def update_points(gradient, objectPoints, limit):
    op = np.copy(objectPoints)
    op[0, 0] += - gradient.dx1_eval
    op[1, 0] += - gradient.dy1_eval
    op[2, 0] += - gradient.dz1_eval

    op[0, 1] += - gradient.dx2_eval
    op[1, 1] += - gradient.dy2_eval
    op[2, 1] += - gradient.dz2_eval

    op[0, 2] += - gradient.dx3_eval
    op[1, 2] += - gradient.dy3_eval
    op[2, 2] += - gradient.dz3_eval

    op[0, 3] += - gradient.dx4_eval
    op[1, 3] += - gradient.dy4_eval
    op[2, 3] += - gradient.dz4_eval

    op[0, 4] += - gradient.dx5_eval
    op[1, 4] += - gradient.dy5_eval
    op[2, 4] += - gradient.dz5_eval

    op[0, 5] += - gradient.dx6_eval
    op[1, 5] += - gradient.dy6_eval
    op[2, 5] += - gradient.dz6_eval

    radius = sph.radius  # define limits x,y,z
    for i in range(op.shape[1]):
        distance = np.sqrt(op[0, i]**2+op[1, i]**2+op[2, i]**2)
        if distance > radius:
            op[:4, i] = op[:4, i]*radius/distance
        else:
            op[0, :] = np.clip(op[0, :], -limit, limit)
            op[1, :] = np.clip(op[1, :], -limit, limit)
            op[2, :] = np.clip(op[2, :], -limit, limit)
    return op

def plot_error_r():
    x = []
    y = []
    with open('error_r.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for column in plots:
            x.append((float(column[1])))
            y.append((float(column[0])))
        plt.plot(x, y, label='Loaded from file!')
        plt.xlabel('number of iterations')
        plt.ylabel('Rotation Error (degrees))')
        plt.title('Rotation Error while changing the points')
        plt.legend()
        plt.show()

def mean_noise_points(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5,
                      z5, x6, y6, z6):
    errorpoints = np.full((3, 6), 0.0)
    real_points = np.array([[x1, x2, x3, x4, x5, x6],
                            [y1, y2, y3, y4, y5, y6],
                            [z1, z2, z3, z4, z5, z6],
                            [1., 1., 1., 1., 1., 1.]])
    imagepoints = cam.project(real_points)
    for i in range(2000):
        errorpoints = errorpoints + add_noise(imagepoints, sd=8., mean=2.,
                                              size=0)
    errorpoints = errorpoints/2000.
    H = dLt.DLT3D(cam, real_points, errorpoints, normalization=False)
    forreproject = dLt.DLTproject(H, real_points, quant_error=False)
    reproject = dLt.reprojection_error(imagepoints, forreproject, 6)
    estim_R = dLt.estimate_R_withQR(H)
    estim_center = dLt.camera_center(H)
    estim_t = dLt.estimate_t(cam, estim_center)
    t_error = dLt.error_t(cam, estim_t)
    r_angle = dLt.error_R(cam, estim_R)
    return t_error, reproject, r_angle


def plot_error_r():
    x = []
    y = []
    with open('error_r.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for column in plots:
            x.append((float(column[1])))
            y.append((float(column[0])))
        plt.figure(19)
        plt.plot(x, y, c='y', label='random noise mean=2, sd=8, points r<0.5')
        plt.xlabel('number of iterations')
        plt.ylabel('Rotation Error (degrees)')
        plt.title('Rotation Error while changing the points')
        plt.legend()
        plt.show()


def plot_error_t():
    x = []
    y = []
    with open('error_t.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for column in plots:
            x.append((float(column[1])))
            y.append((float(column[0])))
        plt.figure(21)
        plt.plot(x, y, label='random noise mean=2, sd=8, points r<0.5')
        plt.xlabel('number of iterations')
        plt.ylabel('Translation Error (100%)')
        plt.title('Translation Error while changing the set of points')
        plt.legend()
        plt.show()


def plot_reprojection_error():
    x = []
    y = []
    with open('error_reproject.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for column in plots:
            x.append((float(column[1])))
            y.append((float(column[0])))
        plt.figure(20)
        plt.plot(x, y, c='g', label='random noise mean=2 , sd=8, points r<0.5')
        plt.xlabel('number of iterations')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error while changing the set of points')
        plt.legend()
        plt.show()


def plot_condition():
    x = []
    y = []
    with open('condition.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for column in plots:
            x.append((float(column[1])))
            y.append((float(column[0])))
        plt.figure(22)
        plt.plot(x, y, c='r', label='random noise mean=2, sd=8, points r<0.5')
        plt.xlabel('number of iterations')
        plt.ylabel('Condition Number')
        plt.title('Condition Number while changing the set of points')
        plt.legend()
        plt.show()        

def mean_noise_points(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5,
                      z5, x6, y6, z6):
    errorpoints = np.full((3, 6), 0.0)
    real_points = np.array([[x1, x2, x3, x4, x5, x6],
                            [y1, y2, y3, y4, y5, y6],
                            [z1, z2, z3, z4, z5, z6],
                            [1., 1., 1., 1., 1., 1.]])
    imagepoints = cam.project(real_points)
    for i in range(1000):
        errorpoints = errorpoints + add_noise(imagepoints, sd=8., mean=2.,
                                              size=0)
    errorpoints = errorpoints/1000.
    H = dlt.DLT3D(cam, real_points, errorpoints, normalization=False)
    covar = np.cov(H)
    conditioncov = LA.cond(covar)
    forreproject = dlt.DLTproject(H, real_points, quant_error=False)
    reproject = dlt.reprojection_error(imagepoints, forreproject, 6)
    estim_R = dlt.estimate_R_withQR(H)
    estim_center = dlt.camera_center(H)
    estim_t = dlt.estimate_t(cam, estim_center)
    t_error = dlt.error_t(cam, estim_t)
    r_angle = dlt.error_R(cam, estim_R)
    return t_error, reproject, r_angle, conditioncov           
# -------  DEFINE CAMERA AND SPHERE
cam = Camera()


sph = Sphere()

cam.set_K(fx=800., fy=800., cx=640., cy=480.)
cam.set_width_heigth(1280, 960)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
cam.set_t(0.0, -0.0, 0.5, frame='world')

# ----- TEST AND PLOT OPTIMAL-----------------
gradient = create_gradient(metric='condition_number')
objectPoints_des = sph.get_sphere_points()
imagePoints_des = np.array(cam.project(objectPoints_des, False))
objectPoints_list = list()
imagePoints_list = list()
transfer_error_list = list()
new_objectPoints = objectPoints_des
for i in range(100):
    objectPoints = np.copy(new_objectPoints)
    gradient = evaluate_gradient(gradient, objectPoints,
                                 np.array(cam.P), imagePoints_des)
    new_objectPoints = update_points(gradient, objectPoints, sph.radius)
    new_imagePoints = np.array(cam.project(new_objectPoints, False))
    objectPoints_list.append(new_objectPoints)
    imagePoints_list.append(new_imagePoints)
    if (i == 1 or i == 50 or i == 99):
        # plt.figure('Object Points')    
        phisim = np.linspace((-math.pi)/2., (math.pi/2.))
        thetasim = np.linspace(0, 2 * np.pi)
        x = np.outer(np.sin(thetasim), np.cos(phisim))
        y = np.outer(np.sin(thetasim), np.sin(phisim))
        z = np.outer(np.cos(thetasim), np.ones_like(phisim))
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_wireframe(sph.radius*x, sph.radius*y,
                          sph.radius*z, color='g')
        plt.plot(new_objectPoints[0], new_objectPoints[1], new_objectPoints[2],
                 '.', color='blue',)
        plt.pause(0.6)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        plt.plot(new_objectPoints[0], new_objectPoints[1], new_objectPoints[2],
                 '.', color='red',)
        plt.pause(0.6)


    x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6 = extract_objectpoints_vars(new_objectPoints)
    
    t_error, reproject, r_error = mean_noise_points(x1, y1, z1, x2, y2, z2, x3,
                                                    y3, z3, x4, y4, z4, x5, y5,
                                                    z5, x6, y6, z6)
    with open('error_t.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(t_error), number])
    with open('error_r.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(r_error), number])
    with open('error_reproject.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(reproject), number])
    mat_cond_autrograd = matrix_condition_number_autograd(x1, y1, z1, x2, y2,
                                                          z2, x3, y3, z3, x4,
                                                          y4, z4, x5, y5, z5,
                                                          x6, y6, z6,
                                                          np.array(cam.P),
                                                          imagePoints_des)
    with open('condition.csv', 'ab') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=' ')
            filewriter.writerow([float(mat_cond_autrograd), number])
    number = number + 1
  
plot_error_t()
plot_error_r()
plot_reprojection_error()
plot_condition()
