"""
Created on 2024/6/13 
Author: Hao Chen (chen960216@gmail.com)
"""
from typing import List

import numpy as np
import modeling.geometric_model as gm
import basis.robot_math as rm
from scipy.spatial import ConvexHull

def gaussian(x, mu, sigma):
    """ Gaussian function. """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def assign_force_values(points, point_center, applied_force, b):
    point_center = np.array(point_center)
    points = np.array(points)

    distances = np.linalg.norm(points - point_center, axis=1)

    # Calculate Gaussian weights
    sigma = b / 3  # assuming b is the range for ~99.7% of the data in Gaussian distribution
    weights = gaussian(distances, 0, sigma)

    # Normalize weights so that their sum is 1
    weights_sum = np.sum(weights)
    normalized_weights = weights / weights_sum

    # Scale normalized weights to match the applied force
    force_values = normalized_weights * applied_force

    return force_values.tolist()


def gen_hex_vec(coefficient, height=1, polygon=6, normal=np.array([0, 0, 1])):
    '''generate vectors for simulating the friction cone'''
    init_mat = np.zeros([polygon, 3])
    # initVector =pg.rm.unit_vector(np.array([0,coefficient,1]))
    init_vec = np.array([0, coefficient, 1]) * height
    # tf_mat = pg.trigeom.align_vectors(vector_start=np.array([0, 0, 1]), vector_end=-normal)
    tf_rot = rm.rotmat_between_vectors(np.array([0, 0, 1]), -normal)
    tf_mat = rm.homomat_from_posrot([0, 0, 0], tf_rot)
    init_mat[0, :] = rm.homomat_transform_points(tf_mat, init_vec)
    angle = np.radians(360.0) / polygon
    for i in range(1, polygon):
        rotMat = rm.rotmat_from_axangle([0, 0, 1], angle * i)
        init_mat[i, :] = rm.homomat_transform_points(tf_mat, np.dot(rotMat, init_vec))
    return init_mat


def plot_FC(FC_vectors, position, ratio=1.0):
    for i in FC_vectors:
        gm.gen_arrow(spos=position, epos=position + i * ratio,
                     rgba=np.array([0, 0, 1, 1])).attach_to(base)  # length =5


def gen_FC(position, u=0.25, normal=np.array([0, 0, 1]), polygon=6, height=1, show=False):
    # Coulomb friction model
    # friction coefficient u
    fc_u = u
    FC = gen_hex_vec(coefficient=fc_u, height=height, polygon=polygon, normal=normal)
    if show:
        # base.pggen.plotSphere(base.render, pos=position, radius=5, rgba=[0, 1, 0, 1])
        if height > 1e-2:
            plot_FC(FC, position, ratio=height)
    return FC


def minidistance_hull(p, hull):
    if not isinstance(hull, ConvexHull):
        hull = ConvexHull(hull, qhull_options="QJ")
    return np.max(np.dot(hull.equations[:, :-1], p.T).T + hull.equations[:, -1], axis=-1)

def cal_stability(point_clutter_1: List[np.ndarray],
                  point_center_1: np.ndarray,
                  point_clutter_2: List[np.ndarray],
                  point_center_2: np.ndarray,
                  obj_com: np.ndarray,
                  applied_force_1: int or float,
                  applied_force_2: int or float, ):
    applied_force_1 = assign_force_values(point_clutter_1, point_center_1, applied_force_1, 0.003)
    applied_force_2 = assign_force_values(point_clutter_2, point_center_2, applied_force_2, 0.003)
    FC_list_1 = []
    FC_list_2 = []
    for f in applied_force_1:
        FC_list_1.append(gen_FC(position=obj_com, u=f, show=False))
    for f in applied_force_2:
        FC_list_2.append(gen_FC(position=obj_com, u=f, show=False))
    wrench_list_1 = []
    wrench_list_2 = []
    for force in FC_list_1:
        for f in force:
            wrench_list_1.append(np.hstack([f, np.cross(obj_com, f)]))
    for force in FC_list_2:
        for f in force:
            wrench_list_2.append(np.hstack([f, np.cross(obj_com, f)]))
    wrench = np.array(wrench_list_1 + wrench_list_2)
    hull = ConvexHull(wrench, qhull_options="QJ")
    mindistance = minidistance_hull(np.array([0, 0, 0]), hull)
    return mindistance