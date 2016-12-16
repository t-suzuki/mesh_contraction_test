# -*- coding: utf-8 -*-
# contraction part proposedd in :
# "Skeleton Extraction by Mesh Contraction", O.Au+, SIGGRAPH 2008
# - http://graphics.csie.ncku.edu.tw/paper_video/ACM_SIGGRAPH/siggraph2008/Skeleton/skeleton-paperfinal_old.pdf
#
# solve [WL*L; WH] * V' = [0; WH*V] iteratively.

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

use_sparse = True

def generate_model(sigma = 0.1):
    # 0---1---2---3
    #  \ / \ / \ / \
    #   4---5---6--7
    #  / \ / \ / \ /
    # 8---9---a---b
    nrows = 5
    ncols = 20
    nodes = []
    faces = []
    for irow in range(nrows):
        xs = np.arange(ncols) + 0.5 * (irow % 2)
        ys = np.ones(ncols) * irow
        zs = np.zeros(ncols)
        nodes.extend(zip(xs, ys, zs))
    nodes = np.vstack(nodes)
    nodes = nodes.astype(np.float64) + np.random.randn(*nodes.shape) * sigma
    # i0---i1
    #   \ /
    #    i2
    for irow in range(0, nrows - 1):
        for icol in range(ncols - 1):
            i0 = irow * ncols + icol
            i1 = i0 + 1
            i2 = i0 + ncols + irow % 2
            faces.append((i0, i1, i2))
    #    i2
    #   /  \
    # i0---i1
    for irow in range(1, nrows):
        for icol in range(ncols - 1):
            i0 = irow * ncols + icol
            i1 = i0 + 1
            i2 = i0 - ncols + irow % 2
            faces.append((i0, i1, i2))
    return nodes, faces

def plot_model(nodes, faces, color, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'o', color=color)
    for i0, i1, i2 in faces:
        ax.plot(
                [nodes[i0, 0], nodes[i1, 0], nodes[i2, 0], nodes[i0, 0]],
                [nodes[i0, 1], nodes[i1, 1], nodes[i2, 1], nodes[i0, 1]],
                [nodes[i0, 2], nodes[i1, 2], nodes[i2, 2], nodes[i0, 2]],
                '-', alpha=0.5, color='gray')
    return ax

def cot(nodes, i0, i1, i2):
    h0 = nodes[i0] - nodes[i2]
    h1 = nodes[i1] - nodes[i2]
    return 1.0 / np.tan(np.arccos(h0.dot(h1) / (np.linalg.norm(h0) * np.linalg.norm(h1))))

def Laplacian(nodes, faces):
    n = nodes.shape[0]
    L = np.zeros((3 * n, 3 * n))
    for i0, i1, i2 in faces:
        for e0, e1, v in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)]:
            L_ij = cot(nodes, e0, e1, v)
            L[3 * e0 + 0, 3 * e1 + 0] += L_ij
            L[3 * e0 + 1, 3 * e1 + 1] += L_ij
            L[3 * e0 + 2, 3 * e1 + 2] += L_ij
            L[3 * e1 + 0, 3 * e0 + 0] += L_ij
            L[3 * e1 + 1, 3 * e0 + 1] += L_ij
            L[3 * e1 + 2, 3 * e0 + 2] += L_ij
    # diag
    for i in range(n):
        L_ii = - L[3 * i + 0, :].sum()
        L[3 * i + 0, 3 * i + 0] = L_ii
        L[3 * i + 1, 3 * i + 1] = L_ii
        L[3 * i + 2, 3 * i + 2] = L_ii
    #print L
    #plt.imshow(L)
    #plt.show()
    if use_sparse: return scipy.sparse.csc_matrix(L)
    else:          return L

def one_ring_vector(nodes, faces):
    u'''A[i] = one-ring area for nodes[i].'''
    n = nodes.shape[0]
    A = np.zeros(n)
    for i0, i1, i2 in faces:
        area = 0.5 * np.linalg.norm(np.cross(nodes[i0] - nodes[i2], nodes[i1] - nodes[i2]))
        A[i0] += area
        A[i1] += area
        A[i2] += area
    return A

def average_face_area(nodes, faces):
    n = nodes.shape[0]
    A = 0.0
    for i0, i1, i2 in faces:
        area = 0.5 * np.linalg.norm(np.cross(nodes[i0] - nodes[i2], nodes[i1] - nodes[i2]))
        A += area
    return A / len(faces)

def expand3(A):
    n = A.shape[0]
    M = np.zeros(3 * n)
    for i in range(n):
        M[3 * i + 0] = A[i]
        M[3 * i + 1] = A[i]
        M[3 * i + 2] = A[i]
    return M

def make_V(nodes):
    n = nodes.shape[0]
    V = np.zeros((3 * n, 1))
    for i in range(n):
        V[3 * i + 0, 0] = nodes[i, 0]
        V[3 * i + 1, 0] = nodes[i, 1]
        V[3 * i + 2, 0] = nodes[i, 2]
    return V

def parse_V(V):
    V = V.reshape((-1, 1))
    n = V.shape[0] // 3
    return np.vstack([(V[3 * i + 0, 0], V[3 * i + 1, 0], V[3 * i + 2, 0])] for i in range(n))

def contraction_step(nodes, faces, A0, WL, WH0):
    n = nodes.shape[0]
    L = Laplacian(nodes, faces)
    A = one_ring_vector(nodes, faces)
    if use_sparse: WH = WH0 * scipy.sparse.diags(expand3(np.sqrt(A0 / A)), 0)
    else:          WH = WH0 * np.diag(expand3(np.sqrt(A0 / A)), 0)
    V = make_V(nodes)
    # [WL*L; WH] * V' = [0; WH*V]
    Z = np.zeros_like(V)
    if use_sparse: Q = scipy.sparse.vstack([WL*L, WH])
    else:          Q = np.vstack([WL*L, WH])
    R = np.vstack([Z, WH.dot(V)])
    if use_sparse: newV = scipy.sparse.linalg.lsqr(Q, R, show=False)[0]
    else:          newV = np.linalg.lstsq(Q, R)[0]
    nodes = parse_V(newV)
    return nodes

def contraction(nodes, faces, WL0, sL, n_iter):
    A0 = one_ring_vector(nodes, faces)
    WH0 = 1.0 # from the paper, 1.0
    print('WL_0 = {}, WH_0 = {}'.format(WL0, WH0))
    WL = WL0
    for i in range(n_iter):
        print('iter = {}'.format(i))
        nodes = contraction_step(nodes, faces, A0, WL, WH0)
        WL *= sL
    return nodes

if __name__ == '__main__':
    np.random.seed(0)
    nodes, faces = generate_model(0.1)
    print('n = {}'.format(nodes.shape[0]))
    ax = plot_model(nodes, faces, 'blue')

    n_iter = 14
    A = average_face_area(nodes, faces)
    WL = 1.0e-3 * A ** 0.5 # from the paper, 1.0e-3 * A^0.5
    print('A = {}, WL_0 = {}'.format(A, WL))
    sL = 2.0 # from the paper, 2.0
    A0 = one_ring_vector(nodes, faces)
    #nodes = contraction_step(nodes, faces, A0, WL, 1.0)
    nodes = contraction(nodes, faces, WL, sL, n_iter)

    plot_model(nodes, faces, 'red', ax)
    plt.show()

