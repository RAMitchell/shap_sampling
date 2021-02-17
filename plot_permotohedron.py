import numpy as np
from itertools import permutations
from scipy.spatial import SphericalVoronoi, geometric_slerp, Delaunay
from mayavi import mlab


def find_basis(n_features):
    A = np.zeros((n_features * 2, n_features))
    identity = np.arange(0, n_features)
    for i in range(n_features * 2):
        A[i] = np.random.permutation(n_features) - identity
    B = np.ones((n_features * 2, n_features + 1))
    B[:, :-1] = A
    u, s, vh = np.linalg.svd(A)
    return vh[:-1]


n_features = 4
p_int = list(permutations(np.arange(0, n_features)))
p = np.array(list(permutations(np.arange(0, n_features)))) - 1.5
basis = find_basis(4)
points = p.dot(basis.T)
points /= np.linalg.norm(points[0])


def mayavi_plot_cayley():
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))
    mlab.clf()

    # Create and visualize the mesh
    tri = Delaunay(points)
    mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], tri.convex_hull, opacity=0.7,
                         color=(52 / 255, 168 / 255, 235 / 255))
    poly = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05,
                         color=(0.2, 0.2, 0.2))
    poly.actor.property.backface_culling = True

    for i in range(points.shape[0]):
        mlab.text3d(points[i][0] * 1.05, points[i][1] * 1.05, points[i][2] * 1.05, str(np.argsort(p_int[i])),
                    scale=0.05)

    mlab.show()


def mayavi_plot_cayley_voronai():
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))
    mlab.clf()
    sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                           scale_factor=2,
                           color=(0.67, 0.77, 0.93),
                           resolution=50,
                           opacity=1.0
                           )

    sphere.actor.property.backface_culling = True
    sphere.actor.property.interpolation = 'flat'
    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(points, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05,
                  color=(0.2, 0.2, 0.2))
    mlab.points3d(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], scale_factor=0.05,
                  color=(0.2, 0.5, 0.2))
    # indicate Voronoi regions (as Euclidean polygons)
    t_vals = np.linspace(0, 1, 100)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = np.array(geometric_slerp(start, end, t_vals))
            mlab.plot3d(result[:, 0],
                        result[:, 1],
                        result[:, 2],
                        color=(0.8, 0.8, 0.8), opacity=0.5, tube_radius=None)
    for i in range(points.shape[0]):
        mlab.text3d(points[i][0] * 1.05, points[i][1] * 1.05, points[i][2] * 1.05, str(np.argsort(p_int[i])),
                    scale=0.05)

    mlab.show()


mayavi_plot_cayley_voronai()
