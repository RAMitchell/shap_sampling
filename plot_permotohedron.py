import numpy as np
from itertools import permutations
from scipy.spatial import SphericalVoronoi, geometric_slerp, Delaunay
from mayavi import mlab


def zero_sum_projection(d):
    basis = np.array([[1.0] * i + [-i] + [0.0] * (d - i - 1) for i in range(1, d)])
    return np.array([v / np.linalg.norm(v) for v in basis])


def get_permutation_points():
    n_features = 4
    p = np.array(list(permutations(np.arange(0, n_features)))) - 1.5
    basis = zero_sum_projection(n_features)
    points = p.dot(basis.T)
    points /= np.linalg.norm(points[0])
    return points


def mayavi_plot_cayley():
    p_int = list(permutations(np.arange(0, 4)))
    points = get_permutation_points()
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
        mlab.text3d(points[i][0] * 1.05, points[i][1] * 1.05, points[i][2] * 1.05,
                    str(np.argsort(p_int[i])),
                    scale=0.05)

    mlab.show()


def mayavi_plot_cayley_voronai():
    points = get_permutation_points()
    p_int = list(permutations(np.arange(0, 4)))
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
        mlab.text3d(points[i][0] * 1.05, points[i][1] * 1.05, points[i][2] * 1.05,
                    str(np.argsort(p_int[i])),
                    scale=0.05)

    mlab.show()


def mayavi_plot_cayley_orthogonal():
    import algorithms
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))
    mlab.clf()
    sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                           scale_factor=2,
                           color=(0.67, 0.77, 0.93),
                           resolution=50,
                           opacity=0.6
                           )
    basis = zero_sum_projection(4)
    orthogonal_sample = algorithms.get_orthogonal_vectors(3, 4)
    orthogonal_sample = np.concatenate(
        (orthogonal_sample, -orthogonal_sample))
    valid_permutations = np.zeros((orthogonal_sample.shape[0], 4), dtype=np.int64)
    valid_permutations_projection = np.zeros((orthogonal_sample.shape[0], 3))
    for i, s in enumerate(orthogonal_sample):
        valid_permutations[i] = np.argsort(s)
        valid_permutations_projection[i] = np.argsort(valid_permutations[i]).dot(basis.T)
        valid_permutations_projection[i] /= np.linalg.norm(valid_permutations_projection[i])

    orthogonal_sample_reduced = orthogonal_sample.dot(basis.T)
    arrow_origin = np.zeros(orthogonal_sample_reduced.shape[0])
    arrows = mlab.quiver3d(arrow_origin, arrow_origin, arrow_origin,
                           orthogonal_sample_reduced[:, 0], orthogonal_sample_reduced[:, 1],
                           orthogonal_sample_reduced[:, 2])
    sphere.actor.property.backface_culling = True
    sphere.actor.property.interpolation = 'flat'
    mlab.points3d(valid_permutations_projection[:, 0], valid_permutations_projection[:, 1],
                  valid_permutations_projection[:, 2], scale_factor=0.05,
                  color=(0.2, 0.2, 0.2))
    for i in range(valid_permutations.shape[0]):
        mlab.text3d(valid_permutations_projection[i][0] * 1.05,
                    valid_permutations_projection[i][1] * 1.05,
                    valid_permutations_projection[i][2] * 1.05,
                    str(valid_permutations[i]),
                    scale=0.05)

    mlab.show()


def plot_fibonacci_sphere():
    from plotoptix import TkOptiX
    from algorithms import uniform_hypersphere
    from plotoptix.materials import m_clear_glass, m_plastic, m_matt_glass

    optix = TkOptiX()  # create and configure, show the window later

    optix.setup_material("plastic", m_plastic)
    optix.setup_material("glass", m_clear_glass)
    optix.setup_material("matt", m_matt_glass)
    optix.set_param(max_accumulation_frames=300)

    points = uniform_hypersphere(3, 100)

    optix.set_data("fibonacci", pos=points, r=0.04, c=(0.5, 0.5, 1.0), mat="plastic")
    optix.set_data("sphere", pos=[(0, 0, 0)], r=1.0, c=(0.9, 0.9, 0.9), mat="plastic")
    optix.show()
    optix.setup_camera("cam1", cam_type="DoF",
                       eye=[-2.1, 2.4, 0], target=[0, 0, 0], up=[0.28, 0.96, 0.05],
                       aperture_radius=0.01, fov=30, focal_scale=0.91)

    optix.setup_light("light1", pos=np.array([4, 5.1, 3]) * 2, color=np.array([12, 11, 10]) * 10,
                      radius=1.9)
    optix.setup_light("light2", pos=np.array([-1.5, 3, -2]) * 2, color=np.array([8, 9, 10]) * 10,
                      radius=1.0)
    optix.set_background(0.99)
    optix.set_ambient(0.5)

    exposure = 0.4;
    gamma = 2.2
    optix.set_float("tonemap_exposure", exposure)
    optix.set_float("tonemap_gamma", gamma)
    optix.set_float("denoiser_blend", 0.25)
    optix.add_postproc("Denoiser")
    optix.camera_fit(geometry="sphere")


# mayavi_plot_cayley()
mayavi_plot_cayley_orthogonal()
# plot_fibonacci_sphere()
