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


def mayavi_plot_cayley_3d():
    p_int = np.array(list(permutations(np.arange(0, 4))))
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
                    str(np.argsort(p_int[i]) + 1),
                    scale=0.05)

    mlab.view(focalpoint=(0, 0, 0), distance=4.5)
    mlab.savefig("figures/cayley.png")
    # mlab.show()


def mayavi_plot_cayley_2d():
    p_int = list(permutations(np.arange(0, 3)))
    points = np.array(p_int)
    points_nsphere = np.array([x / np.linalg.norm(x) for x in points - 1])
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))
    mlab.clf()

    # axis
    mlab.quiver3d(0, 0, 0, 1, 0, 0, color=(1, 0, 0))
    mlab.quiver3d(0, 0, 0, 0, 1, 0, color=(0, 1, 0))
    mlab.quiver3d(0, 0, 0, 0, 0, 1, color=(0, 0, 1))

    # norm
    n_len = np.sqrt(3)
    mlab.quiver3d(1, 1, 1, n_len, n_len, n_len, color=(0, 0, 0))
    # Create and visualize the mesh
    tri = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 3, 5]]
    mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], tri, opacity=0.7,
                         color=(52 / 255, 168 / 255, 235 / 255))
    poly = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05,
                         color=(0.2, 0.2, 0.2))
    circle = np.array(geometric_slerp(points_nsphere[2], points_nsphere[1], np.linspace(0, 1, 100)))
    circle = np.append(circle, np.array(
        geometric_slerp(points_nsphere[1], points_nsphere[5], np.linspace(0, 1, 100))), axis=0)
    circle = np.append(circle, np.array(
        geometric_slerp(points_nsphere[5], points_nsphere[2], np.linspace(0, 1, 100))), axis=0)
    circle = (circle * np.sqrt(2)) + 1
    mlab.plot3d(circle[:, 0],
                circle[:, 1],
                circle[:, 2],
                color=(0.8, 0.8, 0.8), opacity=0.9, tube_radius=None)
    poly.actor.property.backface_culling = True

    for i in range(points.shape[0]):
        mlab.text3d(points[i][0] * 1.05, points[i][1] * 1.05, points[i][2] * 1.05,
                    str(np.argsort(p_int[i]) + 1),
                    scale=0.10)

    mlab.view(focalpoint=(1, 1, 1), distance=6, elevation=80, azimuth=5)
    mlab.savefig("figures/cayley2d.png")
    # mlab.show()


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
                    str(np.argsort(p_int[i]) + 1),
                    scale=0.05)
    mlab.view(focalpoint=(0, 0, 0), distance=4.5, azimuth=50)
    mlab.savefig("figures/cayley_voronoi.png")
    # mlab.show()


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
    mlab.quiver3d(arrow_origin, arrow_origin, arrow_origin,
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
                    str(valid_permutations[i] + 1),
                    scale=0.05)

    mlab.view(focalpoint=(0, 0, 0), distance=4.5, azimuth=79, elevation=20)
    mlab.savefig("figures/ortho.png")
    # mlab.show()


def plot_sobol_sphere():
    from sobol_sphere import sobol_sphere
    points = sobol_sphere(200, 3)
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
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05,
                  color=(0.5, 0.2, 0.2))

    mlab.view(focalpoint=(0, 0, 0), distance=4.5)
    mlab.savefig("figures/sobol_sphere.png")
    # mlab.show()


def orthogonal_samples_kt():
    import kernel_methods
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import algorithms
    plt.style.use("seaborn")
    n = 10000
    num_features = [10, 100]
    kernel = kernel_methods.KTKernel()
    df = pd.DataFrame(columns=["sample", "kernel"])
    for d in num_features:
        for _ in range(n):
            # sample 2 orthogonal vectors
            orth = algorithms._orthogonal_permutations((d - 1) * 2, d)
            sigma = orth[0]
            rand = np.random.permutation(d)
            df = df.append({"sample": "random", "kernel": kernel(sigma, rand)}, ignore_index=True)
            df = df.append({"sample": "orth",
                            "kernel": kernel(sigma, orth[np.random.randint(1, orth.shape[0])])},
                           ignore_index=True)
        # sns.set(rc={'figure.figsize': (5.2, 3.62)})
        sns.set_style({'font.family': 'serif'})
        sns.displot(df, x="kernel", hue="sample", kind="kde", fill=True, height=3.62,
                    aspect=4.4 / 3.62)
        plt.xlabel("$K_{\\tau}(\sigma,\sigma')$")
        plt.xlim((-1, 1))
        plt.tight_layout()
        plt.savefig("figures/orthogonal_kt_d" + str(d) + ".png", dpi=200)
        plt.show()


def orthogonal_dot_product():
    import kernel_methods
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.style.use("seaborn")
    n = 10000
    d = 15
    kernel = kernel_methods.KTKernel()
    I = np.arange(d)
    df = pd.DataFrame(columns=["sample", "dot", "kernel"])
    p = np.sqrt(d * (d ** 2 - 1) / 12)
    mu = (I + 1).mean()
    A = lambda y: ((y + 1) - mu) / p
    upper_bound = lambda k: 2 + 3 * k - 4 * np.power((1 + k) / 2, 3 / 2)
    lower_bound = lambda k: -2 + 3 * k + 4 * np.power((1 - k) / 2, 3 / 2)
    for _ in range(n):
        rand = np.random.permutation(d)
        dot = A(I).dot(A(rand))
        df = df.append({"sample": "random", "dot": dot, "kernel": kernel(I, rand)},
                       ignore_index=True)

    for k in np.linspace(-1, 1, 1000):
        df = df.append({"sample": "upper bound", "dot": upper_bound(k), "kernel": k},
                       ignore_index=True)
        df = df.append({"sample": "lower bound", "dot": lower_bound(k), "kernel": k},
                       ignore_index=True)

    sns.set(rc={'figure.figsize': (5.2, 3.62)})
    sns.set_style({'font.family': 'serif'})
    markers = {"random": "X", "upper bound": ".", "lower bound": ".", "new upper bound": "."}
    sns.scatterplot(data=df, x="kernel", y="dot", s=5, style="sample", markers=markers,
                    hue="sample", linewidth=0)
    plt.xlabel("$K_{\\tau}(I,\sigma)$")
    plt.ylabel("$A(I)^TA(\sigma)$")
    plt.xlim((-1, 1))
    plt.tight_layout()
    plt.savefig("figures/orthogonal_dot_product" + str(d) + ".png", dpi=200)
    # plt.show()


# mayavi_plot_cayley_3d()
# mayavi_plot_cayley_2d()
# mayavi_plot_cayley_voronai()
# mayavi_plot_cayley_orthogonal()
# plot_sobol_sphere()
# orthogonal_samples_kt()
# orthogonal_dot_product()
