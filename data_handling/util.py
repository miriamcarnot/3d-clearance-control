import numpy as np

def dist_to_origin(points):
    points = np.asarray(points)
    distances = np.linalg.norm(points, axis=1)
    return distances


def remove_close(points, distances, radius):
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    filtered_points = points[distances >= radius]
    return filtered_points