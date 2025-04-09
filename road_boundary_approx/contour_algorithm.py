import numpy as np
import math
import ray
from scipy.spatial import KDTree
import open3d as o3d
import copy
import statistics


def find_contour_points(sampled_pcd, radius=6, angle_thresh=90):
    """
    creates a tree of all points and checks which of them are contour points
    uses ray for multiprocessing
    creates a new point cloud including all the contour points
    Parameters
    ----------
    radius: size of the neighborhood radius
    angle_thresh: threshold for the angle between the neighbors
    sampled_pcd: the sampled point cloud of the road

    Returns
    -------
    a point cloud including all contour points
    """
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    cont_candidates = []
    points = np.asarray(sampled_pcd.points)
    tree = KDTree(points)

    for voxel in points:
        cont_candidates.append(is_cont_candidate.remote(query_point=voxel, all_points=points, radius=radius, tree=tree, angle_thresh=angle_thresh))

    # Wait for the tasks to complete and retrieve the results
    results = ray.get(cont_candidates)
    cont_voxels = [x for x in results if isinstance(x, np.ndarray)]
    contour_points = np.vstack(cont_voxels)

    cont_pcd = o3d.geometry.PointCloud()
    cont_pcd.points = o3d.utility.Vector3dVector(contour_points)
    cont_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(cont_pcd.points))])

    # make a point cloud of points that are not contour points (better for separating the colors)
    dists = np.asarray(sampled_pcd.compute_point_cloud_distance(cont_pcd))
    indices = np.where(dists > 0.00001)[0]
    road_without_contours_pcd = sampled_pcd.select_by_index(indices)

    return cont_pcd, road_without_contours_pcd


def points_in_radius(query_point, all_points, radius, tree):
    """
    finds all the points in the radius of the query point
    Parameters
    ----------
    query_point
    all_points
    radius
    tree

    Returns
    -------
    the nearest points in a list
    """
    # Find the nearest neighbors to the query point
    indices = tree.query_ball_point(query_point, r=radius)
    # Get the coordinates of the nearest neighbors, skip the point itself (always the first)
    nearest_neighbors = all_points[indices]
    return nearest_neighbors


def unit_vector(vector):
    """
    normalizes a given vector to a length of one
    :param vector: any vector
    :return: its unit vector
    """
    if len(vector) == 0:
        return 0
    return vector / np.linalg.norm(vector)


def angle_between_vectors(vector1, vector2):
    v1, v2 = unit_vector(vector1[:2]), unit_vector(vector2[:2])
    if np.array_equal(v1, v2):
        return 0
    determinant = np.linalg.det([v1, v2])
    dot_product = np.dot(v1, v2)
    angle = np.math.atan2(determinant, dot_product)
    angle = np.degrees(angle)
    if angle < 0:
        angle = 360 + angle
    return angle


# %%
@ray.remote
def is_cont_candidate(query_point, all_points, radius, tree, angle_thresh=135):
    # get the k nearest neighbors
    nearest_neighbors = points_in_radius(query_point, all_points, radius, tree)

    # Find the index of the query_point in the nearest neighbors
    # Create a boolean mask that indicates which elements to keep
    mask = ~np.all(nearest_neighbors == query_point, axis=1)
    # Use the boolean mask to create a new array without the element
    nearest_neighbors = nearest_neighbors[mask]

    length_nn = len(nearest_neighbors)
    if length_nn == 1:
        return query_point
    elif length_nn == 0:
        return False

    # get the vectors from query point to the neighbors
    vectors = [[nn[0] - query_point[0], nn[1] - query_point[1], nn[2] - query_point[2]] for nn in nearest_neighbors]

    # check all possible pairs one after the other
    vector = vectors[0]
    # copy of vectors, take out vectors that were already looked at
    tmp_vectors = vectors.copy()
    smallest_angles = []
    while len(tmp_vectors) > 1:
        # remove this element from temporary list
        tmp_vectors.remove(vector)
        # get the smallest angle between this vector and any other vector
        angles = [angle_between_vectors(vector, tmp_vec) for tmp_vec in tmp_vectors]

        positive_angles = [angle for angle in angles if angle > 0]
        smallest_clockwise_angle = min(positive_angles)
        smallest_angles.append(smallest_clockwise_angle)

        # change working vector to the one with the smallest angle
        vector = tmp_vectors[angles.index(smallest_clockwise_angle)]
        print()
    # compare last to first
    vector = tmp_vectors[0]
    first_vector = vectors[0]
    smallest_angles.append(angle_between_vectors(vector, first_vector))

    # Iterate over the angles
    for angle in smallest_angles:
        # Check if the current angle is greater than 90 degrees
        if angle > angle_thresh:
            return query_point
    return False


def dist(p1, p2):
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p1, p2)))


def sort_points_dist(poly_points):
    """
    sorts the contour points to have adjacent points following each other in the list
    Parameters
    ----------
    poly_points

    Returns
    -------
    the sorted list of points
    """
    # Convert the list of points to a numpy array
    poly_points = np.array(poly_points)

    # start at point 0 and check for the shortest connection
    p1 = poly_points[0]
    tmp_points = poly_points.copy()[1:]
    sorted_points = []
    list_min_dists = []

    while len(tmp_points) > 0:
        sorted_points.append(p1)

        # get the point that is closest to p1
        first_p = tmp_points[0]
        min_dist = dist(p1, first_p)
        min_pt = first_p
        for p2 in tmp_points[1:]:
            tmp_dist = dist(p1, p2)
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                min_pt = p2

        # check if the distance is not twice larger than the average (only in the end)
        if len(tmp_points) < 50:
            if min_dist > 5 * statistics.mean(list_min_dists):
                print(min_dist, statistics.mean(list_min_dists))
                break

        # check if the distance to the starting point is shorter
        # if yes stop the while loop
        # > 0 to not stop after first iteration
        # < min_dist/3 to avoid points that were left out and would come at the end
        dist_to_start = dist(p1, poly_points[0])
        if 0 < dist_to_start < min_dist / 3 and len(poly_points) // 2 > len(tmp_points):
            break

        # if not continue with the next point
        else:
            p1 = min_pt
            # remove p1 from tmp_points
            mask = ~np.all(tmp_points == p1, axis=1)
            tmp_points = tmp_points[mask]

        list_min_dists.append(min_dist)

    return sorted_points


def line_set_from_poly(poly):
    # Define the points of the polygon
    points = np.array(poly)

    # Define the edges of the polygon
    lines = [[i, (i + 1) % len(points)] for i in range(len(points))]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set


def get_polygon_lineset(cont_voxels):
    # sort the polygon points, make a polygone using the points, and another 4m above
    sorted_contour_poly = sort_points_dist(cont_voxels)
    polygon_clearance_height = copy.deepcopy(sorted_contour_poly)
    for i in range(len(polygon_clearance_height)):
        polygon_clearance_height[i][-1] += 4

    # create line sets for both polygons
    ls_road = line_set_from_poly(sorted_contour_poly)
    ls_cl_h = line_set_from_poly(polygon_clearance_height)

    ls_road.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(ls_road.lines))])
    ls_cl_h.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(ls_cl_h.lines))])

    return sorted_contour_poly, ls_road, ls_cl_h
