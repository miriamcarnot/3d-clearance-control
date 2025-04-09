import open3d as o3d
import numpy as np


def sample_points(pcd, nb_spl_points):
    """
    creates a point cloud from given data points, removes outliers, samples the points (less points, equally distributed)
    Parameters
    ----------
    nb_spl_points: number of points after the sampling
    data: the data points to be sampled (road points)

    Returns
    -------
    the sampled point cloud
    """
    sampling_geometries = [[pcd]]
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=4.0)  # before 20, 3 # 15,5
    sampling_geometries.append([pcd])
    gt_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=2.0)
    sampled_pcd = gt_mesh.sample_points_poisson_disk(nb_spl_points)
    sampled_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    sampled_pcd.estimate_normals()
    sampled_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(sampled_pcd.points))])
    sampling_geometries.append([sampled_pcd])
    return sampled_pcd, sampling_geometries


def crop_vege_pcd(veg_data, polygon):
    # make a point cloud only with vegetation
    vege_pcd = o3d.geometry.PointCloud()
    vege_pcd.points = o3d.utility.Vector3dVector(veg_data[:, :3])

    # Calculate the average z-value
    z_values = np.array([point[2] for point in polygon])
    average_z = np.mean(z_values)

    # Create SelectionPolygonVolume object
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_min = average_z
    vol.axis_max = average_z + 4
    vol.bounding_polygon = o3d.utility.Vector3dVector(polygon)

    # Crop point cloud
    vege_inliers_pcd = vol.crop_point_cloud(vege_pcd)

    # Get indices of points inside the volume
    dists = np.asarray(vege_pcd.compute_point_cloud_distance(vege_inliers_pcd))
    indices = np.where(dists > 0.00001)[0]
    vege_outliers_pcd = vege_pcd.select_by_index(indices)

    # set colors
    vege_inliers_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(vege_inliers_pcd.points))])
    vege_outliers_pcd.colors = o3d.utility.Vector3dVector(
        [[0.3, 0.47, 0.23] for _ in range(len(vege_outliers_pcd.points))])

    return vege_inliers_pcd, vege_outliers_pcd


def filter_road_and_vege_data(concatenated_frames_data, road_label_index, vegetation_label_index):
    road_rows = concatenated_frames_data[:, -1].astype(int) == road_label_index
    road_data = concatenated_frames_data[road_rows]
    vege_rows = concatenated_frames_data[:, -1].astype(int) == vegetation_label_index
    vege_data = concatenated_frames_data[vege_rows]
    return road_data, vege_data


def get_road_and_vege_pcd(road_data, vege_data):
    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(road_data[:, :3])
    road_pcd.colors = o3d.utility.Vector3dVector([[0.33, 0.33, 0.33] for _ in range(len(road_pcd.points))])

    vege_pcd = o3d.geometry.PointCloud()
    vege_pcd.points = o3d.utility.Vector3dVector(vege_data[:, :3])
    vege_pcd.colors = o3d.utility.Vector3dVector([[0, 0.5, 0] for _ in range(len(vege_pcd.points))])

    return road_pcd, vege_pcd


def process_color_map(COLOR_MAP):
    # Convert class colors to doubles from 0 to 1, as expected by the visualizer
    for label in COLOR_MAP:
        COLOR_MAP[label] = tuple(val / 255 for val in COLOR_MAP[label])
    return COLOR_MAP
