# functions parse_calibration() and parse_poses() are taken from the semantic kitti api git
# (https://github.com/PRBonn/semantic-kitti-api)

from data_handling.util import *

import os
import open3d as o3d
from numpy.linalg import inv


def parse_calibration(filename):
    """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def concatenate_scans_kitti(scans, poses_list, preds_list):
    concatenated_scans = []
    pose_1 = poses_list[0]
    scan_1 = scans[0]
    first = True

    for cur_scan, cur_pose, pred in zip(scans, poses_list, preds_list):
        num_rows = cur_scan.shape[0]
        ones_column = np.ones((num_rows, 1))
        points = np.concatenate((cur_scan[:, :3], ones_column), axis=1)

        # calculate distance to origin for each point, close points will be removed later
        distances_to_origin = dist_to_origin(points)

        if first:
            transformed_points = points
            first = False
        else:

            threshold = 0.02
            source_temp = o3d.geometry.PointCloud()
            source_temp.points = o3d.utility.Vector3dVector(cur_scan[:, :3])

            target_temp = o3d.geometry.PointCloud()
            target_temp.points = o3d.utility.Vector3dVector(scan_1[:, :3])

            # Calculate the initial transformation matrix based on the poses
            initial_transformation_matrix = np.dot(np.linalg.inv(pose_1), cur_pose)

            # Perform ICP using the initial transformation matrix
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_temp, target_temp, threshold, initial_transformation_matrix,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            # print(reg_p2p)

            final_transformation_matrix = reg_p2p.transformation
            transformed_points = np.dot(initial_transformation_matrix, points.T).T

        np_preds = np.array([[x] for x in pred])
        arr_with_preds = np.append(transformed_points[:, :3], np_preds, axis=1)
        # remove points too close to origin (cannot be before, because len of preds wont match)
        arr_with_preds = remove_close(arr_with_preds, distances_to_origin, radius=3.0)

        concatenated_scans.append(arr_with_preds)

    concatenated_scans = np.concatenate(concatenated_scans, axis=0)
    return concatenated_scans

# def concatenate_scans_kitti(scans, poses_list, preds_list):
#     concatenated_scans = []
#     pose_1 = poses_list[0]
#     first = True
#
#     for cur_scan, cur_pose, pred in zip(scans, poses_list, preds_list):
#         num_rows = cur_scan.shape[0]
#         ones_column = np.ones((num_rows, 1))
#         points = np.concatenate((cur_scan[:, :3], ones_column), axis=1)
#
#         if first:
#             transformed_points = points
#             first = False
#         else:
#             transformed_points = np.dot(inv(pose_1), np.dot(cur_pose, points.T)).T
#
#         np_preds = np.array([[x] for x in pred])
#         arr_with_preds = np.append(transformed_points[:, :3], np_preds, axis=1)
#
#         concatenated_scans.append(arr_with_preds)
#
#     concatenated_scans = np.concatenate(concatenated_scans, axis=0)
#     return concatenated_scans


def get_points_kitti(dataset_path, sequence_name: str, start, stop, step):
    seq_path = os.path.join(dataset_path, sequence_name.zfill(2))
    frames = sorted(os.listdir(os.path.join(seq_path, 'velodyne')), key=lambda x: int(x.split('.')[0]))

    poses_path = os.path.join(seq_path, "poses.txt")
    calib_path = os.path.join(seq_path, "calib.txt")

    calibration = parse_calibration(calib_path)
    poses = parse_poses(poses_path, calibration)
    scan_list, poses_list = [], []

    for i in range(start, stop, step):
        pcd_file_name = frames[i]
        scan_dir = os.path.join(seq_path, 'velodyne', pcd_file_name)
        scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
        scan_list.append(scan)
        poses_list.append(poses[i])

    return scan_list, poses_list


def get_poses_kitti(data_path, sequence):
    seq_path = os.path.join(data_path, 'sequences', str(sequence).zfill(2))

    poses_path = os.path.join(seq_path, "poses.txt")
    calib_path = os.path.join(seq_path, "calib.txt")

    calibration = parse_calibration(calib_path)
    poses = parse_poses(poses_path, calibration)
    return poses
