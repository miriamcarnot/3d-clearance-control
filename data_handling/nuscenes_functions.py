from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from functools import reduce
from data_handling.util import *

import os
import open3d as o3d


def get_points_nuscenes(nusc, sequence, start=0, stop=40, step=5):
    scene = nusc.scene[int(sequence)]

    scan_list, sd_rec_list, labels_list = [], [], []

    # Get the first sample token in the scene
    first_sample_token = scene['first_sample_token']

    # Traverse through the samples in the scene
    sample_token = first_sample_token
    for _ in range(start, stop):
        sample = nusc.get('sample', sample_token)
        sd_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sd_token)

        # Load the point cloud data
        pcd = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
        scan = pcd.points.transpose()[:, :3]

        scan_list.append(scan)
        sd_rec_list.append(sd_rec)

        # Move to the next sample in the scene
        sample_token = sample['next']
        if sample_token == '':
            break  # Reached the end of the scene

    return scan_list, sd_rec_list


def concatenate_sweeps_nuscenes(nusc, points_list, sd_rec_list, preds_list):

    # calculate only for the first sweep (reference)
    ref_sd_rec = sd_rec_list[0]
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    concatenated_scans = []
    first = True
    for points, sd_rec, preds in zip(points_list, sd_rec_list, preds_list):
        # calculate distance to origin for each point, close points will be removed later
        distances_to_origin = dist_to_origin(points)

        if first:
            first = False
        else:
            current_pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
            current_cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            threshold = 0.02
            source_temp = o3d.geometry.PointCloud()
            source_temp.points = o3d.utility.Vector3dVector(points_list[0][:, :3])
            target_temp = o3d.geometry.PointCloud()
            source_temp.points = o3d.utility.Vector3dVector(points[:, :3])

            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_temp, target_temp, threshold, trans_matrix,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            trans_matrix = reg_p2p.transformation

            stacked_points = np.append(points[:, :3], np.ones((len(points), 1)), axis=1)
            points = trans_matrix.dot(stacked_points.transpose()).transpose()

        # append preds
        np_preds = np.array([[x] for x in preds])
        arr_with_preds = np.append(points[:, :3], np_preds, axis=1)
        # remove points too close to origin (cannot be before, because len of preds won't match)
        arr_with_preds = remove_close(arr_with_preds, distances_to_origin, radius=3.0)
        concatenated_scans.append(arr_with_preds)

    concatenated_scans = np.concatenate(concatenated_scans, axis=0)

    return concatenated_scans
