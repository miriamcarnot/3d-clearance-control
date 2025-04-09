from tqdm import tqdm
import pandas as pd

from data_handling.pandaset_functions import *
from data_handling.kitti_functions import *
from data_handling.nuscenes_functions import *


def get_annos_pandaset(path, sequence, start, stop):
    seq = load_pandaset_sequence(path, sequence)
    scans = [seq.lidar[i].to_numpy() for i in range(80)]
    segmentations = seq.semseg
    selected_data = []

    for i in tqdm(range(start, stop)):
        points = pd.DataFrame(scans[i][:, :3])
        frame_seg = segmentations[i]
        data_with_class = pd.concat([points, frame_seg], axis=1)
        selected_data.append(data_with_class)
        # print(selected_data[:3])

    concatenated_frames_data = pd.concat(selected_data)
    return np.asarray(concatenated_frames_data)


def read_kitti_label_file(label_file_path):
    # Read the label file as an array of 32-bit unsigned integers
    labels = np.fromfile(label_file_path, dtype=np.uint32)

    # Split the label into its semantic and instance parts if needed
    # Semantic KITTI uses the lower 16 bits for the semantic label
    # and the upper 16 bits for the instance id
    semantic_labels = labels & 0xFFFF  # Lower 16 bits
    instance_ids = labels >> 16  # Upper 16 bits

    return semantic_labels, instance_ids


def get_annos_semantickitti(path, sequence, start, stop):
    pcd_files_path = os.path.join(path, 'sequences', sequence, 'velodyne')
    all_files_in_sequence = os.listdir(pcd_files_path)
    all_files_in_sequence.sort()
    all_seq_poses = get_poses_kitti(path, sequence)

    scans_list, labels_list, poses_list = [], [], []

    for i in tqdm(range(start, stop)):
        pcd_file_name = all_files_in_sequence[i].split(".")[0]

        scan_dir = os.path.join(path, 'sequences', sequence, 'velodyne', pcd_file_name + '.bin')
        scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)

        label_file = os.path.join(path, 'sequences', sequence, 'labels', pcd_file_name + '.label')
        semantic_labels, _ = read_kitti_label_file(label_file)

        scans_list.append(scan)
        labels_list.append(semantic_labels)
        poses_list.append(all_seq_poses[i])

    concatenated_frames_data = concatenate_scans_kitti(scans_list, poses_list, labels_list)

    return concatenated_frames_data


def get_annos_nuscenes(nusc, sequence, start, stop):
    scene = nusc.scene[int(sequence)]

    scan_list, sd_rec_list, labels_list = [], [], []

    # Get the first sample token in the scene
    first_sample_token = scene['first_sample_token']

    # Traverse through the samples in the scene
    sample_token = first_sample_token
    for _ in tqdm(range(start, stop)):
        sample = nusc.get('sample', sample_token)
        sd_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sd_token)

        # Load the point cloud data
        pcd = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
        scan = pcd.points.transpose()[:, :3]

        # Load the segmentation labels
        record_lidarseg = nusc.get('lidarseg', sd_token)
        lidarseg_labels_filename = os.path.join(nusc.dataroot, record_lidarseg['filename'])
        label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

        # Append the data to the lists
        scan_list.append(scan)
        labels_list.append(label)
        sd_rec_list.append(sd_rec)

        # Move to the next sample in the scene
        sample_token = sample['next']
        if sample_token == '':
            break  # Reached the end of the scene

    concatenated_frames_data = concatenate_sweeps_nuscenes(nusc, scan_list, sd_rec_list, labels_list)

    return concatenated_frames_data
