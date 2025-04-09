import pandaset
import numpy as np


def load_pandaset_sequence(pandaset_path, sequence_number):
    """
    loads one sequence (including multiple frames = recording points) from the pandaset data set
    Parameters
    ----------
    pandaset_path: path to the pandaset folder
    sequence_number: number of the sequence to load

    Returns
    -------
    the loaded sequence
    """
    dataset = pandaset.DataSet(pandaset_path)
    sequence = dataset[sequence_number]
    sequence.load()

    return sequence


def get_points_of_one_frame(sequence, frame_number):
    """
    gets the points of both Lidar sensors for a frame of the sequence
    Parameters
    ----------
    sequence: the loaded pandaset sequence
    frame_number: the frame number of interest (one sequence includes multiple frames)

    Returns
    -------
    the x,y,z coordinates of the points belonging to this frame
    """
    frame_points = sequence.lidar[frame_number].to_numpy()
    labels = sequence.semseg[frame_number]
    labels_list = labels['class'].tolist()
    return frame_points, labels_list


def get_points_pandaset(start, stop, step, seq):
    scan_list = []
    for i in range(start, stop, step):
        scan_list.append(seq.lidar[i].to_numpy())

    return scan_list


def concatenate_frames_pandaset(list_of_scans, list_of_preds):
    concatenated_frames_data = []

    # Iterate over each pair of prediction and scan data
    for preds, scan in zip(list_of_preds, list_of_scans):
        np_preds = np.array([[x] for x in preds])
        arr_with_preds = np.append(scan[:, :3], np_preds, axis=1)
        concatenated_frames_data.append(arr_with_preds)

    # Convert lists to numpy arrays
    concatenated_frames_data = np.concatenate(concatenated_frames_data, axis=0)
    return concatenated_frames_data