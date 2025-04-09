import os
import pickle
from os.path import join
from pathlib import Path
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import DATASET
from .utils import BEVBox3D
import open3d as o3d

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

log = logging.getLogger(__name__)


class NuScenes_SemSeg(BaseDataset):
    """This class is used to create a dataset based on the NuScenes 3D dataset,
    and used in object detection, visualizer, training, or testing.

    The NuScenes 3D dataset is best suited for autonomous driving applications.
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='NuScenes',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about the
                dataset. This is default to dataset path if nothing is provided.
            name: The name of the dataset (NuScenes in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.

        Returns:
            class: The corresponding class.
        """
        if info_path is None:
            info_path = dataset_path

        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 3
        self.label_to_names = self.get_label_to_names()

        if os.path.exists(join(info_path, 'nuscenes_seg_infos_1sweeps_train.pkl')):
            self.train_info = pickle.load(
                open(join(info_path, 'nuscenes_seg_infos_1sweeps_train.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'nuscenes_seg_infos_1sweeps_val.pkl')):
            self.val_info = pickle.load(
                open(join(info_path, 'nuscenes_seg_infos_1sweeps_val.pkl'), 'rb'))

        print("\nTrain and test files: ", len(self.train_info), len(self.val_info))
        # get splits
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.dataset_path, verbose=True)
        # scene_splits = create_splits_scenes()
        # self.test_split = scene_splits['test']
        # self.train_split = scene_splits['train']
        # self.val_split = scene_splits['val']

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {'1': 'Road', # before 7
                          '2': 'Vegetation', # before 5
                          '3': 'Other'}
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        path = path.replace('\\', '/')
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def read_label(path):
        path = path.replace('\\', '/')
        assert Path(path).exists()

        annotated_data =  np.fromfile(path, dtype=np.uint8, count=-1).reshape([-1])
        return annotated_data.astype(np.uint8)


    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return NuSceneSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
                split name should be one of 'training', 'test', 'validation', or
                'all'.
        """
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.val_info
        elif split in ['val', 'validation']:
            return self.val_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested():
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
                        attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
                attribute passed.
            attr: The attributes that correspond to the outputs passed in
                results.
        """
        pass


class NuSceneSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='train'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    # def __init__(self, dataset, split='train'):
    #     self.cfg = dataset.cfg
    #     self.info = dataset.get_split_list(split)
    #     self.path_list = []
    #     self.split = split
    #     self.dataset = dataset
    #
    # def __len__(self):
    #     return len(self.info)

    def get_data(self, idx):
        scene_info = self.path_list[idx]

        lidar_path = scene_info['lidar_path']
        lidarseg_label_path = scene_info['lidarseg_label_path']

        sweep = self.dataset.read_lidar(join(self.dataset.dataset_path, lidar_path))
        pc = sweep[:, :3]
        intensity = sweep[:, 3]
        label = self.dataset.read_label(join(self.dataset.dataset_path, lidarseg_label_path))

        # Define conditions
        condition_road = (label == 24)
        condition_vege = (label == 30)

        # Update values based on conditions
        label[condition_road] = 1
        label[condition_vege] = 2

        # Set the remaining values to 3
        label[~(condition_road | condition_vege)] = 3

        data = {
            'point': pc,
            'intensity': intensity,
            'label': label
        }

        return data

    def get_attr(self, idx):
        scene_info = self.path_list[idx]

        pc_path = scene_info['lidar_path']
        token = scene_info['token']

        attr = {'name': token, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(NuScenes_SemSeg)
