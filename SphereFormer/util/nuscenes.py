import os
import random
import numpy as np
import torch
import yaml
import pickle
import glob
from pathlib import Path
from os.path import join, exists
from SphereFormer.util.data_util import data_prepare
from nuscenes.utils.data_classes import LidarPointCloud


class nuScenes(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 voxel_size=[0.1, 0.1, 0.1],
                 scene=0,
                 return_ref=True,
                 label_mapping="dataset/nuscenes.yaml",
                 rotate_aug=True,
                 flip_aug=True,
                 scale_aug=True,
                 transform_aug=True,
                 trans_std=[0.1, 0.1, 0.1],
                 ignore_label=255,
                 voxel_max=None,
                 xyz_norm=False,
                 pc_range=None,
                 use_tta=None,
                 vote_num=4,
                 nusc=None
                 ):
        super().__init__()
        self.scene = scene
        self.nusc = nusc
        self.return_ref = return_ref
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.class_names = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                            'traffic_cone',
                            'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                            'vegetation']
        self.use_tta = use_tta
        self.vote_num = vote_num

        with open("SphereFormer/util/nuscenes.yaml", 'r') as stream:
            self.nuscenes_dict = yaml.safe_load(stream)

        # sweeps_path = os.path.join(self.data_path, 'sweeps/LIDAR_TOP')
        # self.files = os.listdir(sweeps_path)

        self.sweeps = []
        my_scene = nusc.scene[self.scene]
        sweep = nusc.get('sample', my_scene['first_sample_token'])
        while sweep['next'] != '':
            self.sweeps.append(sweep)
            sweep = nusc.get('sample', sweep['next'])

        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)

        self.voxel_size = voxel_size

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sweeps)

    def __getitem__(self, index):
        if self.use_tta:
            samples = []
            for i in range(self.vote_num):
                sample = tuple(self.get_single_sample(index, vote_idx=i))
                samples.append(sample)
            return tuple(samples)
        return self.get_single_sample(index)

    def get_nusc_sweep_object(self, index):
        return self.sweeps[index]

    def remove_close(self, points, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]
        return points

    def get_single_sample(self, index, vote_idx=0):
        # file_name = self.files[index]
        # lidar_path = os.path.join(self.data_path, 'sweeps/LIDAR_TOP', file_name)
        # points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        sample = self.sweeps[index]
        sd_token = sample['data']['LIDAR_TOP']
        sd_rec = self.nusc.get('sample_data', sd_token)
        pcl = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, sd_rec['filename']))
        points = self.remove_close(pcl.points, 1.0)
        points = points.transpose()[:, :4]

        org_points = points[:, :3]

        labels_in = np.zeros(points.shape[0]).astype(np.uint8)

        # Augmentation
        # ==================================================
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.use_tta:
                flip_type = vote_idx % 4
            else:
                flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]

        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            points[:, 0:3] += noise_translate
        # ==================================================

        if self.return_ref:
            feats = points[:, :4]
        else:
            feats = points[:, :3]
        xyz = points[:, :3]

        if self.pc_range is not None:
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1])

        coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.voxel_size,
                                                                    self.voxel_max, None, self.xyz_norm)

        return [coords, xyz, feats, labels, inds_reconstruct], org_points
