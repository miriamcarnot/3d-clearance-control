import random
import torch
import numpy as np
from torch_scatter import scatter_mean
from SphereFormer.util.voxelize import voxelize


def collate_fn_limit(batch, max_batch_points, logger):
    coord, xyz, feat, label = list(zip(*batch))
    offset, count = [], 0

    new_coord, new_xyz, new_feat, new_label = [], [], [], []
    k = 0
    for i, item in enumerate(xyz):

        count += item.shape[0]
        if count > max_batch_points:
            break

        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(label[i])

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in xyz])
        s_now = sum([x.shape[0] for x in new_xyz[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(
        new_label[:k]), torch.IntTensor(offset[:k])


def collation_fn_voxelmean(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, xyz, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)

    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)

    return coords, xyz, feats, labels, offset, inds_recons


def collation_fn_voxelmean_tta(batch_list):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    samples = []
    batch_list = list(zip(*batch_list))

    for batch in batch_list:
        coords, xyz, feats, labels, inds_recons = list(zip(*batch))
        inds_recons = list(inds_recons)

        accmulate_points_num = 0
        offset = []
        for i in range(len(coords)):
            inds_recons[i] = accmulate_points_num + inds_recons[i]
            accmulate_points_num += coords[i].shape[0]
            offset.append(accmulate_points_num)

        coords = torch.cat(coords)
        xyz = torch.cat(xyz)
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        offset = torch.IntTensor(offset)
        inds_recons = torch.cat(inds_recons)

        sample = (coords, xyz, feats, labels, offset, inds_recons)
        samples.append(sample)

    return samples


def data_prepare(coord, feat, label, voxel_size=np.array([0.1, 0.1, 0.1]), voxel_max=None, transform=None,
                 xyz_norm=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord_norm = coord - coord_min

    idx_recon = voxelize(coord_norm, voxel_size, mode=1)
    # print('icx coord before', idx_recon)
    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min

    coord = torch.FloatTensor(coord)

    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)

    coord_norm = torch.FloatTensor(coord_norm)
    idx_recon = torch.LongTensor(idx_recon)
    coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
    coords_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
    # print('beforecoord', len(coord))
    coord = scatter_mean(coord, idx_recon, dim=0)
    # print('after coord', len(coords_voxel))
    # print('labe', len(label))
    # print('idx_recon', len(idx_recon))
    feat = scatter_mean(feat, idx_recon, dim=0)
    return coords_voxel, coord, feat, label, idx_recon
