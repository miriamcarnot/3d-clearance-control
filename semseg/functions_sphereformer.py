import torch
import yaml

from data_handling.kitti_functions import *

import spconv.pytorch as spconv

from SphereFormer.model.unet_spherical_transformer import Semantic as Model
from SphereFormer.util.nuscenes import nuScenes
from SphereFormer.util.semantic_kitti import SemanticKITTI
from SphereFormer.util.pandaset import PandaSet


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


def load_config(config_path):
    # load config information
    with open(config_path, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)
    return config_data


def load_model(config_data, weights_path):
    patch_size = config_data['TRAIN']["patch_size"]
    window_size = config_data['TRAIN']["window_size"]
    voxel_size = config_data['DATA']["voxel_size"]
    window_size_sphere = config_data['TRAIN']["window_size_sphere"]
    input_c = config_data['TRAIN']["input_c"]
    m = config_data['TRAIN']["m"]
    classes = config_data['DATA']["classes"]
    block_reps = config_data['TRAIN']["block_reps"]
    block_residual = config_data['TRAIN']["block_residual"]
    layers = config_data['TRAIN']["layers"]
    quant_size_scale = config_data['TRAIN']["quant_size_scale"]
    rel_query = config_data['TRAIN']["rel_query"]
    rel_key = config_data['TRAIN']["rel_key"]
    rel_value = config_data['TRAIN']["rel_value"]
    drop_path_rate = config_data['TRAIN']["drop_path_rate"]
    window_size_scale = config_data['TRAIN']["window_size_scale"]
    grad_checkpoint_layers = config_data['TRAIN']["grad_checkpoint_layers"]
    sphere_layers = config_data['TRAIN']["sphere_layers"]
    a = config_data['TRAIN']["a"]

    patch_size = np.array([voxel_size[i] * patch_size for i in range(3)]).astype(np.float32)
    window_size = patch_size * window_size
    window_size_sphere = np.array(window_size_sphere)

    # load model architecture
    model = Model(input_c=input_c,
                  m=m,
                  classes=classes,
                  block_reps=block_reps,
                  block_residual=block_residual,
                  layers=layers,  #
                  window_size=window_size,
                  window_size_sphere=window_size_sphere,
                  quant_size=window_size / quant_size_scale,
                  quant_size_sphere=window_size_sphere / quant_size_scale,
                  rel_query=rel_query,
                  rel_key=rel_key,
                  rel_value=rel_value,
                  drop_path_rate=drop_path_rate,
                  window_size_scale=window_size_scale,
                  grad_checkpoint_layers=grad_checkpoint_layers,
                  sphere_layers=sphere_layers,
                  a=a
                  )

    # load the weights
    model = torch.nn.DataParallel(model.cuda())
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


def load_data(dataset, data_path, sequence, config_data, nuscenes_dataset_object=None):
    voxel_size = config_data['DATA']["voxel_size"]
    use_tta = config_data['TRAIN']["use_tta"]
    xyz_norm = config_data['TRAIN']["xyz_norm"]
    vote_num = config_data['TRAIN']["vote_num"]
    pc_range = config_data['TRAIN']["pc_range"]

    if dataset == "semantickitti":
        test_data = SemanticKITTI(data_path=data_path,
                                  voxel_size=voxel_size,
                                  seq=sequence,
                                  rotate_aug=use_tta,
                                  flip_aug=use_tta,
                                  scale_aug=use_tta,
                                  transform_aug=use_tta,
                                  xyz_norm=xyz_norm,
                                  pc_range=pc_range,
                                  use_tta=use_tta,
                                  vote_num=vote_num,
                                  )
    elif dataset == "nuscenes":
        test_data = nuScenes(data_path=data_path,
                             voxel_size=voxel_size,
                             scene=int(sequence),
                             rotate_aug=use_tta,
                             flip_aug=use_tta,
                             scale_aug=use_tta,
                             transform_aug=use_tta,
                             xyz_norm=xyz_norm,
                             pc_range=None,
                             use_tta=use_tta,
                             vote_num=vote_num,
                             nusc=nuscenes_dataset_object
                             )
    else:  # pandaset
        test_data = PandaSet(data_path=data_path,
                                  voxel_size=voxel_size,
                                  seq=sequence,
                                  rotate_aug=use_tta,
                                  flip_aug=use_tta,
                                  scale_aug=use_tta,
                                  transform_aug=use_tta,
                                  xyz_norm=xyz_norm,
                                  pc_range=pc_range,
                                  use_tta=use_tta,
                                  vote_num=vote_num,
                                  )

    print("Scans in Sequence: ", test_data.__len__(), "\n")
    return test_data


def get_torch_data_loader(test_data, batch_size):
    workers = 1
    data_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=True,
                                              sampler=None,
                                              collate_fn=collation_fn_voxelmean
                                              )
    return data_loader


def run_inference(model, data_loader, batch_size, inds_reconstruct):
    for i, batch_data in enumerate(data_loader):
        (coord, xyz, feat, target, offset, inds_reverse) = batch_data

        # inds_reverse = inds_reverse.cuda(non_blocking=True)
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)

        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(
            non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, batch_size)

        with torch.no_grad():
            output = model(sinput, xyz, batch)
    # restore the original array
    inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)
    output = output[inds_reconstruct, :]

    preds = torch.argmax(output, 1)
    preds_np = np.asarray(preds.cpu())
    xyz_np = np.asarray(xyz.cpu())

    torch.cuda.empty_cache()

    return preds_np, xyz_np


def sphere_kitti(dataset, data_path, config_path, weights_path, start, stop, step, sequence):
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    config_data = load_config(config_path)
    model = load_model(config_data, weights_path)
    batch_size = 1
    test_data = load_data(dataset, data_path, sequence, config_data)

    scan_list, poses_list, preds_list = [], [], []
    all_seq_poses = get_poses_kitti(data_path, sequence)

    for i in range(start, stop, step):
        single_sample, org = test_data.get_single_sample(i)
        idx_recon = single_sample[4]
        single_sample = [single_sample]
        data_loader = get_torch_data_loader(single_sample, batch_size)
        preds_np, xyz_np_not_needed = run_inference(model, data_loader, batch_size, idx_recon)

        np_preds = np.array([[x] for x in preds_np])
        arr_with_preds = np.append(org[:, :3], np_preds, axis=1)
        arr_with_preds = arr_with_preds[arr_with_preds[:, 0] <= 50, :]
        arr_with_preds = arr_with_preds[arr_with_preds[:, 1] <= 50, :]
        arr_with_preds = arr_with_preds[arr_with_preds[:, 0] >= -50, :]
        arr_with_preds = arr_with_preds[arr_with_preds[:, 1] >= -50, :]

        preds = arr_with_preds[:, -1]
        xyz = arr_with_preds[:, :3]

        preds_list.append(preds)
        scan_list.append(xyz)
        poses_list.append(all_seq_poses[i])

    return scan_list, poses_list, preds_list


def sphere_nuscenes(dataset, data_path, config_path, weights_path, start, stop, step, nusc, sequence):
    # get the points, the predictions and the file names
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    config_data = load_config(config_path)
    model = load_model(config_data, weights_path)
    batch_size = 1
    test_data = load_data(dataset, data_path, sequence, config_data, nusc)

    scan_list, sd_rec_list, preds_list = [], [], []
    for i in range(start, stop, step):
        single_sample, org = test_data.get_single_sample(i)
        idx_recon = single_sample[4]
        single_sample = [single_sample]
        data_loader = get_torch_data_loader(single_sample, batch_size)
        preds_np, xyz_np = run_inference(model, data_loader, batch_size, idx_recon)

        preds_list.append(preds_np)
        scan_list.append(org)

        sample = test_data.get_nusc_sweep_object(i)
        sd_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sd_token)
        sd_rec_list.append(sd_rec)

    return scan_list, sd_rec_list, preds_list


def sphere_pandaset(dataset, data_path, config_path, weights_path, start, stop, step, sequence):
    # get the points, the predictions and the file names
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    config_data = load_config(config_path)
    model = load_model(config_data, weights_path)
    batch_size = 1
    test_data = load_data(dataset, data_path, sequence, config_data)

    scan_list, preds_list = [], []
    for i in range(start, stop, step):
        single_sample, org = test_data.get_single_sample(i)
        idx_recon = single_sample[4]
        single_sample = [single_sample]
        # print('single length before model', len(org))
        data_loader = get_torch_data_loader(single_sample, batch_size)
        preds_np, xyz_np = run_inference(model, data_loader, batch_size, idx_recon)

        preds_list.append(preds_np)
        scan_list.append(org)


    return scan_list, preds_list
