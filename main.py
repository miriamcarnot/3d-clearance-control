from road_boundary_approx.contour_algorithm import *
from road_boundary_approx.mesh_algorithm import *
from open3d_visualization import *
from clear_height_functions import *

from data_handling.pandaset_functions import *
from data_handling.nuscenes_functions import *
from data_handling.kitti_functions import *

from parse_args import parse_arguments

import open3d as o3d

from nuscenes.nuscenes import NuScenes

from SphereFormer.colormaps.semantickitti_colormap import COLOR_MAP_KITTI
from SphereFormer.colormaps.nuscenes_colormap import COLOR_MAP_NUSCENES
from SphereFormer.colormaps.pandaset_colormap import COLOR_MAP_PANDASET
from open3dML.randlanet_colormap import COLOR_MAP_RANDLANET


def make_segmentation(dataset, dataset_path, start, stop, step, sequence_name, seg_model):
    if dataset == "semantickitti":
        sequence_name = sequence_name.zfill(2)
        color_map = COLOR_MAP_KITTI
        road_label_index, vegetation_label_index = 8, 14

        if seg_model == "sphere":
            from semseg.functions_sphereformer import sphere_kitti
            config_path = 'SphereFormer/config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml'
            weights_path = 'SphereFormer/weights/model_semantic_kitti.pth'

            # sequences ranging from 00 to 21
            scan_list, poses_list, preds_list = sphere_kitti(dataset, dataset_path, config_path, weights_path, start,
                                                             stop,
                                                             step, sequence_name)
        else:
            from semseg.functions_randlanet import run_inference
            dataset_path = os.path.join(dataset_path, "sequences")
            # sequences ranging from 00 to 21
            scan_list, poses_list = get_points_kitti(dataset_path, sequence_name, start=start, stop=stop, step=step)
            # get the predictions from the trained model (0: road, 1: vegetation, 2: others)
            preds_list = run_inference(dataset, dataset_path, scan_list)

        concatenated_frames_data = concatenate_scans_kitti(scan_list, poses_list, preds_list)

    elif dataset == "nuscenes":
        # nusc = NuScenes(version='v1.0-mini', dataroot=dataset_path, verbose=True)
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_path, verbose=True)
        if seg_model == "sphere":
            from semseg.functions_sphereformer import sphere_nuscenes
            config_path = 'SphereFormer/config/nuscenes/nuscenes_unet32_spherical_transformer.yaml'
            weights_path = 'SphereFormer/weights/model_nuscenes.pth'
            scan_list, sd_rec_list, preds_list = sphere_nuscenes(dataset, dataset_path, config_path, weights_path,
                                                                 start,
                                                                 stop, step, nusc, sequence_name)
            color_map = COLOR_MAP_NUSCENES
            road_label_index, vegetation_label_index = 10, 15

        else:
            from semseg.functions_randlanet import run_inference
            scan_list, sd_rec_list = get_points_nuscenes(nusc, sequence_name, start=start, stop=stop, step=step)
            preds_list = run_inference(dataset, dataset_path, scan_list)
            road_label_index, vegetation_label_index = 0, 1
            color_map = COLOR_MAP_RANDLANET
            # road_label_index, vegetation_label_index = 8, 14
            # color_map = COLOR_MAP_KITTI

        concatenated_frames_data = concatenate_sweeps_nuscenes(nusc, scan_list, sd_rec_list, preds_list)

    else:  # "pandaset"
        if seg_model == "sphere":
            from semseg.functions_sphereformer import sphere_pandaset
            config_path = 'SphereFormer/config/pandaset/pandaset_unet32_spherical_transformer.yaml'
            weights_path = 'SphereFormer/weights/model_pandaset.pth'
            # sequences ranging from 001 to 047 (some do not exist)
            scan_list, preds_list = sphere_pandaset(dataset, dataset_path, config_path, weights_path, start, stop,
                                                    step, sequence_name)
            print("after calling sphere_pandaset: ", len(scan_list[0]), len(preds_list[0]))
            color_map = COLOR_MAP_PANDASET
            road_label_index, vegetation_label_index = 2, 1
        else:
            from semseg.functions_randlanet import run_inference
            sequence = load_pandaset_sequence(dataset_path, sequence_name)
            scan_list = get_points_pandaset(start=start, stop=stop, step=step, seq=sequence)
            # get the predictions from the trained model (0: road, 1: vegetation, 2: others)
            preds_list = run_inference(dataset, dataset_path, scan_list)
            road_label_index, vegetation_label_index = 0, 1
            color_map = COLOR_MAP_RANDLANET

        concatenated_frames_data = concatenate_frames_pandaset(scan_list, preds_list)

    # Convert class colors to doubles from 0 to 1, as expected by the visualizer
    for label in color_map:
        color_map[label] = tuple(val / 255 for val in color_map[label])

    road_data, vege_data = filter_road_and_vege_data(concatenated_frames_data, road_label_index, vegetation_label_index)

    first_scan, first_pred = scan_list[0], preds_list[0]

    return road_data, vege_data, concatenated_frames_data, color_map, first_scan, first_pred
    # return road_data, vege_data, concatenated_frames_data, color_map


def get_road_shape_polygon(sampled_pcd, vege_data, road_shape_algorithm):
    if road_shape_algorithm == 'contour':
        cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd)
        road_polygon, ls_road, ls_cl_h = get_polygon_lineset(cont_voxels=cont_pcd.points)
        multiple_geometries_list = [[road_without_contours_pcd, cont_pcd, ls_road]]

    else:  # mesh
        radius = [2]
        sampled_pcd.orient_normals_consistent_tangent_plane(100)
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            sampled_pcd, o3d.utility.DoubleVector(radius))

        outer_edges = find_outer_edges(rec_mesh)
        initial_ls_road = create_lineset(rec_mesh, outer_edges)

        filtered_outer_edges, road_polygon = filter_edges(outer_edges, rec_mesh)
        ls_road = create_lineset(rec_mesh, filtered_outer_edges)
        multiple_geometries_list = [[rec_mesh, ls_road]] #, [initial_ls_road], [ls_road]]

    return vege_data, ls_road, road_polygon, multiple_geometries_list


def main(args):
    dataset = args.dataset
    dataset_path = args.path
    sequence_name = args.sequence
    start = int(args.start)
    step = int(args.step)
    stop = int(args.stop)
    seg_model = args.model

    print("\nStarting Analysis")
    print("Dataset: " + dataset)
    print("Sequence: " + sequence_name)

    ############################################################################
    # Read lidar scans and concatenate frames
    ############################################################################

    print("\nSegmenting point cloud with", seg_model, "model")

    road_data, vege_data, concatenated_frames_data, color_map, first_scan, first_pred = make_segmentation(dataset, dataset_path, start, stop,
                                                                                  step, sequence_name, seg_model)
    road_pcd, vege_pcd = get_road_and_vege_pcd(road_data, vege_data)

    first_scan_pcd = o3d.geometry.PointCloud()
    first_scan_pcd.points = o3d.utility.Vector3dVector(first_scan[:, :3])

    first_scan_semseg_pcd = o3d.geometry.PointCloud()
    first_scan_semseg_pcd.points = o3d.utility.Vector3dVector(first_scan[:, :3])
    colors = [color_map[pred] for pred in first_pred]
    first_scan_semseg_pcd.colors = o3d.utility.Vector3dVector(colors)

    concat_pcd = o3d.geometry.PointCloud()
    concat_pcd.points = o3d.utility.Vector3dVector(concatenated_frames_data[:, :3])

    colors = [color_map[clr] for clr in concatenated_frames_data[:, 3]]
    semseg_pcd = o3d.geometry.PointCloud()
    semseg_pcd.points = o3d.utility.Vector3dVector(concatenated_frames_data[:, :3])
    semseg_pcd.colors = o3d.utility.Vector3dVector(colors)

    ############################################################################
    # Get the boundary of the road and create volume to cut out vegetation points
    # depending on the chosen algorithm
    ############################################################################

    print("\nApproximating road boundaries with", args.algorithm, "algorithm")

    # sample street points, find contour points, and crop the vegetation point cloud
    sampled_pcd, sampling_geometries = sample_points(pcd=road_pcd, nb_spl_points=1000)

    vege_data, ls_road, road_polygon, multiple_geometries_list = get_road_shape_polygon(sampled_pcd, vege_data, args.algorithm)

    vege_inliers_pcd, vege_outliers_pcd = crop_vege_pcd(vege_data, road_polygon)
    multiple_geometries_list.append([road_pcd, ls_road])

    num_red_points = len(vege_inliers_pcd.points)

    print("\nNumber of Red Points found: ", num_red_points)

    ############################################################################
    # show the results and project the results onto images for pandaset
    ############################################################################
    if args.show == "True":
        if num_red_points > 0:
            multiple_geometries_list.append([road_pcd, vege_inliers_pcd, vege_outliers_pcd])
        all_geometries = [[first_scan_pcd]] + [[first_scan_semseg_pcd]] + [[semseg_pcd]] + [[road_pcd, vege_pcd]] + sampling_geometries + multiple_geometries_list
        # all_geometries = [[semseg_pcd] + [road_pcd, vege_pcd]] + sampling_geometries + multiple_geometries_list

        visualize_pcds_one_after_another(all_geometries)

    ############################################################################

    return vege_data, road_polygon, ls_road


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
