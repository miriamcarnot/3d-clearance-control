from main import *
from evaluation.get_lidarseg_annotations import *
from evaluation.util.metrics import *

import geopandas as gpd

import argparse
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Evaluation of the Road Shape')
    parser.add_argument('-d', '--dataset', help='choose a dataset, pandaset, semantickitti or nuscenes',
                        default="pandaset")
    parser.add_argument('-p', '--path', help='path to the dataset', default='/pandaset')
    parser.add_argument('-s', '--sequence', help='dataset sequence to run, check readme for options', default="001")
    parser.add_argument('-m', '--seg_model', help='segmentation model, sphere or randlanet',
                        default="sphere")
    parser.add_argument('--start', help='index of first scan',
                        default="0")
    parser.add_argument('--stop', help='index of last scan',
                        default="38")
    args = parser.parse_args()
    return args


def get_label_polygon(dataset, seq_name):
    labels_file_path = "evaluation/manual_annotations/"
    test_file_name = os.path.join(labels_file_path, dataset + '_' + seq_name + '_label.shp')
    shapefile = gpd.read_file(test_file_name)

    gdf = shapefile.set_geometry('geometry')

    # Extract coordinates
    coordinates = gdf.geometry.apply(
        lambda geom: list(geom.exterior.coords) if geom.geom_type == 'Polygon' else list(geom.coords))
    coordinates = coordinates.tolist()[0]

    points = np.array(coordinates)
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


def get_semseg_annotations(dataset, dataset_path, seq_name, start, stop):
    # get all points with the corresponding semseg annotation depending the dataset
    if dataset == 'pandaset':
        concatenated_frames_data = get_annos_pandaset(dataset_path, seq_name, start, stop)
        road_label_index, vegetation_label_index = 7, 5
    elif dataset == 'semantickitti':
        concatenated_frames_data = get_annos_semantickitti(dataset_path, seq_name, start, stop)
        road_label_index, vegetation_label_index = 40, 70
    else:
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_path, verbose=True)
        concatenated_frames_data = get_annos_nuscenes(nusc, seq_name, start, stop)
        road_label_index, vegetation_label_index = 24, 30

    # get road and vegetation points
    road_data, vege_data = filter_road_and_vege_data(concatenated_frames_data, road_label_index,
                                                     vegetation_label_index)

    return road_data, vege_data, concatenated_frames_data


def log_info(seg_model, road_shape_algorithm, version):
    logging.info('\n================================================')
    logging.info("V (" + str(version) + ") -> Model: " + seg_model + ", Shape Algorithm: " + road_shape_algorithm)
    logging.info('================================================')
    print("V (" + str(version) + ") -> Model: " + seg_model + ", Shape Algorithm: " + road_shape_algorithm)


def main():
    args = parse_arguments()
    dataset = args.dataset
    dataset_path = args.path
    seq_name = args.sequence
    seg_model = args.seg_model
    start = int(args.start)
    stop = int(args.stop)
    step = 1

    print("\n================\nRun experiment for " + str(dataset) + " scene " + str(seq_name) + "\n================\n")

    logging.basicConfig(filename="./evaluation/evaluation_logs/" + dataset + "_" + seq_name + "_" + seg_model + ".log",
                        level=logging.INFO, format='%(message)s')
    # logging.info('\nRoad sampling with 500, vegetation sampling with 10000\n')
    now = datetime.now()
    logging.info(now.strftime("%d/%m/%Y %H:%M:%S"))

    label_polygon = get_label_polygon(dataset, seq_name)

    print("Segmenting with ", seg_model)
    road_data, vege_data, concatenated_frames_data, _ , _, _ = make_segmentation(dataset, dataset_path, start, stop,
                                                                          step, seq_name, seg_model)

    # ==========================================
    # 1) Segmentation model + road shape label
    # ==========================================
    version = 1
    log_info(seg_model, "labeled polygon", version)
    road_data_annos, vege_data_annos, concatenated_frames_data_annos = get_semseg_annotations(dataset, dataset_path, seq_name, start, stop)

    inliers, outliers = crop_vege_pcd(vege_data, label_polygon.points)
    inliers_label, outliers_label = crop_vege_pcd(vege_data_annos, label_polygon.points)

    # road_annos_pcd, vege_annos_pcd = get_road_and_vege_pcd(road_data_annos, vege_data_annos)
    # road_pcd, vege_pcd = get_road_and_vege_pcd(road_data, vege_data)
    # inliers_label.colors = o3d.utility.Vector3dVector([[0, 0, 0.5] for _ in range(len(inliers_label.points))])
    # o3d.visualization.draw_geometries([road_pcd, label_polygon, inliers, inliers_label])

    calculate_performance(inliers, outliers, inliers_label, outliers_label)

    # ==================================================
    # 2) + 3) Segmentation model + road shape algorithms
    # ==================================================
    road_pcd, vege_pcd = get_road_and_vege_pcd(road_data, vege_data)
    # sample the road pcd
    sampled_pcd, sampling_geometries = sample_points(pcd=road_pcd, nb_spl_points=1000)
    # sample the vege data for equally distributed points (like ROI's)
    print("Number of vege points: ", len(vege_data))
    sampled_vege_pcd, sampling_geometries = sample_points(pcd=vege_pcd, nb_spl_points=10000)
    vege_data = np.asarray(sampled_vege_pcd.points)

    # for road_shape_algorithm in ['contour', 'mesh']:
    for road_shape_algorithm in ['mesh']:
        print("Finding Contour with  ", road_shape_algorithm)
        version = 3 if road_shape_algorithm == 'mesh' else 2
        log_info(seg_model, road_shape_algorithm, version)

        vege_data, ls_road, road_polygon, multiple_geometries_list = get_road_shape_polygon(
            sampled_pcd, vege_data, road_shape_algorithm)

        # crop the pcd with the detected polygon and the labeled polygon
        inliers, outliers = crop_vege_pcd(vege_data, road_polygon)
        inliers_label, outliers_label = crop_vege_pcd(vege_data, label_polygon.points)

        # points = np.array(road_polygon)
        # lines = [[i, i + 1] for i in range(len(points) - 1)]
        # road_polygon_ls = o3d.geometry.LineSet()
        # road_polygon_ls.points = o3d.utility.Vector3dVector(points)
        # road_polygon_ls.lines = o3d.utility.Vector2iVector(lines)
        # colors = [[1, 0, 0] for _ in range(len(lines))]  # Red lines
        # road_polygon_ls.colors = o3d.utility.Vector3dVector(colors)
        #
        # o3d.visualization.draw_geometries([road_pcd, inliers, outliers, road_polygon_ls])

        calculate_performance(inliers, outliers, inliers_label, outliers_label)
        check_polygon_coverage(road_polygon, label_polygon.points)

    if seg_model == 'sphere':
        # =================================================
        # 4) + 5) Segmentation label + road shape algorithm
        # =================================================
        road_data, vege_data, concatenated_frames_data = get_semseg_annotations(dataset, dataset_path, seq_name, start,
                                                                                stop)
        road_pcd, vegetation_pcd = get_road_and_vege_pcd(road_data, vege_data)
        sampled_pcd, _ = sample_points(pcd=road_pcd, nb_spl_points=1000)
        sampled_vege_pcd, sampling_geometries = sample_points(pcd=vege_pcd, nb_spl_points=10000)
        vege_data = np.asarray(sampled_vege_pcd.points)

        for road_shape_algorithm in ['contour', 'mesh']:
            print("Finding Contour with  ", road_shape_algorithm)
            version = 5 if road_shape_algorithm == 'mesh' else 4
            log_info('label', road_shape_algorithm, version)

            # road shape algorithm
            vege_data, ls_road, road_polygon, multiple_geometries_list = get_road_shape_polygon(
                sampled_pcd, vege_data, road_shape_algorithm)

            # crop the pcd with the detected polygon and the labeled polygon
            inliers, outliers = crop_vege_pcd(vege_data, road_polygon)
            inliers_label, outliers_label = crop_vege_pcd(vege_data, label_polygon.points)

            calculate_performance(inliers, outliers, inliers_label, outliers_label)
            check_polygon_coverage(road_polygon, label_polygon.points)


if __name__ == "__main__":
    main()
