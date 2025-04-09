import logging
import numpy as np
# from shapely import hausdorff_distance
from shapely.geometry import Polygon


def log_results(precision, recall, f1, iou):
    # logging.info("precision: %.3f", precision)
    # logging.info("recall: %.3f", recall)
    # logging.info("F1 SCORE: %.3f", f1)
    logging.info("IoU: %.1f", iou)


def calculate_performance(inliers, outliers, inliers_label, outliers_label):
    pred_points = np.asarray(inliers.points)
    label_points = np.asarray(inliers_label.points)
    pred_points_out = np.asarray(outliers.points)
    label_points_out = np.asarray(outliers_label.points)

    # Convert lists of lists to sets of tuples
    pred_set = set(tuple(sublist) for sublist in pred_points)
    label_set = set(tuple(sublist) for sublist in label_points)
    pred_set_out = set(tuple(sublist) for sublist in pred_points_out)
    label_set_out = set(tuple(sublist) for sublist in label_points_out)

    print()

    # logging.info("Length of predicted and labeled points: ", len(pred_points), len(label_set))  # 148098 54143

    tp = len(label_set.intersection(pred_set))
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    tn = len(label_set_out.intersection(pred_set_out))

    logging.info("TP, FP, FN, TN : %s, %s, %s, %s", tp, fp, fn, tn)

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    iou = 100 * tp / (tp + fp + fn)

    log_results(precision, recall, f1, iou)


def check_polygon_coverage(polygon1_3d, polygon2_3d):
    """
    Calculate the percentage of polygon2 covered by polygon1.

    Args:
    - polygon1_coords: List of tuples representing the coordinates of the first polygon [(x1, y1), (x2, y2), ...].
    - polygon2_coords: List of tuples representing the coordinates of the second polygon [(x1, y1), (x2, y2), ...].

    Returns:
    - A tuple containing:
        - The percentage of polygon2 covered by polygon1.
        - The percentage of polygon2 not covered by polygon1.
    """

    def remove_duplicate_points(polygon_coords):
        unique_coords = []
        for coord in polygon_coords:
            if coord not in unique_coords:
                unique_coords.append(coord)
        return unique_coords

    polygon1_2d_coords = [(x, y) for x, y, z in polygon1_3d]
    polygon2_2d_coords = [(x, y) for x, y, z in np.asarray(polygon2_3d)]

    polygon1 = Polygon(remove_duplicate_points(polygon1_2d_coords)).buffer(0)
    polygon2 = Polygon(remove_duplicate_points(polygon2_2d_coords)).buffer(0)

    hausdorff_dist = polygon1.hausdorff_distance(polygon2)

    if not polygon1.is_valid or not polygon2.is_valid:
        raise ValueError("One of the polygons is invalid.")

    intersection = polygon1.intersection(polygon2)
    intersection_area = intersection.area
    polygon1_area = polygon1.area
    polygon2_area = polygon2.area

    coverage2 = (intersection_area / polygon2_area) * 100
    # uncovered2 = 100 - coverage2

    coverage1 = (intersection_area / polygon1_area) * 100
    uncovered1 = 100 - coverage1

    logging.info("---")
    logging.info("Hausdorff: %.1f ", hausdorff_dist)
    # logging.info("Label Polygon Covered by Found Polygon: %.1f percent", coverage2)
    # # logging.info("Label Polygon NOT Covered by Found Polygon: %.1f percent", uncovered2)
    # logging.info("Found Polygon outside of Label Polygon: %.1f percent", uncovered1)
