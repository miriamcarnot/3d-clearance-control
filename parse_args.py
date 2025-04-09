import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Clear Height Analysis')
    parser.add_argument('-d', '--dataset', help='choose a dataset, pandaset, semantickitti or nuscenes',
                        default="semantickitti")
    parser.add_argument('-p', '--path', help='path to the dataset', default='')
    parser.add_argument('-s', '--sequence', help='dataset sequence to run, check readme for options', default="1")
    parser.add_argument('-a', '--algorithm', help='algorithm for finding the road area, either contour or mesh',
                        default="mesh")
    parser.add_argument('-m', '--model', help='choose a semantic segmentation model: sphere or randlanet',
                        default="sphere")
    parser.add_argument('--start', help='start from the i-th scan',
                        default="0")
    parser.add_argument('--stop', help='stop at the i-th scan',
                        default="30")
    parser.add_argument('--step', help='every i-th scan will be concatenated',
                        default="5")
    parser.add_argument('--show', help='set to False for no visualization',
                        default="True")

    args = parser.parse_args()
    return args
