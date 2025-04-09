import os
import subprocess
import time
from statistics import mean
import torch


def get_avg_times(scenes, dataset, path, method, model):
    print("================================================")
    print(dataset, method, model)

    if dataset == 'pandaset':
        scenes = scenes[0][2:]
    else:
        scenes = range(scenes)

    times = []
    for scene in scenes:
        start_time = time.time()
        arguments = ["-d", dataset, "-p", path, "-s", str(scene), "-a", method, "-m", model, "--show", "False"]
        subprocess.run(["python", "main.py"] + arguments)
        torch.cuda.empty_cache()
        total_time = time.time() - start_time
        print("\t\tTime for scene ", scene, " : ", total_time, " seconds")
        times.append(total_time)
    print(times)
    print("================================================\n")

    return mean(times)


def main():
    path = "/data/path/"

    for model in ['sphere', 'randlanet']:
        # semantickitti
        dataset = "semantickitti"
        data_path = os.path.join(path, "SemanticKITTI/dataset/")
        scenes = range(22)

        mesh_avg = get_avg_times(scenes, dataset, data_path, "mesh", model)
        contour_avg = get_avg_times(scenes, dataset, data_path, "contour", model)
        print("\nMesh: ", mesh_avg, "\nContour: ", contour_avg)

        # nuscenes
        dataset = "nuscenes"
        data_path = os.path.join(path, "nuscenes_mini")
        scenes = range(10)

        mesh_avg = get_avg_times(scenes, dataset, data_path, "mesh", model)
        contour_avg = get_avg_times(scenes, dataset, data_path, "contour", model)
        print("\nMesh: ", mesh_avg, "\nContour: ", contour_avg)

        # pandaset
        dataset = "pandaset"
        data_path = os.path.join(path, "pandaset")
        scenes = [os.listdir(data_path)]

        mesh_avg = get_avg_times(scenes, dataset, data_path, "mesh", model)
        contour_avg = get_avg_times(scenes, dataset, data_path, "contour", model)
        print("\nMesh: ", mesh_avg, "\nContour: ", contour_avg)


if __name__ == "__main__":
    main()
