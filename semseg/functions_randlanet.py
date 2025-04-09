import numpy as np
import open3d.ml as _ml3d

def run_inference(dataset, dataset_path, scan_list):

    if dataset == "semantickitti":
        import open3d.ml.tf as ml3d
        cfg_file = "open3dML/weights_randlanet_semantickitti/randlanet_semantickitti.yml"
        ckpt_path = "open3dML/weights_randlanet_semantickitti/ckpt-1"

    elif dataset == "pandaset":
        import open3d.ml.tf as ml3d
        cfg_file = "open3dML/to_include_in Open3D-ML/randlanet_pandaset.yml"
        ckpt_path = "open3dML/weights_randlanet_pandaset/ckpt_randlanet_pandaset"

    else:  # nuscenes
        import open3d.ml.torch as ml3d
        cfg_file = "open3dML/weights_randlanet_nuscenes/randlanet_nuscenes.yml"
        ckpt_path = "open3dML/weights_randlanet_nuscenes/ckpt_00010.pth"

    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    model = ml3d.models.RandLANet(**cfg.model)
    cfg.dataset['dataset_path'] = dataset_path
    dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    results = []
    for i in range(len(scan_list)):
        scan = scan_list[i]
        pcd_dict = {
            'point': scan[:, :3],
            'intensity': scan[3].astype(int),
            'label': np.zeros(len(scan)).astype(int)
        }
        result = pipeline.run_inference(pcd_dict)
        results.append(result['predict_labels'])

    return results
