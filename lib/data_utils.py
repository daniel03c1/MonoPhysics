import json
import numpy as np
import os
import torch
from plyfile import PlyData


def load_gt_pcds(path):
    if os.path.exists(path.replace("render", "simulation")) and path != path.replace(
        "render", "simulation"
    ):
        print("Loading synthetic Spring-Gauss Point Clouds")
        path = path.replace("render", "simulation")
    elif "vid2sim" in path.lower():
        print("Vid2Sim dataset does not provide GT point clouds")
        return None
    elif os.path.exists(os.path.join(path, "all_data.json")):
        print("Loading synthetic PAC-NeRF Point Clouds")
        path = path.replace("pacnerf", "simulation_data")
    elif os.path.exists(os.path.join(path, "particles")):
        print("Loading Jukebox Point Clouds")
        path = os.path.join(path, "particles")
    else:
        return None

    gts = []
    # Only .ply files (skip hidden files/directories)
    all_files = os.listdir(path)
    ply_files = [f for f in all_files if f.endswith(".ply")]

    if len(ply_files) == 0:
        print(f"Warning: No .ply files found in {path}")
        return None

    indices = [ply.split(".")[0] for ply in ply_files]
    indices.sort()
    n_digits = len(indices[0])
    indices = [int(idx) for idx in indices]
    indices.sort()

    # Use actual indices instead of range to handle gaps or non-zero start
    for idx in indices:
        ply_path = os.path.join(path, f"{idx:0{n_digits}d}.ply")
        if not os.path.exists(ply_path):
            print(f"Warning: Expected file {ply_path} not found, skipping")
            continue

        try:
            plydata = PlyData.read(ply_path)
            vertex = plydata["vertex"]
            np_pcd = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
        except Exception as e:
            print(f"Warning: Failed to load {ply_path}: {e}, skipping")
            continue

        if np_pcd.shape[0] == 0:
            print(f"Warning: Empty point cloud in {ply_path}, skipping")
            continue
        gts.append(torch.tensor(np_pcd, dtype=torch.float32, device="cuda"))

    if len(gts) == 0:
        print(f"Warning: No point clouds loaded from {path}")
        return None

    return gts


def write_dict_to_json(data: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def read_estimation_result(dataset, phys_args, pred_file=None):
    if pred_file is None:
        pred_file = os.path.join(dataset.model_path, f"{phys_args.config_id}-pred.json")
    with open(pred_file, "r") as f:
        result = json.load(f)
    return result


def load_gt_params(path):
    # Spring-Gaus Synthetic
    if os.path.exists(os.path.join(path, "physical.json")):
        with open(os.path.join(path, "physical.json"), "r") as f:
            params = json.load(f)
            gt_params = {
                "v": params[1]["INIT_VELOCITY"],
                "E": params[0]["E"],
                "nu": params[0]["NU"],
            }
    # Vid2Sim
    elif os.path.exists(os.path.join(path, "gt_phys_params.yaml")):
        with open(os.path.join(path, "gt_phys_params.yaml"), "r") as f:
            lines = f.readlines()
        gt_params = {
            "E": float(lines[0].split()[1]),
            "nu": float(lines[1].split()[1]),
            "v": [0.0, 0.0, 0],
        }
    # Jukebox
    elif os.path.exists(os.path.join(path, "metadata.json")):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)
        gt_params = {
            "E": meta["youngs_modulus"],
            "nu": meta["poisson_ratio"],
            "v": meta["initial_velocity"],
        }
        if "yield_stress" in meta:
            gt_params["ys"] = meta["yield_stress"]
    # PAC-NeRF
    else:
        try:
            folder, index = os.path.split(path)
            with open(f"{folder}.json", "r") as f:
                gt_params = json.load(f)[index]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load GT params from {path}: {e}")
            gt_params = {}
    return gt_params
