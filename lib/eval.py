import json
import numpy as np
import os
import random
import torch
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from simulator import Estimator, MPMSimulator
from lib.loss_utils import earth_movers_distance


def export_result(
    dataset,
    phys_args,
    estimator: Estimator,
    losses,
    config_id,
    prefix="0",
    postfix="",
    output_dir=None,
):
    save_attr = ["mpm_iter_cnt", "rho", "voxel_size", "bc", "fps", "density_grid_size"]
    pred = {}
    pred["config_id"] = config_id
    for attr in save_attr:
        pred[attr] = getattr(phys_args, attr)

    pred["vel"] = estimator.init_vel.detach().cpu().numpy().tolist()
    pred["gravity"] = estimator.gravity.detach().cpu().numpy().tolist()

    mat_params = dict()
    m = phys_args.material
    mat_params["material"] = m

    estimator.eval()

    if m in [MPMSimulator.elasticity, MPMSimulator.neo_hookean]:
        mat_params["E"] = estimator.get_derived_E().item()
        mat_params["nu"] = estimator.get_derived_nu().item()
        mat_params["K"] = estimator.get_derived_K().item()
        mat_params["G"] = estimator.get_derived_G().item()
    elif m == MPMSimulator.viscous_fluid:
        mat_params["mu"] = estimator.get_mu().item()
        mat_params["kappa"] = estimator.get_kappa().item()
    elif m in [MPMSimulator.plasticine, MPMSimulator.plasticine_corotated]:
        mat_params["E"] = estimator.get_derived_E().item()
        mat_params["nu"] = estimator.get_derived_nu().item()
        mat_params["K"] = estimator.get_derived_K().item()
        mat_params["G"] = estimator.get_derived_G().item()
        mat_params["yield_stress"] = estimator.get_yield_stress().item()
    elif m == MPMSimulator.non_newtonian:
        mat_params["mu"] = estimator.get_mu().item()
        mat_params["kappa"] = estimator.get_kappa().item()
        mat_params["yield_stress"] = estimator.get_yield_stress().item()
        mat_params["plastic_viscosity"] = estimator.get_plastic_viscosity().item()
    elif m == MPMSimulator.drucker_prager:
        mat_params["E"] = estimator.get_derived_E().item()
        mat_params["nu"] = estimator.get_derived_nu().item()
        mat_params["K"] = estimator.get_derived_K().item()
        mat_params["G"] = estimator.get_derived_G().item()
        # Stored as a friction angle in degrees under the inherited key name.
        mat_params["friction_alpha"] = estimator.get_friction_angle().item()
    else:
        raise ValueError(f"Invalid material: {m}")

    pred["mat_params"] = mat_params
    pred["losses"] = losses

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        estimator.scene.gaussians.save_ply(os.path.join(output_dir, "gs.ply"))
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(pred, f, indent=4)
    else:
        estimator.scene.gaussians.save_ply(
            os.path.join(dataset.model_path, "gs", f"{prefix}-gs{postfix}.ply")
        )

        with open(
            os.path.join(dataset.model_path, f"{prefix}-pred{postfix}.json"), "w"
        ) as f:
            json.dump(pred, f, indent=4)


def evaluate(preds, gts, train_frames, loss_type="CD", seed=None, until=None):
    print(f"Prediction sequence {len(preds)}, gts sequence {len(gts)}")

    if len(preds) != len(gts):
        raise ValueError(
            f"prediction sequence has {len(preds)} frames but gt has {len(gts)}"
        )

    print(
        f"Prediction pcd particles cnt {preds[0].shape[0]}, "
        f"gt pcd particles cnt {gts[0].shape[0]}"
    )
    max_f = len(preds)
    fit_loss = 0.0
    predict_loss = 0.0

    losses = []

    if seed is not None:
        random.seed(seed)

    for f in tqdm(range(max_f), desc=f"Evaluate {loss_type} Loss"):
        if until is not None and f > until:
            loss = 0.0
        else:
            # cd align with https://zlicheng.com/spring_gaus/
            n_sample = 2048 if loss_type == "EMD" else 8192
            pcd0 = preds[f]
            pcd1 = gts[f]
            n_sample = min(n_sample, pcd0.shape[0], pcd1.shape[0])

            pcd0 = pcd0[random.sample(range(pcd0.shape[0]), n_sample), :]
            pcd1 = pcd1[random.sample(range(pcd1.shape[0]), n_sample), :]

            if loss_type == "CD":
                loss = (chamfer_distance(pcd0[None], pcd1[None])[0] * 1e3).item()
            elif loss_type == "EMD":
                loss = earth_movers_distance(pcd0, pcd1).item()
            else:
                raise ValueError(
                    f"unknown loss_type {loss_type!r}; expected 'CD' or 'EMD'"
                )

        if f < train_frames:
            fit_loss += loss
        else:
            predict_loss += loss

        losses.append(loss)

    fit_loss /= train_frames
    if max_f - train_frames > 0.0:
        predict_loss /= max_f - train_frames
    print(
        f"{loss_type} loss train: {fit_loss}, "
        f"{loss_type} loss predict: {predict_loss}"
    )

    return fit_loss, predict_loss, losses


def compute_pcd_metrics(pred, gt):
    """CD (×1e3) and EMD for one pair of point clouds ([N, 3] / [M, 3] cuda tensors)."""
    n_cd = min(8192, pred.shape[0], gt.shape[0])
    n_emd = min(2048, pred.shape[0], gt.shape[0])

    idx0 = random.sample(range(pred.shape[0]), n_cd)
    idx1 = random.sample(range(gt.shape[0]), n_cd)
    cd = (chamfer_distance(pred[idx0][None], gt[idx1][None])[0] * 1e3).item()

    idx0 = random.sample(range(pred.shape[0]), n_emd)
    idx1 = random.sample(range(gt.shape[0]), n_emd)
    emd = earth_movers_distance(pred[idx0], gt[idx1]).item()

    return cd, emd


def evaluate_depth_scaled(pred, gt, camera):
    """
    Frame-0 CD (×1e3) and EMD after depth-scale alignment: predicted points
    are rescaled uniformly in camera space so their center depth matches the GT
    center depth, then transformed back to world space.
    """
    pred_cam = camera.pw2pc(pred)
    gt_cam = camera.pw2pc(gt)

    gt_depth = gt_cam[:, 2].mean()
    pred_depth = pred_cam[:, 2].mean()
    pred_cam_scaled = pred_cam * (gt_depth / pred_depth)

    # Camera → world: pw = (pc - T) @ R.T
    R = torch.from_numpy(camera.R).to(pred)
    T = torch.from_numpy(camera.T).to(pred).reshape(1, 3)
    pred_scaled = (pred_cam_scaled - T) @ R.T

    return compute_pcd_metrics(pred_scaled, gt)


def eval_mat_est_acc(
    base_simulator, target_params, prefix="MAE ", use_default_value=True
):
    """
    Evaluate material estimation accuracy against target_params;
    use_default_value inserts 0.0 for params absent from target_params.
    """
    results = {}

    # Estimator exposes getter methods; Simulator exposes direct attributes
    is_estimator = hasattr(base_simulator, "get_derived_E")

    if "v" in target_params:
        if is_estimator:
            vel = base_simulator.init_vel.detach().cpu().numpy()
        else:
            vel = base_simulator.vel.detach().cpu().numpy()
        results[f"{prefix}v"] = np.mean(np.abs(vel - np.array(target_params["v"])))
        gt_v = np.array(target_params["v"])
        vel_norm = np.linalg.norm(vel)
        gt_norm = np.linalg.norm(gt_v)
        if vel_norm > 1e-8 and gt_norm > 1e-8:
            results[f"{prefix}cos v"] = float(np.dot(vel, gt_v) / (vel_norm * gt_norm))
        else:
            results[f"{prefix}cos v"] = 0.0
    elif use_default_value:
        results[f"{prefix}v"] = 0.0

    if "E" in target_params:
        if is_estimator:
            E = base_simulator.get_derived_E().item()
        else:
            E = base_simulator.mat.get("E", None)
        if E is not None:
            results[f"{prefix}log E"] = abs(np.log10(E) - np.log10(target_params["E"]))
    elif use_default_value:
        results[f"{prefix}log E"] = 0.0

    if "nu" in target_params:
        if is_estimator:
            nu = base_simulator.get_derived_nu().item()
        else:
            nu = base_simulator.mat.get("nu", None)
        if nu is not None:
            results[f"{prefix}nu"] = abs(nu - target_params["nu"])
    elif use_default_value:
        results[f"{prefix}nu"] = 0.0

    if "mu" in target_params:
        if is_estimator:
            mu = base_simulator.get_derived_G().item()
        else:
            mu = base_simulator.mat.get("mu", None)
        if mu is not None:
            results[f"{prefix}log mu"] = abs(
                np.log10(mu) - np.log10(target_params["mu"])
            )
    elif use_default_value:
        results[f"{prefix}log mu"] = 0.0

    if "kappa" in target_params:
        if is_estimator:
            kappa = base_simulator.get_derived_K().item()
        else:
            kappa = base_simulator.mat.get("kappa", None)
        if kappa is not None:
            results[f"{prefix}log kappa"] = abs(
                np.log10(kappa) - np.log10(target_params["kappa"])
            )
    elif use_default_value:
        results[f"{prefix}log kappa"] = 0.0

    if "ys" in target_params:
        if is_estimator:
            ys = base_simulator.get_yield_stress().item()
        else:
            ys = base_simulator.mat.get("yield_stress", None)
        if ys is not None:
            results[f"{prefix}log yield stress"] = abs(
                np.log10(ys) - np.log10(target_params["ys"])
            )
    elif use_default_value:
        results[f"{prefix}log yield stress"] = 0.0

    if "eta" in target_params:
        if is_estimator:
            eta = base_simulator.get_plastic_viscosity().item()
        else:
            eta = base_simulator.mat.get("plastic_viscosity", None)
        if eta is not None:
            results[f"{prefix}log eta"] = abs(
                np.log10(eta) - np.log10(target_params["eta"])
            )
    elif use_default_value:
        results[f"{prefix}log eta"] = 0.0

    if "fa" in target_params:
        if is_estimator:
            fa = base_simulator.get_friction_angle().item()
        else:
            fa = base_simulator.mat.get("friction_alpha", None)
        if fa is not None:
            results[f"{prefix}friction_alpha"] = abs(fa - target_params["fa"])
    elif use_default_value:
        results[f"{prefix}friction_alpha"] = 0.0

    return results
