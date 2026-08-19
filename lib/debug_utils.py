import numpy as np
import os
import torch
import torchvision
from plyfile import PlyData, PlyElement

from gaussian_renderer import render, render_optical_flow
from lib.flow import flow_to_image


def write_point_cloud_ply(path, xyz, rgb=None, normals=None):
    """
    Write a simple point cloud PLY: xyz [N, 3] float32, optional rgb [N, 3]
    uint8 and normals [N, 3] float32.
    """
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if normals is not None:
        dtype += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if rgb is not None:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    if normals is not None:
        elements["nx"] = normals[:, 0]
        elements["ny"] = normals[:, 1]
        elements["nz"] = normals[:, 2]
    if rgb is not None:
        elements["red"] = rgb[:, 0]
        elements["green"] = rgb[:, 1]
        elements["blue"] = rgb[:, 2]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def save_volume_colored_ply(gaussians, path):
    """
    Save a PLY colored by per-particle scale (induced_volumes^(1/3));
    black = small scale, white = large scale.
    """
    volumes = gaussians.get_induced_volumes().detach()
    scale = volumes ** (1.0 / 3.0)
    normalized = (scale - scale.min()) / (scale.max() - scale.min() + 1e-8)
    gray = (normalized * 255).clamp(0, 255).byte().cpu().numpy()  # [N]
    rgb = np.stack([gray, gray, gray], axis=1)  # [N, 3]

    xyz = gaussians.get_xyz().detach().cpu().numpy()
    write_point_cloud_ply(path, xyz, rgb=rgb)


def save_gradient_ply(gaussians, path):
    """
    Save a PLY encoding position gradients: color = normal-map direction
    (R=X, G=Y, B=Z) modulated by log-magnitude; normals = gradient direction
    (MeshLab arrows). Positions world-space; grads from _xyz.grad (scaled camera space).
    """
    grad = gaussians._xyz.grad
    if grad is None:
        print(f"[save_debug_snapshot] _xyz.grad is None, skipping gradient PLY: {path}")
        return

    grad = grad.detach()
    mag = grad.norm(dim=-1)  # [N]

    # Log-scale normalization: gradient magnitudes are skewed, log spreads the range
    log_mag = torch.log1p(mag)
    normalized_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

    direction = grad / (mag.unsqueeze(-1) + 1e-12)  # [N, 3], normalized

    modulated = direction * normalized_mag.unsqueeze(-1)  # [N, 3], range ~[-1, 1]
    rgb = ((modulated * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()  # [N, 3]

    xyz = gaussians.get_xyz().detach().cpu().numpy()
    write_point_cloud_ply(path, xyz, rgb=rgb, normals=direction.cpu().numpy())


@torch.no_grad()
def save_debug_snapshot(estimator, iteration, experiment_dir, pipe_args, background):
    debug_dir = os.path.join(experiment_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    ply_path = os.path.join(debug_dir, f"iter_{iteration:05d}.ply")
    estimator.scene.gaussians.save_ply(ply_path)

    vol_ply_path = os.path.join(debug_dir, f"iter_{iteration:05d}_volumes.ply")
    save_volume_colored_ply(estimator.scene.gaussians, vol_ply_path)

    grad_ply_path = os.path.join(debug_dir, f"iter_{iteration:05d}_gradients.ply")
    save_gradient_ply(estimator.scene.gaussians, grad_ply_path)

    def _flow_to_rgb(flow_tensor):
        img = flow_to_image(flow_tensor.permute(1, 2, 0).cpu().numpy())
        return (
            torch.from_numpy(img).permute(2, 0, 1).float().to(flow_tensor.device)
            / 255.0
        )

    current_xyz = estimator.scene.gaussians.get_xyz()
    render_view = estimator.scene.getTrainCameras(exact_fid=0.0)[0]

    for t in range(len(estimator.cached_positions)):
        results = render(
            render_view,
            estimator.scene.gaussians,
            pipe_args,
            background,
            estimator.cached_positions[t] - current_xyz,
        )
        rendered_rgb = results["render"]
        rendered_alpha = results["alpha"].expand(3, -1, -1)

        flow_view = estimator.views[t][0]
        gt_rgb = flow_view.original_image.to(rendered_rgb.device)
        diff_rgb = ((gt_rgb - rendered_rgb) / 2.0 + 0.5).clamp(0.0, 1.0)
        blank = torch.zeros_like(rendered_rgb)
        if t == 0:
            velocities = torch.zeros_like(current_xyz)
            d_xyz = torch.zeros_like(current_xyz)
        else:
            d_xyz = estimator.cached_positions[t - 1] - current_xyz
            velocities = (
                estimator.cached_positions[t] - estimator.cached_positions[t - 1]
            )

        flow_results = render_optical_flow(
            flow_view, estimator.scene.gaussians, velocities, pipe_args, d_xyz
        )
        rendered_flow = _flow_to_rgb(flow_results["flow"])

        fid = flow_view.fid.item()
        uid = flow_view.uid
        flow_data = estimator.get_optical_flow(fid, uid)
        if flow_data is not None and "f" in flow_data:
            target_flow = _flow_to_rgb(flow_data["f"][0])
        else:
            target_flow = torch.zeros_like(rendered_flow)

        top_row = torch.cat([rendered_rgb, rendered_alpha, diff_rgb], dim=-1)
        bottom_row = torch.cat([rendered_flow, target_flow, blank], dim=-1)
        image = torch.cat([top_row, bottom_row], dim=-2)

        img_path = os.path.join(debug_dir, f"iter_{iteration:05d}_frame{t}.png")
        torchvision.utils.save_image(image, img_path)
