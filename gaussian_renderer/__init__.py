#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import functools
import math
import torch
from gsplat import rasterization

from scene.gaussian_model import GaussianModel
from lib.sh_utils import eval_sh


@functools.lru_cache(maxsize=8)
def pixel_center_grid(height, width, device):
    """Cached [2, H, W] grid of pixel centers (u, v) under the +0.5 convention."""
    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    grid_v, grid_u = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=0)


def compute_camera_intrinsics(viewpoint_camera):
    """
    Compute intrinsics and view matrix; K is [1, 3, 3] and viewmats is
    [1, 4, 4] row-major, both on CUDA.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device="cuda", dtype=torch.float32
    ).unsqueeze(0)
    viewmats = viewpoint_camera.world_view_transform.T.unsqueeze(0)
    return fx, fy, cx, cy, width, height, K, viewmats


def compute_means3d(pc, d_xyz):
    """Apply position deformation; d_xyz is scalar 0 or an additive [N, 3] delta."""
    return pc.get_xyz() + d_xyz


def prepare_gaussian_params(pc, pipe, scaling_modifier):
    """
    Prepare (scales, quats, covars, opacities) for rasterization. With
    pipe.compute_cov3D_python: covars precomputed, scales/quats None; else reversed.
    """
    opacities = pc.get_opacity().squeeze(-1)
    if pipe.compute_cov3D_python:
        covars = pc.get_covariance(scaling_modifier)
        scales, quats = None, None
    else:
        scales = pc.get_scaling() * scaling_modifier
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        quats = pc.get_rotation()
        covars = None
    return scales, quats, covars, opacities


def render_optical_flow(
    viewpoint_camera,
    pc: GaussianModel,
    velocities_world,
    pipe,
    d_xyz,
    scaling_modifier=1.0,
):
    """
    Render optical flow from per-Gaussian world-space velocities [N, 3].
    Returns 'flow' [2, H, W] in pixels (u, v), 'gaussian_ids' (packed), 'alpha'.
    """
    fx, fy, cx, cy, width, height, K, viewmats = compute_camera_intrinsics(
        viewpoint_camera
    )
    means3D = compute_means3d(pc, d_xyz)
    scales, quats, covars, opacities = prepare_gaussian_params(
        pc, pipe, scaling_modifier
    )

    # Single pass: sh_degree=None renders velocities as raw features, RGB+ED adds depth
    bg_velocity = torch.zeros(1, 3, device="cuda", dtype=torch.float32)

    velocity_and_depth, render_alphas, meta = rasterization(
        means=means3D,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=velocities_world,
        viewmats=viewmats,
        Ks=K,
        width=width,
        height=height,
        sh_degree=None,
        render_mode="RGB+ED",
        backgrounds=bg_velocity,
        covars=covars,
        packed=True,
    )

    depth_map = velocity_and_depth[0, :, :, 3]  # [H, W]
    alpha_map = render_alphas[0, :, :, 0]  # [H, W]
    velocity_map_world = velocity_and_depth[0, :, :, :3].permute(2, 0, 1)  # [3, H, W]

    # Transform velocity from world to camera space
    velocity_cam = torch.einsum(
        "ij,jab->iab",
        viewpoint_camera.world_view_transform.T[:3, :3],  # column to row major
        velocity_map_world,
    )

    # Pixel grids: indexing="ij" gives vs = row (y), us = column (x)
    vs, us = torch.meshgrid(
        torch.arange(height, device="cuda", dtype=torch.float32),
        torch.arange(width, device="cuda", dtype=torch.float32),
        indexing="ij",
    )

    valid_mask = alpha_map > 0.0  # [H, W]

    # Dummy depth where alpha == 0 avoids div-by-zero; that flow is zeroed below
    Z = torch.where(valid_mask, depth_map, torch.ones_like(depth_map))
    Z = Z.clamp(min=0)

    # Unproject to camera space, advect by velocity, reproject
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    P_cam = torch.stack([X, Y, Z], dim=0)  # [3, H, W]
    P_next = P_cam + velocity_cam  # [3, H, W]

    Z_next = P_next[2].clamp(min=0)
    u_next = fx * P_next[0] / Z_next + cx
    v_next = fy * P_next[1] / Z_next + cy

    flow_u = u_next - us
    flow_v = v_next - vs

    flow_u = flow_u * valid_mask
    flow_v = flow_v * valid_mask

    flow = torch.stack([flow_u, flow_v], dim=0)  # [2, H, W]

    gaussian_ids = meta.get("gaussian_ids")
    return {"flow": flow, "gaussian_ids": gaussian_ids, "alpha": alpha_map}


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    d_xyz=None,
    scaling_modifier=1.0,
    override_color=None,
):
    """Render the scene using gsplat. bg_color must be on GPU."""
    if d_xyz is None:
        d_xyz = torch.zeros_like(pc.get_xyz())

    fx, fy, cx, cy, width, height, K, viewmats = compute_camera_intrinsics(
        viewpoint_camera
    )
    means3D = compute_means3d(pc, d_xyz)
    scales, quats, covars, opacities = prepare_gaussian_params(
        pc, pipe, scaling_modifier
    )

    # Prepare colors or SH coefficients
    if override_color is not None:
        colors = override_color
        sh_degree = None
    elif pipe.convert_SHs_python:
        shs_view = (
            pc.get_features().transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        )
        dir_pp = pc.get_xyz() - viewpoint_camera.camera_center.repeat(
            pc.get_features().shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        sh_degree = None
    else:
        colors = pc.get_features()
        sh_degree = pc.active_sh_degree

    render_colors, render_alphas, meta = rasterization(
        means=means3D,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
        backgrounds=bg_color.unsqueeze(0) if bg_color.dim() == 1 else bg_color,
        covars=covars,
        packed=True,
    )

    rendered_image = render_colors[0, :, :, :3].permute(2, 0, 1)  # [3, H, W]
    depth = render_colors[0, :, :, 3:4].permute(2, 0, 1)  # [1, H, W]
    alpha = render_alphas[0].permute(2, 0, 1)  # [1, H, W]

    radii_raw = meta["radii"]
    if radii_raw.dim() > 1:
        radii = radii_raw[0]  # [N]
    else:
        radii = radii_raw

    # packed=True returns only visible Gaussians; expand radii to full [N]
    N = means3D.shape[0]
    if "gaussian_ids" in meta:
        full_radii = torch.zeros(N, dtype=radii.dtype, device=radii.device)
        full_radii[meta["gaussian_ids"]] = radii
        visibility_filter = full_radii > 0
        radii = full_radii
    else:
        visibility_filter = radii > 0

    return {
        "render": rendered_image,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": depth,
        "alpha": alpha,
    }


def render_pixel_coords(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    d_xyz=None,
    scaling_modifier=1.0,
):
    """
    Render the per-pixel image-space coordinate map via the diff_pixel_coords CUDA path.
    The returned values are the pixel grid itself by construction; the point is the
    gradient, which flows to Gaussian positions (mu_2D) and 2D-covariance Cholesky
    factors rather than to blending weights. Returns pixel_coords [2, H, W],
    alpha [1, H, W], radii.
    """
    from diff_pixel_coords import (
        PixelCoordSettings,
        rasterize_pixel_coords,
    )

    if d_xyz is None:
        d_xyz = torch.zeros_like(pc.get_xyz())

    means3D = compute_means3d(pc, d_xyz)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)

    # Dummy screenspace_points for gradient retention (unused in this path)
    screenspace_points = torch.zeros_like(means3D[:, :2], requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # This path has no covars-precompute support. Scale/opacity are detached
    # so the silhouette gradient flows only to positions; appearance params are
    # supervised solely by the photometric/alpha losses.
    scales = (pc.get_scaling() * scaling_modifier).detach()
    if scales.shape[-1] == 1:
        scales = scales.repeat(1, 3)
    rotations = pc.get_rotation()
    opacities = pc.get_opacity().detach()
    cov3Ds_precomp = torch.Tensor([]).cuda()

    # SH is used internally for preprocessing even though color is unused
    sh = pc.get_features()

    raster_settings = PixelCoordSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    raw_pixel_coords, radii = rasterize_pixel_coords(
        means3D,
        screenspace_points,
        sh,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

    # Uncovered pixels show their own pixel coordinate.
    alpha = raw_pixel_coords[2:3]  # [1, H, W], accumulated alpha
    T_remaining = 1.0 - alpha  # [1, H, W]
    pixel_grid = pixel_center_grid(height, width, means3D.device)  # [2, H, W]
    pixel_coords = raw_pixel_coords[:2] + T_remaining * pixel_grid

    # Snap tiny numeric noise to exact half-integer pixel coords (gradient-free)
    pixel_coords += (torch.round(pixel_coords + 0.5) - 0.5 - pixel_coords).detach()

    return {
        "pixel_coords": pixel_coords,  # [2, H, W] — covered pixels + background grid
        "alpha": alpha,  # [1, H, W]
        "radii": radii,
    }
