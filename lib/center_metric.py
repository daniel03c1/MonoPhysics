"""
Appearance-invariant geometry gate: fraction of Gaussian centers that project
INSIDE the GT silhouette mask.

Depends only on particle positions (sim output), the camera, and the GT mask --
never on opacity / scaling / color. So it is invariant to appearance refinement
and to opacity/scaling resets, unlike the rendered alpha-IoU. Used to decide which
frames are geometrically settled enough to refine appearance on.

Projection uses the same intrinsics/extrinsics the gsplat rasterizer uses
(compute_camera_intrinsics -> K, W2C viewmats), so pixel coords align with the
rendered alpha and the GT mask. Pure forward, no autograd, no rendering.
"""

import torch


@torch.no_grad()
def project_centers(points_world, view):
    """
    World-space centers [N,3] -> pixel (u,v) [N], camera-space depth z [N],
    and (W, H). gsplat convention: K + W2C viewmats, z forward.
    """
    from gaussian_renderer import compute_camera_intrinsics

    fx, fy, cx, cy, W, H, K, viewmats = compute_camera_intrinsics(view)
    N = points_world.shape[0]
    ones = torch.ones(N, 1, device=points_world.device, dtype=points_world.dtype)
    hom = torch.cat([points_world, ones], dim=1)  # [N,4]
    p_cam = (viewmats[0].to(points_world.dtype) @ hom.T).T[:, :3]  # [N,3]
    z = p_cam[:, 2]
    proj = (K[0].to(points_world.dtype) @ p_cam.T).T  # [N,3]
    u = proj[:, 0] / proj[:, 2].clamp_min(1e-8)
    v = proj[:, 1] / proj[:, 2].clamp_min(1e-8)
    return u, v, z, W, H


@torch.no_grad()
def center_inside_fraction(points_world, view, mask_threshold=0.0):
    """
    Fraction of centers that are in-frame, in front of the camera, and land on
    a GT-foreground pixel. Returns a Python float in [0, 1].
    """
    u, v, z, W, H = project_centers(points_world, view)
    gt = view.gt_alpha_mask.to(points_world.device)  # [1, H, W]
    in_frame = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui = u.round().long().clamp(0, W - 1)
    vi = v.round().long().clamp(0, H - 1)
    on_fg = gt[0, vi, ui] > mask_threshold
    return (in_frame & on_fg).float().mean().item()


@torch.no_grad()
def per_frame_center_inside(cached_positions, views_per_frame, mask_threshold=0.0):
    """
    Per-frame center-inside fraction, averaged over each frame's views.

    cached_positions: list of [N,3] world-space positions (one per frame).
    views_per_frame:  parallel list; element f is the views at frame f.
    Returns list[float].
    """
    out = []
    for x, views in zip(cached_positions, views_per_frame):
        vals = [center_inside_fraction(x, v, mask_threshold) for v in views]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out
