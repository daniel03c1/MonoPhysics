"""
Filtering and sampling of particles from Gaussian representations using
multi-view consistency and density volumes.
"""

import torch
from tqdm import tqdm

from gaussian_renderer import render
from lib.system_utils import write_particles


def estimate_volume_and_voxel_size(
    density_volume,
    curr_grid_size,
    density_min_th,
    n_init_particles=None,
):
    """
    Estimate total volume and the sampling grid size yielding ~n_init_particles.
    Returns (total_volume, None, occupied_count) if n_init_particles is None,
    else (total_volume, effective_sampling_grid_size, n_init_particles).
    """
    occupied_count = (density_volume >= density_min_th).sum().item()

    voxel_vol = curr_grid_size**3
    total_volume = occupied_count * voxel_vol

    if n_init_particles is None:
        return total_volume, None, occupied_count

    # From n_particles = (bbox / grid_size)^3 * fill_ratio
    effective_sampling_grid_size = curr_grid_size * (
        occupied_count / n_init_particles
    ) ** (1 / 3)

    return total_volume, effective_sampling_grid_size, n_init_particles


def filter_particles_multiview(
    init_inner_points,
    xyzt,
    random_cameras,
    gaussians,
    pipeline,
    background,
):
    """
    Filter particles using multi-view depth and mask consistency.
    Returns (filtered_inner_points, filtered_xyzt).
    """
    for viewpoint_cam in tqdm(random_cameras, desc="Rendering progress"):
        results = render(viewpoint_cam, gaussians, pipeline, background, 0.0)
        depth = results["depth"][0]

        # Use rendered alpha as mask (random cameras don't have gt_alpha_mask)
        render_mask = results["alpha"][0] > 1 / 255
        if viewpoint_cam.gt_alpha_mask is not None:
            render_mask = torch.logical_and(
                render_mask, viewpoint_cam.gt_alpha_mask[0] > 0
            )

        pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(init_inner_points)

        # Remove points outside image space
        in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
        init_inner_points = init_inner_points[in_mask]
        pix_w = pix_w[in_mask]
        pix_h = pix_h[in_mask]
        pix_d = pix_d[in_mask]

        # Remove points outside object mask
        pix_mask = render_mask[pix_h, pix_w]
        init_inner_points = init_inner_points[pix_mask]
        pix_w = pix_w[pix_mask]
        pix_h = pix_h[pix_mask]
        pix_d = pix_d[pix_mask]

        # Remove points in front of rendered depth
        render_pix_d = depth[pix_h, pix_w]
        depth_mask = render_pix_d < pix_d
        init_inner_points = init_inner_points[depth_mask]

        # Remove outliers from xyzt
        render_mask = results["alpha"][0] > 1 / 255
        pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(xyzt)
        in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
        pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
        xyzt = xyzt[in_mask]
        pix_mask = render_mask[pix_h, pix_w]
        xyzt = xyzt[pix_mask]

    return init_inner_points, xyzt


def build_density_volume(
    init_inner_points,
    xyzt,
    bbox_mins,
    bbox_bounds,
    grid_size,
    num_iter,
    smoothing_iters=20,
    conv_kernel_size=3,
):
    """
    Build a hierarchical 3D density volume through iterative grid refinement.
    Returns (density_volume, curr_grid_size, volume_size, bbox_mins, bbox_bounds).
    """
    kernel_shape = (1, 1, conv_kernel_size, conv_kernel_size, conv_kernel_size)
    weight = torch.ones(kernel_shape).to(xyzt)
    weight = weight / weight.sum()

    # Initialize at coarse level
    curr_grid_size = grid_size / 2
    volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
    bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
    bbox_bounds = bbox_maxs - bbox_mins

    density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(
        init_inner_points
    )

    # Mark initial points
    ids = torch.round(
        (init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size
    ).to(torch.int64)
    valid_mask = (ids >= 0).all(dim=1) & (ids < volume_size.reshape(1, 3)).all(dim=1)
    ids = ids[valid_mask]
    density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0

    ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
    valid_mask = (ids >= 0).all(dim=1) & (ids < volume_size.reshape(1, 3)).all(dim=1)
    ids = ids[valid_mask]
    density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0

    # Hierarchical refinement
    for i in range(2, num_iter):
        curr_grid_size = grid_size / 2**i
        volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
        bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size

        grid_xyz = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(0, volume_size[0] - 1, volume_size[0]),
                    torch.linspace(0, volume_size[1] - 1, volume_size[1]),
                    torch.linspace(0, volume_size[2] - 1, volume_size[2]),
                ),
                dim=-1,
            ).to(bbox_mins)
            * curr_grid_size
            + bbox_mins[None, None, None]
        )

        # Upsample density volume
        ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[
            None, None, None
        ] * 2 - 1
        ids_norm = ids_norm[None].flip((-1,))
        density_volume = torch.nn.functional.grid_sample(
            density_volume[None, None],
            ids_norm,
            mode="bilinear",
            align_corners=True,
        )
        density_volume = torch.nn.functional.conv3d(
            density_volume, weight=weight, padding="same"
        )[0, 0]
        density_volume[density_volume < 0.5] = 0.0

        # Re-mark points
        ids = torch.round(
            (init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size
        ).to(torch.int64)
        valid_mask = (ids >= 0).all(dim=1) & (ids < volume_size.reshape(1, 3)).all(
            dim=1
        )
        ids = ids[valid_mask]
        density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0

        ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(
            torch.int64
        )
        valid_mask = (ids >= 0).all(dim=1) & (ids < volume_size.reshape(1, 3)).all(
            dim=1
        )
        ids = ids[valid_mask]
        density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
        bbox_bounds = bbox_maxs - bbox_mins

    # Final smoothing iterations
    for _ in range(smoothing_iters):
        density_volume = torch.nn.functional.conv3d(
            density_volume[None, None], weight=weight, padding="same"
        )[0, 0]
        density_volume[density_volume < 0.5] = 0.0

        ids = torch.round(
            (init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size
        ).to(torch.int64)
        valid_mask = (ids >= 0).all(dim=1) & (ids < volume_size.reshape(1, 3)).all(
            dim=1
        )
        ids = ids[valid_mask]
        density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0

        ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(
            torch.int64
        )
        valid_mask = (ids >= 0).all(dim=1) & (ids < volume_size.reshape(1, 3)).all(
            dim=1
        )
        ids = ids[valid_mask]
        density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0

    return density_volume, curr_grid_size, volume_size, bbox_mins, bbox_bounds


def sample_particles(
    density_volume,
    volume_size,
    curr_grid_size,
    bbox_mins,
    bbox_bounds,
    density_min_th,
    random_sample,
    model_path,
    conv_kernel_size=3,
    sampling_grid_size=None,
):
    """
    Sample particles from the density volume; if sampling_grid_size is given,
    sample at that resolution to hit the target particle count.
    Returns (sampled_points, densities, final_grid_size).
    """
    kernel_shape = (1, 1, conv_kernel_size, conv_kernel_size, conv_kernel_size)
    weight = torch.ones(kernel_shape).to(density_volume)
    weight = weight / weight.sum()

    target_grid_size = (
        sampling_grid_size if sampling_grid_size is not None else curr_grid_size
    )

    # arange (not linspace) ensures exact target_grid_size spacing
    new_volume_size = torch.ceil(bbox_bounds / target_grid_size).to(torch.int64) + 1

    steps_x = torch.arange(0, new_volume_size[0], 1.0, device=bbox_mins.device)
    steps_y = torch.arange(0, new_volume_size[1], 1.0, device=bbox_mins.device)
    steps_z = torch.arange(0, new_volume_size[2], 1.0, device=bbox_mins.device)

    # Candidate particle positions in world space
    grid_xyz = (
        torch.stack(
            torch.meshgrid(steps_x, steps_y, steps_z, indexing="ij"),
            dim=-1,
        )
        * target_grid_size
        + bbox_mins[None, None, None]
    )

    # Sample the ORIGINAL density_volume to avoid downsampling aliasing
    ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[
        None, None, None
    ] * 2 - 1
    ids_norm = ids_norm[None].flip((-1,))

    def sample_vol(vol, coords):
        return torch.nn.functional.grid_sample(
            vol[None, None],
            coords,
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    density_at_grid = sample_vol(density_volume, ids_norm)

    particles = grid_xyz[density_at_grid > 0.5]

    if random_sample:
        # Random jitter within the voxel
        delta = torch.rand_like(particles) * target_grid_size - (target_grid_size * 0.5)
        particles = particles + delta

    # Re-verify density at exact particle positions
    ids_norm_particles = (
        particles[None, None] - bbox_mins[None, None, None]
    ) / bbox_bounds[None, None, None] * 2 - 1
    ids_norm_particles = ids_norm_particles[None].flip((-1,))

    density_at_particles = sample_vol(density_volume, ids_norm_particles)[0, 0]

    sampled_pts = particles[density_at_particles > density_min_th]
    vol_densities = density_at_particles[density_at_particles > density_min_th]
    final_grid_size = target_grid_size

    write_particles(sampled_pts, 0, model_path, "static")

    return sampled_pts, vol_densities, final_grid_size
