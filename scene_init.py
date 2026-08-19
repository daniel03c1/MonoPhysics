import os

import numpy as np
import torch
import torchvision
from knn import knn_cross_3d, knn_idx

from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.gaussian_model import BasicPointCloud
from lib.camera_utils import create_random_orbit_cameras
from lib.particle_sampling import (
    build_density_volume,
    estimate_volume_and_voxel_size,
    filter_particles_multiview,
    sample_particles,
)
from lib.sh_utils import SH2RGB


def reference_camera(scene, scale=1.0, fid=0.0):
    """
    The camera whose space the loaded Gaussians live in: lowest eligible UID.
    Never getTrainCameras(...)[0] -- that shuffles, and the per-camera PLY and
    every later c2w must come from the same camera.
    """
    views = scene.getTrainCameras(scale=scale, exact_fid=fid, shuffle=False)
    if not views:
        raise RuntimeError(f"No train camera at fid={fid} for cam_idx={scene.cam_idx}.")
    return min(views, key=lambda v: v.uid)


"""    Scene Init Helpers    """


def load_frame0_gaussians(scene, dataset):
    """
    Load the frame-0 Gaussians from dataset.init_ply, verbatim.

    The PLY must be a metric-scale, pose-aligned 3DGS reconstruction of frame 0,
    expressed in the reference camera's OpenCV frame, with RGB in f_dc (sh_degree 0).
    Everything downstream (opacity filter, density volume, particle sampling) inherits
    this geometry as-is, so the whole pipeline is only as good as the reconstruction
    handed to it. Neither 6-DOF alignment nor per-Gaussian appearance refinement runs
    here -- re-aligning only sank the object through the floor (measured on Hippo cam0:
    2.7% of points below the floor before, 24.8% after).
    """
    aligned_gaussian_dir = os.path.join(dataset.model_path, "aligned_gaussians")
    cache_name = "frame_000.ply"
    aligned_gaussian_path = os.path.join(aligned_gaussian_dir, cache_name)
    if os.path.exists(aligned_gaussian_path):
        print(f"[Aligned Gaussians] Loading cached: {aligned_gaussian_path}")
        gaussians = GaussianModel(0)
        gaussians.load_ply(aligned_gaussian_path)
        return gaussians

    if not dataset.init_ply:
        raise ValueError(
            "No initial Gaussian PLY given. Pass --init_ply <path>; see the "
            "'Initial Gaussians' section of the README for the expected contract."
        )
    if not os.path.exists(dataset.init_ply):
        raise FileNotFoundError(f"Initial Gaussian PLY not found: {dataset.init_ply}")

    # The PLY lives in this camera's frame; a mismatch is otherwise silent.
    camera = reference_camera(scene)
    print(f"[Gaussians] Loading {dataset.init_ply} (reference camera uid {camera.uid})")

    gaussians = GaussianModel(0)
    gaussians.load_ply(dataset.init_ply)

    # Cache the frame-0 Gaussians (also the NN color source for debug)
    os.makedirs(aligned_gaussian_dir, exist_ok=True)
    gaussians.save_ply(aligned_gaussian_path)

    return gaussians


"""    Main Scene Init Function    """


@torch.no_grad()
def prepare_pcd(
    dataset,
    pipeline,
    phys_args,
    image_scale: float = 1.0,
):
    """
    Load the per-camera aligned Gaussians, filter by multi-view consistency,
    build a density volume, and sample particles.
    Returns (particles, densities, camera_info_dict).
    """

    # Create scene for camera access
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        resolution_scales=[image_scale],
        n_train_frames=getattr(phys_args, "n_frames", None),
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras(scale=image_scale)
    fids = torch.unique(torch.stack([view.fid for view in views]))

    # Extract parameters
    grid_size = phys_args.density_grid_size
    num_iter = 4 if phys_args.random_sample else 5
    filling_grid_size = grid_size / 2**5

    cam_info = {
        "cameras": scene.cameras,
        "cameras_extent": scene.cameras_extent,
        "n_train_frames": getattr(phys_args, "n_frames", None),
    }

    # Particle cache: skip load/sample by loading the sampled particles
    # (positions + opacities), saved as a camera-space Gaussian-splat ply.
    particle_cache_dir = os.path.join(dataset.model_path, "particle_cache")
    particle_cache_path = os.path.join(particle_cache_dir, "frame_000.ply")
    if os.path.exists(particle_cache_path):
        print(f"[Particle Cache] Loading cached particles: {particle_cache_path}")
        cached = GaussianModel(0)
        cached.load_ply(particle_cache_path)
        return cached.get_xyz(), cached.get_opacity().squeeze(), cam_info

    gaussians = load_frame0_gaussians(scene=scene, dataset=dataset)

    # Filter by opacity
    xyzt = gaussians.get_xyz()
    opacity = gaussians.get_opacity().squeeze()
    xyzt = xyzt[opacity > phys_args.opacity_threshold]

    # Compute bounding box and initial grid
    bbox_mins = xyzt.amin(dim=0) - grid_size
    bbox_maxs = xyzt.amax(dim=0) + grid_size
    bbox_bounds = bbox_maxs - bbox_mins

    volume_size = torch.round(bbox_bounds / filling_grid_size).to(torch.int64) + 1
    grid_ids = [torch.arange(size) for size in volume_size]
    grid_coords = (
        torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1).reshape(-1, 3)
        * filling_grid_size
    )
    grid_coords = grid_coords.to(xyzt)
    init_inner_points = grid_coords + bbox_mins.reshape(1, 3)

    # Generate random cameras
    center = gaussians.get_xyz().mean(dim=0)
    ref_camera = reference_camera(scene, scale=image_scale, fid=fids[0])
    random_cameras = create_random_orbit_cameras(
        center, ref_camera, gaussians, n_cameras=20
    )

    # Filter particles using multi-view consistency
    init_inner_points, xyzt = filter_particles_multiview(
        init_inner_points, xyzt, random_cameras, gaussians, pipeline, background
    )

    density_volume, curr_grid_size, volume_size, bbox_mins, bbox_bounds = (
        build_density_volume(
            init_inner_points,
            xyzt,
            bbox_mins,
            bbox_bounds,
            grid_size,
            num_iter,
        )
    )

    # Estimate volume and compute effective sampling_grid_size if target specified
    n_init_particles = phys_args.n_init_particles
    total_volume, sampling_grid_size, est_count = estimate_volume_and_voxel_size(
        density_volume,
        curr_grid_size,
        phys_args.density_min_th,
        n_init_particles,
    )

    if sampling_grid_size is not None:
        print(
            f"[Particle Sampling] Target: {n_init_particles}, "
            f"Total volume: {total_volume:.6f}, "
            f"Computed sampling_grid_size: {sampling_grid_size:.6f}"
        )
        phys_args.voxel_size = sampling_grid_size

    # Sample particles (resample at sampling_grid_size if n_init_particles specified)
    vol, vol_densities, final_grid_size = sample_particles(
        density_volume,
        volume_size,
        curr_grid_size,
        bbox_mins,
        bbox_bounds,
        phys_args.density_min_th,
        phys_args.random_sample,
        dataset.model_path,
        sampling_grid_size=sampling_grid_size,
    )

    # Cache the sampled particles (positions + opacities) as a camera-space GS ply.
    # Scales/colors are placeholders -- assign_gs_to_pcd re-derives them; only
    # get_xyz()/get_opacity() are read back on a cache hit.
    os.makedirs(particle_cache_dir, exist_ok=True)
    n_particles = vol.shape[0]
    cache_gs = GaussianModel(0)
    cache_gs._xyz = torch.nn.Parameter(vol.clone())
    cache_gs._opacity = torch.nn.Parameter(
        cache_gs.inverse_opacity_activation(
            vol_densities.reshape(-1, 1).clamp(max=1 - 1e-4)
        )
    )
    cache_gs._features_dc = torch.nn.Parameter(
        torch.zeros(n_particles, 1, 3, device=vol.device)
    )
    cache_gs._features_rest = torch.nn.Parameter(
        torch.zeros(n_particles, 0, 3, device=vol.device)
    )
    cache_gs._scaling = torch.nn.Parameter(
        torch.zeros(n_particles, 1, device=vol.device)
    )
    cache_rots = torch.zeros(n_particles, 4, device=vol.device)
    cache_rots[:, 0] = 1.0
    cache_gs._rotation = torch.nn.Parameter(cache_rots)
    cache_gs.save_ply(particle_cache_path)
    print(f"[Particle Cache] Saved {n_particles} particles: {particle_cache_path}")

    # Visualization only: color particles by nearest loaded Gaussian (not read downstream)
    ref_xyz = gaussians.get_xyz().contiguous()
    part_xyz = vol.to(ref_xyz).contiguous()
    nn_idx = knn_cross_3d(ref_xyz, part_xyz).long()
    colored_gs = GaussianModel(gaussians.max_sh_degree)
    colored_gs._xyz = torch.nn.Parameter(part_xyz.clone())
    colored_gs._features_dc = torch.nn.Parameter(gaussians._features_dc.data[nn_idx])
    colored_gs._features_rest = torch.nn.Parameter(
        gaussians._features_rest.data[nn_idx]
    )
    colored_gs._opacity = torch.nn.Parameter(gaussians._opacity.data[nn_idx])
    colored_gs._scaling = torch.nn.Parameter(gaussians._scaling.data[nn_idx])
    colored_gs._rotation = torch.nn.Parameter(gaussians._rotation.data[nn_idx])
    colored_dir = os.path.join(dataset.model_path, "colored_particles")
    os.makedirs(colored_dir, exist_ok=True)
    frame_name = "frame_000"
    colored_ply_path = os.path.join(colored_dir, f"{frame_name}.ply")
    colored_gs.save_ply(colored_ply_path)
    colored_gs.set_c2w(ref_camera.world_view_transform.inverse().T)
    colored_img = render(
        ref_camera, colored_gs, pipeline, background, torch.zeros_like(colored_gs._xyz)
    )["render"]
    colored_png_path = os.path.join(colored_dir, f"{frame_name}.png")
    torchvision.utils.save_image(colored_img, colored_png_path)
    print(
        f"[Colored Particles] Saved NN-colored particle viz (unused downstream): {colored_png_path}, {colored_ply_path}"
    )

    return vol, vol_densities, cam_info


def assign_gs_to_pcd(xyz, xyz_opacity, dataset, cam_info):
    xyz = xyz.cpu().detach().numpy()
    num_pts = xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset, gaussians, resolution_scales=[1.0], pcd=pcd, cam_info=cam_info
    )
    xyz_opacity = xyz_opacity.reshape(-1, 1)
    scene.gaussians._opacity = torch.nn.Parameter(
        scene.gaussians.inverse_opacity_activation(
            torch.clamp(xyz_opacity, max=1 - 1e-4)
        ).requires_grad_(True)
    )

    points = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
    dist2 = (points - points[knn_idx(points)]).square().sum(dim=-1).clamp(min=1e-7)

    scales = scene.gaussians.scaling_inverse_activation(torch.sqrt(dist2))[..., None]
    scene.gaussians._scaling = torch.nn.Parameter(scales.requires_grad_(True))
    return scene
