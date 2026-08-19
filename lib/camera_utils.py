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

import warnings

import numpy as np
import torch
from PIL import Image

from scene.cameras import Camera
from lib.general_utils import PILtoTorch
from lib.graphics_utils import fov2focal


def loadCam(args, cam_idx, cam_info, resolution_scale, bg_cache=None):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1920:
                warnings.warn(
                    "Encountered quite large input images (>1.6K pixels width), "
                    "rescaling to 1.6K. If this is not desired, please explicitly "
                    "specify '--resolution/-r' as 1"
                )
                global_down = orig_w / 1920
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(np.round(orig_w / scale)), int(np.round(orig_h / scale)))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    elif cam_info.alpha is not None:
        alpha = Image.fromarray((cam_info.alpha[..., 0] * 255).astype(np.uint8)).resize(
            resolution
        )
        alpha = np.asarray(alpha).astype(float) / 255
        loaded_mask = alpha[None]

    # Build the realistic background plate once per (viewpoint, resolution) and share
    # it by reference across that viewpoint's frames (id() dedups exactly).
    if bg_cache is None:
        bg_cache = {}
    bg_key = (id(cam_info.background), resolution)
    if bg_key not in bg_cache:
        bg_cache[bg_key] = PILtoTorch(cam_info.background, resolution).to(
            args.data_device
        )
    real_background = bg_cache[bg_key]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=cam_info.uid,
        data_device=args.data_device,
        fid=cam_info.fid,
        real_background=real_background,
        white_background=args.white_background,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    bg_cache = {}

    for idx, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, idx, c, resolution_scale, bg_cache))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def create_random_orbit_cameras(
    center: torch.Tensor,
    ref_camera,
    gaussians,
    n_cameras: int = 1,
    initial_distance: float = 2.0,
    max_distance: float = 20.0,
    margin: float = 0.05,
):
    """
    Generate random orbit cameras looking at center: sample a random direction
    on the unit sphere and grow the distance until all Gaussians project inside
    the image bounds (margin is a fraction of image size). Returns Camera list.
    """
    FoVx = ref_camera.FoVx
    FoVy = ref_camera.FoVy
    image_width = ref_camera.image_width
    image_height = ref_camera.image_height

    fx = 0.5 * image_width / np.tan(FoVx / 2)
    fy = 0.5 * image_height / np.tan(FoVy / 2)
    cx = image_width / 2
    cy = image_height / 2

    min_u = margin * image_width
    max_u = (1 - margin) * image_width
    min_v = margin * image_height
    max_v = (1 - margin) * image_height

    xyz_world = gaussians.get_xyz()  # (N, 3)

    # Dummy image, used only for its size
    dummy_image = torch.zeros(3, image_height, image_width, device="cuda")

    cameras = []
    for _ in range(n_cameras):
        direction = torch.randn(3, device=center.device)
        direction = direction / direction.norm()

        distance = initial_distance
        camera = None

        while distance < max_distance:
            cam_pos = center + direction * distance

            # Compute look-at rotation matrix
            forward = center - cam_pos
            forward = forward / forward.norm()

            up = torch.tensor([0.0, -1.0, 0.0], device=center.device)
            if torch.abs(forward @ up) > 0.99:
                up = torch.tensor([1.0, 0.0, 0.0], device=center.device)

            right = torch.cross(forward, up)
            right = right / right.norm()

            up = torch.cross(right, forward)
            up = up / up.norm()

            # Rotation matrix: columns are right, -up, forward
            R = torch.stack([right, -up, forward], dim=1)  # (3, 3)

            # Transform to camera space: X_cam = R.T @ (X_world - cam_pos)
            xyz_cam = (xyz_world - cam_pos) @ R  # (N, 3)

            z = xyz_cam[:, 2]
            if (z <= 0).any():
                distance *= 1.2
                continue

            u = fx * xyz_cam[:, 0] / z + cx
            v = fy * xyz_cam[:, 1] / z + cy

            if (
                (u >= min_u).all()
                and (u <= max_u).all()
                and (v >= min_v).all()
                and (v <= max_v).all()
            ):
                # Camera class expects R, T where: X_cam = R.T @ X_world + T
                R_np = R.cpu().numpy()  # (3, 3)
                T_np = (-R.T @ cam_pos).cpu().numpy()  # (3,)

                camera = Camera(
                    colmap_id=0,
                    R=R_np,
                    T=T_np,
                    FoVx=FoVx,
                    FoVy=FoVy,
                    image=dummy_image,
                    gt_alpha_mask=None,
                    image_name="random_cam",
                    uid=0,
                    fid=0.0,
                )
                break

            distance *= 1.2

        if camera is not None:
            cameras.append(camera)

    print(f"Generated {len(cameras)} valid random cameras out of {n_cameras} attempts")
    return cameras
