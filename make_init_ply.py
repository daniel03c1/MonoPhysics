#!/usr/bin/env python
"""
make_init_ply - image + camera  ->  pose-aligned 3D Gaussians for one view.

Produces the frame-0 PLY that inv_problem.py consumes via --init_ply. SAM 3D
supplies the shape; this script then (1) 6-DOF rigid aligns it against the
camera and (2) refines each Gaussian tangentially, both photometrically through
gsplat -- the same rasterizer the training pipeline uses -- so the written
Gaussians are consumed directly with no further alignment.

The alignment is the part that matters: SAM 3D returns its own canonical pose,
and nothing downstream re-aligns the PLY, so an error here propagates. Check
compare.png -- the red silhouette should cover the green mask.

SAM 3D itself is used through its public Inference API and is never modified.
It needs its own environment and checkpoints; follow
https://github.com/facebookresearch/sam-3d-objects and run this script in that
environment, pointing --sam3d-repo at the checkout.

  python make_init_ply.py --sam3d-repo /path/to/sam-3d-objects \\
      --dataset vid2sim --root /path/to/Vid2Sim --object bell --cam 2 \\
      --out init_ply/bell/cam02

  python make_init_ply.py --sam3d-repo /path/to/sam-3d-objects \\
      --image frame000.png --camera cam.json --out init_ply/my_object/cam00

Writes into --out: gaussians_cam.ply (aligned + refined, the file to pass to
--init_ply) and compare.png, plus gaussians_cam_raw.ply and compare_raw.png from
SAM 3D's native pose so the two can be compared. Each compare image is
GT | render | overlay, with the render silhouette in red over the mask in green.

ADDING YOUR OWN DATASET: write a generator that yields View objects and add it
to DATASETS below. A View is just an RGB array, a boolean mask and a 3x3 OpenCV
intrinsics matrix. For a one-off, skip the adapter and use --image/--camera.

MANY VIEWS AT ONCE: one run = one view, and loading SAM 3D dominates that run,
so a shell loop over this script pays for the model load every time. For a whole
dataset, load once and reuse the pipeline:

  from make_init_ply import DATASETS, align_view, load_sam3d
  pipe, scene_visualizer = load_sam3d("/path/to/sam-3d-objects")
  for view in DATASETS["vid2sim"]("/path/to/Vid2Sim"):
      out = f"init_ply/{view.object}/cam{view.cam:02d}"
      os.makedirs(out, exist_ok=True)
      align_view(pipe, scene_visualizer, view, out)

align_view() keeps no state between calls beyond what it is handed, so this is
equivalent to running the script once per view -- only without the reloads.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

# ----------------------------- CONSTANTS -------------------------------------
SEED = 42
FRAME = 0

# 6-DOF rigid photometric alignment (rotate positions about the COM + translate)
ALIGN_ITERS = 200  # Adam steps
ALIGN_LR = 0.05  # Adam learning rate (both translation and rotation groups)
LAMBDA_DSSIM = 0.2  # SSIM weight in the image loss (L1 gets 1 - LAMBDA_DSSIM)

# tangential-only refinement (per-Gaussian, camera-depth gradient frozen)
REFINE_ITERS = 1000  # Adam steps
FEATURE_LR = 0.01  # SH DC (color) learning rate
SCALING_LR = 2e-4  # log-scale learning rate
OPACITY_LR = 0.01  # logit-opacity learning rate
POSITION_LR = 2e-4  # tangential position learning rate
BG_DEFAULT = 1.0  # solid background (1.0 white / 0.0 black); must match the framework
MASK_THR = 127  # 0-255 threshold when reading mask PNGs
COLOR_DC = 0.28209479177387814  # SH degree-0 -> RGB constant
DEV = "cuda"


@dataclass
class View:
    """One masked image with its intrinsics: all a dataset adapter must produce."""

    object: str
    cam: int
    rgb: np.ndarray  # HxWx3 uint8
    mask: np.ndarray  # HxW bool
    K: np.ndarray  # 3x3 intrinsics (pixels)


# ------------------------------- helpers -------------------------------------
def affine_fit(src, dst):
    """Least-squares 3x4 affine map taking src points onto dst points."""
    src, dst = src.double(), dst.double()
    X = torch.cat([src, torch.ones(len(src), 1, dtype=torch.double)], 1)
    return torch.linalg.lstsq(X, dst).solution[:3].T


def orthonormalize(A):
    """Nearest rotation matrix to A, via SVD (reflections flipped out)."""
    U, _, Vt = torch.linalg.svd(A)
    if torch.det(U @ Vt) < 0:
        U = U.clone()
        U[:, -1] *= -1
    return U @ Vt


# Photometric losses, kept identical to lib/loss_utils.py (same 11-px Gaussian
# window, C1=0.01^2, C2=0.03^2) so this objective matches the training
# pipeline's exactly. Vendored rather than imported because lib/ pulls in the
# compiled knn extension, which does not exist in the SAM 3D environment.
# Operate on [3, H, W] tensors.
SSIM_WINDOW_CACHE = {}


def gaussian(window_size, sigma):
    g = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return g / g.sum()


def create_window(window_size, channel):
    kernel_1d = gaussian(window_size, 1.5).unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11):
    channel = img1.size(-3)
    key = (window_size, channel, img1.dtype, img1.device)
    window = SSIM_WINDOW_CACHE.get(key)
    if window is None:
        window = create_window(window_size, channel).type_as(img1)
        SSIM_WINDOW_CACHE[key] = window

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def l1_loss(network_output, gt):
    return (network_output - gt).abs().mean()


def image_loss_value(image, gt_image, lambda_dssim):
    """L1 + SSIM on [3, H, W] images (lambda_dssim weights the SSIM term)."""
    return (1.0 - lambda_dssim) * l1_loss(image, gt_image) + lambda_dssim * (
        1.0 - ssim(image, gt_image)
    )


def overlay(rgb, silhouette, mask):
    """Tint the ground-truth mask green and the rendered silhouette red."""
    out = rgb.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    out[silhouette] = (0.5 * out[silhouette] + 0.5 * np.array([255, 0, 0])).astype(
        np.uint8
    )
    return out


def write_gs_ply(path, means, quats, scales, opacities, sh_dc):
    """
    Write Gaussians in the conventional 3D Gaussian Splatting PLY layout
    (INRIA / SAM 3D `GaussianModel.save_ply`): a binary `vertex` element with
    x y z | nx ny nz | f_dc_0..2 | opacity | scale_0..2 | rot_0..3.

    `scales`/`opacities` are the *activated* values straight off the model; the
    3DGS file format stores their pre-activation form, so they are written as
    log(scale) and logit(opacity) - a standard loader (exp / sigmoid) recovers
    the originals. `sh_dc` is the raw degree-0 SH coefficient (not RGB), and
    quaternions are real-part-first (w, x, y, z), matching the 3DGS convention.
    """
    means = means.detach().cpu().numpy().astype(np.float32)
    quats = quats.detach().cpu().numpy().astype(np.float32)  # (w, x, y, z)
    f_dc = sh_dc.detach().cpu().numpy().astype(np.float32)  # raw SH DC, not RGB
    normals = np.zeros_like(means)
    scale = np.log(np.clip(scales.detach().cpu().numpy(), 1e-12, None)).astype(
        np.float32
    )
    opacity = np.clip(opacities.detach().cpu().numpy().reshape(-1, 1), 1e-6, 1 - 1e-6)
    opacity = np.log(opacity / (1.0 - opacity)).astype(np.float32)  # inverse sigmoid

    attrs = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    data = np.concatenate([means, normals, f_dc, opacity, scale, quats], axis=1)
    element = np.empty(len(means), dtype=[(a, "f4") for a in attrs])
    element[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(element, "vertex")]).write(path)


# --------------------------- dataset adapters --------------------------------
# Each yields uniform `View`s. Add a dataset = add one generator + a DATASETS
# entry. Anything not covered here can go through --image/--camera instead.
def iter_generic(image, camera, mask=None):
    """
    One view from an image plus a small camera JSON.

    The image is RGBA with alpha as the object mask, or RGB with a separate
    `mask` PNG. The JSON gives the intrinsics as either "K" (3x3) or
    "camera_angle_x" (horizontal FOV in radians, NeRF style).
    """
    if mask is not None:
        rgb = np.array(Image.open(image).convert("RGB"))
        object_mask = np.array(Image.open(mask).convert("L")) > MASK_THR
    else:
        rgba = np.array(Image.open(image).convert("RGBA"))
        rgb, object_mask = rgba[..., :3], rgba[..., 3] > MASK_THR
        if not object_mask.any():
            raise SystemExit(
                f"{image} has an empty alpha channel; pass --mask, or supply an "
                "RGBA image whose alpha is the object mask."
            )
    height, width = object_mask.shape

    camera_cfg = json.load(open(camera))
    if "K" in camera_cfg:
        K = np.array(camera_cfg["K"], float)
    else:
        # NeRF-style horizontal FOV: square pixels, principal point at center.
        focal = 0.5 * width / np.tan(0.5 * float(camera_cfg["camera_angle_x"]))
        K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], float)

    name = camera_cfg.get("object", os.path.splitext(os.path.basename(image))[0])
    yield View(name, int(camera_cfg.get("cam", 0)), rgb, object_mask, K)


def jukebox_object_views(obj_dir, obj_label):
    """
    Per-camera Views for one jukebox-format object directory (camera.json plus
    images/masks named {cam:03d}_{FRAME:03d}.png).
    """
    cameras = json.load(open(os.path.join(obj_dir, "camera.json")))
    for cam in range(cameras["n_cameras"]):
        image = os.path.join(obj_dir, "images", f"{cam:03d}_{FRAME:03d}.png")
        mask = os.path.join(obj_dir, "masks", f"{cam:03d}_{FRAME:03d}.png")
        if not (os.path.isfile(image) and os.path.isfile(mask)):
            continue
        rgb = np.array(Image.open(image).convert("RGB"))
        object_mask = np.array(Image.open(mask).convert("L")) > MASK_THR
        K = np.array(cameras["cameras"][cam]["intrinsics"], float)
        yield View(obj_label, cam, rgb, object_mask, K)


def iter_jukebox(root):
    for name in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, name)
        if not os.path.isfile(os.path.join(obj_dir, "camera.json")):
            continue
        yield from jukebox_object_views(obj_dir, name)


def iter_vid2sim(root):
    # transforms_train.json: entry i == camera i, all at time 0 already, so
    # FRAME does not apply. The `a_` image variant carries the object alpha;
    # `m_` is RGB-only, `r_` is full-frame.
    for name in sorted(os.listdir(root)):
        transforms = os.path.join(root, name, "transforms_train.json")
        if not os.path.isfile(transforms):
            continue
        scene = json.load(open(transforms))
        for cam, entry in enumerate(scene["frames"]):
            image = os.path.join(root, name, entry["file_path"].replace("m_", "a_"))
            image += ".png"
            if not os.path.isfile(image):
                continue
            rgba = np.array(Image.open(image).convert("RGBA"))
            rgb, object_mask = rgba[..., :3], rgba[..., 3] > MASK_THR
            height, width = object_mask.shape
            focal = 0.5 * width / np.tan(0.5 * scene["camera_angle_x"])
            K = np.array(
                [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], float
            )
            yield View(name, cam, rgb, object_mask, K)


def read_colmap_cameras_bin(path):
    """Minimal COLMAP cameras.bin reader -> {camera_id: (model_id, W, H, params)}."""
    import struct

    n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
    cameras = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cam_id, model = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            params = struct.unpack(
                "<" + "d" * n_params[model], f.read(8 * n_params[model])
            )
            cameras[cam_id] = (model, width, height, np.array(params))
    return cameras


SPRING_GAUS_CAMERAS = ("C0733", "C0787", "C0801")


def iter_springgaus_real(root):
    # SpringGaus real_capture, matching scene/dataset_readers.py: intrinsics come
    # from the static COLMAP scan rescaled to the dynamic videos, with that
    # reader's fx/fy swap (fx <- params[1], fy <- params[0]). Per-camera frame
    # filenames come from dynamic/sequences/<object>/0.json.
    dynamic = os.path.join(root, "dynamic")
    for name in sorted(os.listdir(os.path.join(dynamic, "sequences"))):
        sequence_json = os.path.join(dynamic, "sequences", name, "0.json")
        colmap_bin = os.path.join(root, "static", "colmap", name, "cameras.bin")
        if not (os.path.isfile(sequence_json) and os.path.isfile(colmap_bin)):
            continue
        sequence = json.load(open(sequence_json))
        _, static_w, static_h, params = read_colmap_cameras_bin(colmap_bin)[1]
        for cam, cam_name in enumerate(SPRING_GAUS_CAMERAS):
            if cam_name not in sequence or FRAME >= len(sequence[cam_name]):
                continue
            filename = sequence[cam_name][FRAME]
            image = os.path.join(dynamic, "videos_images", cam_name, filename)
            mask = os.path.join(
                dynamic, "videos_masks", cam_name, filename.replace(".jpg", ".png")
            )
            if not (os.path.isfile(image) and os.path.isfile(mask)):
                continue
            rgb = np.array(Image.open(image).convert("RGB"))
            object_mask = np.array(Image.open(mask).convert("L")) > MASK_THR
            height, width = object_mask.shape
            fx, fy = params[1] * width / static_w, params[0] * height / static_h
            K = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], float)
            yield View(name, cam, rgb, object_mask, K)


DATASETS = {
    "vid2sim": iter_vid2sim,
    "jukebox": iter_jukebox,
    "springgaus_real": iter_springgaus_real,
}


# --------------------------- SAM 3D + alignment ------------------------------
def load_sam3d(sam3d_repo, pipeline_yaml=None):
    """
    Import SAM 3D from a checkout and build its inference pipeline.

    Imported lazily so --help works without the SAM 3D environment. SAM 3D is
    used through its public API only; nothing in the checkout is modified.
    """
    sam3d_repo = os.path.abspath(os.path.expanduser(sam3d_repo))
    if not os.path.isdir(sam3d_repo):
        raise SystemExit(
            f"SAM 3D checkout not found at {sam3d_repo}. Clone it and pass "
            "--sam3d-repo (or set SAM3D_REPO):\n\n"
            "    git clone https://github.com/facebookresearch/sam-3d-objects\n"
        )
    yaml_path = pipeline_yaml or os.path.join(
        sam3d_repo, "checkpoints", "hf", "pipeline.yaml"
    )
    if not os.path.isfile(yaml_path):
        raise SystemExit(
            f"SAM 3D pipeline config not found at {yaml_path}. Download the "
            "checkpoints as described in the SAM 3D repo, or pass --pipeline-yaml."
        )

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    sys.path.insert(0, sam3d_repo)
    sys.path.append(os.path.join(sam3d_repo, "notebook"))

    from inference import Inference  # noqa: E402  (SAM 3D checkout)
    from sam3d_objects.utils.visualization import SceneVisualizer  # noqa: E402

    return Inference(yaml_path, compile=False)._pipeline, SceneVisualizer


def sam3d_gaussians(pipe, scene_visualizer, view):
    """
    Run SAM 3D on one masked image and return its Gaussians in the camera frame.

    SAM 3D emits a shape in its own canonical pose; the rigid transform that
    takes it to the camera is read back from the decoder layout by fitting an
    affine map between the canonical points and the posed point cloud. That is
    only a starting point -- align_view() refines it against the image, which is
    the step that actually makes the render line up.
    """
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    pytorch3d_to_opencv = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))

    rgba = np.concatenate(
        [view.rgb, (view.mask[..., None] * 255).astype(np.uint8)], axis=-1
    )
    sam3d_out = pipe.run(
        rgba,
        None,
        seed=SEED,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
    )
    gaussians = sam3d_out["gaussian"][0]
    xyz_canonical = gaussians.get_xyz.detach().cpu()
    xyz_centered = (xyz_canonical - xyz_canonical.mean(0)).to(DEV)
    quats_canonical = gaussians.get_rotation.detach().to(DEV)
    scales_canonical = gaussians.get_scaling.detach().to(DEV)
    opacities = gaussians.get_opacity.detach().squeeze(-1).to(DEV)
    sh_dc = gaussians.get_features.detach()[:, 0, :].to(DEV)  # raw degree-0 SH

    posed = (
        scene_visualizer.object_pointcloud(
            xyz_canonical.unsqueeze(0),
            sam3d_out["rotation"].cpu(),
            sam3d_out["translation"].cpu(),
            sam3d_out["scale"].cpu(),
        )
        .points_list()[0]
        .cpu()
    )
    cam_pts = posed @ pytorch3d_to_opencv.T
    affine = affine_fit(xyz_canonical, cam_pts)
    rotation = orthonormalize(affine).float().to(DEV)
    scale = float((torch.det(affine).abs() ** (1.0 / 3)).item())
    translation = cam_pts.float().mean(0).to(DEV)

    means = (xyz_centered @ rotation.T) * scale + translation
    quats = matrix_to_quaternion(rotation @ quaternion_to_matrix(quats_canonical))
    return means, quats, scales_canonical * scale, opacities, sh_dc


def align_view(pipe, scene_visualizer, view, out_dir, bg=BG_DEFAULT):
    """
    SAM 3D shape -> 6-DOF rigid align + tangential refinement -> written PLYs.

    `bg` is the solid background the loss composites on and must match the
    training config's white_background. Writes gaussians_cam.ply (aligned and
    refined) and gaussians_cam_raw.ply (SAM 3D's pose), each with a compare
    image beside it.
    """
    import gsplat
    from pytorch3d.transforms import quaternion_to_matrix

    height, width = view.mask.shape
    gt_mask = torch.from_numpy(view.mask.astype(np.float32)).to(DEV)
    Ks = torch.from_numpy(view.K.astype(np.float32)).to(DEV)[None]

    # ---- A) SHAPE: SAM 3D (frozen), placed in the camera frame ----
    means_raw, quats, scales_raw, opacities_raw, sh_raw = sam3d_gaussians(
        pipe, scene_visualizer, view
    )
    colors_raw = (sh_raw * COLOR_DC + 0.5).clamp(0, 1)

    # ---- B) ALIGN + REFINE (photometric, through gsplat) ----
    # GT target composited on the solid background, matching the training
    # pipeline (img*alpha + bg*(1-alpha)); the render below uses the same bg.
    gt_image = (
        (
            (torch.from_numpy(view.rgb).float().to(DEV) / 255.0) * gt_mask[..., None]
            + float(bg) * (1.0 - gt_mask[..., None])
        )
        .permute(2, 0, 1)
        .contiguous()
    )  # [3, H, W]

    def render(means, quats, scales, opacities, colors):
        # gsplat "RGB" returns premultiplied color (over black) + accumulated
        # alpha; compositing over a solid bg is col + (1 - alpha) * bg.
        color, alpha, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.eye(4, device=DEV)[None],
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
        )
        alpha = alpha[0, ..., 0]  # [H, W]
        image = color[0] + (1.0 - alpha)[..., None] * float(bg)  # [H, W, 3]
        return image.permute(2, 0, 1), alpha  # ([3, H, W], [H, W])

    def photometric_loss(image, alpha):
        return image_loss_value(image, gt_image, LAMBDA_DSSIM) + l1_loss(alpha, gt_mask)

    # B1) 6-DOF rigid align: rotate POSITIONS about the center of mass and
    # translate. Per-Gaussian orientations are left unchanged, faithful to the
    # training pipeline, which also updates only positions during alignment.
    com = means_raw.mean(0)
    delta_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEV, requires_grad=True)
    delta_t = torch.zeros(3, device=DEV, requires_grad=True)
    optimizer = torch.optim.Adam(
        [{"params": [delta_t], "lr": ALIGN_LR}, {"params": [delta_q], "lr": ALIGN_LR}]
    )
    best_loss = float("inf")
    best_q, best_t = delta_q.detach().clone(), delta_t.detach().clone()
    for _ in range(ALIGN_ITERS):
        optimizer.zero_grad()
        rotation = quaternion_to_matrix(F.normalize(delta_q, dim=0))
        means = (means_raw - com) @ rotation.T + com + delta_t
        loss = photometric_loss(
            *render(means, quats, scales_raw, opacities_raw, colors_raw)
        )
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_q, best_t = delta_q.detach().clone(), delta_t.detach().clone()

    best_rotation = quaternion_to_matrix(F.normalize(best_q, dim=0))
    means_aligned = ((means_raw - com) @ best_rotation.T + com + best_t).detach()

    # B2) Tangential per-Gaussian refinement in raw parameter space (the
    # training pipeline's learning rates). The camera-depth gradient of
    # positions is frozen, so points move only in the image plane; rotations and
    # higher-order SH are not touched.
    means_p = means_aligned.clone().requires_grad_(True)
    sh_p = sh_raw.clone().requires_grad_(True)
    log_scales_p = scales_raw.clamp_min(1e-12).log().requires_grad_(True)
    logit_opacities_p = torch.logit(opacities_raw.clamp(1e-6, 1 - 1e-6)).requires_grad_(
        True
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [sh_p], "lr": FEATURE_LR},
            {"params": [log_scales_p], "lr": SCALING_LR},
            {"params": [logit_opacities_p], "lr": OPACITY_LR},
            {"params": [means_p], "lr": POSITION_LR},
        ]
    )
    for _ in range(REFINE_ITERS):
        optimizer.zero_grad()
        colors = (sh_p * COLOR_DC + 0.5).clamp(0, 1)
        loss = photometric_loss(
            *render(
                means_p,
                quats,
                log_scales_p.exp(),
                torch.sigmoid(logit_opacities_p),
                colors,
            )
        )
        loss.backward()
        means_p.grad[:, 2] = 0.0  # freeze camera-space depth (tangential only)
        optimizer.step()

    means_ref = means_p.detach()
    scales_ref = log_scales_p.exp().detach()
    opacities_ref = torch.sigmoid(logit_opacities_p).detach()
    sh_ref = sh_p.detach()
    colors_ref = (sh_ref * COLOR_DC + 0.5).clamp(0, 1)

    # ---- C) OUTPUT: refined and raw Gaussians, each with a compare image ----
    def save(tag, means, scales, opacities, sh, colors):
        write_gs_ply(
            os.path.join(out_dir, f"gaussians_cam{tag}.ply"),
            means,
            quats,
            scales,
            opacities,
            sh,
        )
        with torch.no_grad():
            image, alpha = render(means, quats, scales, opacities, colors)
        rendered = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8
        )
        silhouette = alpha.cpu().numpy() > 0.5
        panels = [view.rgb, rendered, overlay(view.rgb, silhouette, view.mask)]
        Image.fromarray(np.concatenate(panels, axis=1)).save(
            os.path.join(out_dir, f"compare{tag}.png")
        )

    save("", means_ref, scales_ref, opacities_ref, sh_ref, colors_ref)
    save("_raw", means_raw, scales_raw, opacities_raw, sh_raw, colors_raw)


# -------------------------------- driver -------------------------------------
def select_view(args):
    """Resolve the CLI down to exactly one View."""
    if args.image:
        return next(iter_generic(args.image, args.camera, args.mask))

    views = [
        view
        for view in DATASETS[args.dataset](args.root)
        if (args.object in (None, view.object, os.path.basename(view.object)))
        and (args.cam in (None, view.cam))
    ]
    if not views:
        raise SystemExit(
            f"No view matched (dataset={args.dataset}, object={args.object}, "
            f"cam={args.cam}). Check --root, or run without --object/--cam to "
            "see what is available."
        )
    if len(views) > 1:
        found = ", ".join(f"{view.object}/cam{view.cam:02d}" for view in views[:10])
        raise SystemExit(
            f"{len(views)} views matched; this script aligns one view per run. "
            f"Narrow it with --object/--cam. Matched: {found}"
            + (" ..." if len(views) > 10 else "")
        )
    return views[0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_argument_group(
        "input (either a dataset view or an image + camera)"
    )
    source.add_argument("--dataset", choices=sorted(DATASETS), help="built-in adapter")
    source.add_argument("--root", help="dataset root, with --dataset")
    source.add_argument("--object", help="object/scene name, with --dataset")
    source.add_argument("--cam", type=int, help="camera index, with --dataset")
    source.add_argument("--image", help="RGBA image (alpha = mask), or RGB with --mask")
    source.add_argument("--mask", help="mask PNG (>127 = object), if --image is RGB")
    source.add_argument("--camera", help="camera JSON, with --image")

    parser.add_argument("--out", required=True, help="output directory for this view")
    parser.add_argument(
        "--sam3d-repo",
        default=os.environ.get("SAM3D_REPO", "sam-3d-objects"),
        help="SAM 3D checkout (default: $SAM3D_REPO, else ./sam-3d-objects)",
    )
    parser.add_argument(
        "--pipeline-yaml",
        help="SAM 3D pipeline config "
        "(default: <sam3d-repo>/checkpoints/hf/pipeline.yaml)",
    )
    parser.add_argument(
        "--bg",
        type=float,
        default=BG_DEFAULT,
        help="solid background for the photometric loss (1.0 white / 0.0 black); "
        "must match the training config's white_background",
    )
    args = parser.parse_args()

    if bool(args.image) == bool(args.dataset):
        parser.error("give either --dataset (with --root) or --image (with --camera)")
    if args.image and not args.camera:
        parser.error("--image requires --camera")
    if args.dataset and not args.root:
        parser.error("--dataset requires --root")

    view = select_view(args)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pipe, scene_visualizer = load_sam3d(args.sam3d_repo, args.pipeline_yaml)

    os.makedirs(args.out, exist_ok=True)
    align_view(pipe, scene_visualizer, view, args.out, bg=args.bg)
    print(f"[make_init_ply] {view.object}/cam{view.cam:02d} -> {args.out}")


if __name__ == "__main__":
    main()
