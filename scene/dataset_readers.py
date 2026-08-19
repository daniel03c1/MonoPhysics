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

import cv2
import json
import numpy as np
import os
import sys
from PIL import Image
from glob import glob
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from typing import NamedTuple, Optional, Union

from scene.colmap_loader import (
    qvec2rotmat,
    read_cameras_binary,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_images_binary,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from lib.graphics_utils import focal2fov, fov2focal, getWorld2View2
from lib.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    background: Union[int, float, np.array]
    alpha: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            fid=fid,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])

    os.makedirs(os.path.split(path)[0], exist_ok=True)
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame["time"]

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    fid=frame_time,
                    background=1.0 if white_background else 0.0,
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # Random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def get_camera_center(camera):
    R = camera.R.T
    T = camera.T
    RT = np.concatenate([R, T.reshape(3, 1)], -1)
    W2C = np.concatenate([RT, np.array([0, 0, 0, 1.0]).reshape(1, 4)], 0)
    C2W = np.linalg.inv(W2C)
    return C2W[:3, -1]


def readSpringGausCaptureRealInfo(path, white_background, eval, extension=".png"):
    root_dir = path.split("/")
    obj_name = root_dir[-1]
    root_dir = "/".join(root_dir[:-1])

    static_cam_infos, dynamic_cam_infos = readSpringGausCaptureRealCameras(
        root_dir, obj_name, white_background
    )

    train_cam_infos = dynamic_cam_infos  # static_cam_infos + dynamic_cam_infos
    test_cam_infos = train_cam_infos if eval else []
    nerf_normalization = getNerfppNorm(dynamic_cam_infos)  # static_cam_infos)

    num_pts = 100_000
    # Compute cube centered at point closest to optical axes
    centers = np.stack(list(map(get_camera_center, dynamic_cam_infos)), axis=0)
    # Optical axis directions in world coordinates
    dirs = []
    for cam in dynamic_cam_infos:
        R_cam = cam.R.T
        T_cam = cam.T
        RT = np.concatenate([R_cam, T_cam.reshape(3, 1)], axis=-1)
        W2C = np.concatenate([RT, np.array([0, 0, 0, 1.0]).reshape(1, 4)], axis=0)
        C2W = np.linalg.inv(W2C)
        d = C2W[:3, 2]
        dirs.append(d / np.linalg.norm(d))
    dirs = np.stack(dirs, axis=0)

    # Solve for point minimizing distance to all rays
    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in zip(centers, dirs):
        Ai = I - np.outer(d, d)
        A += Ai
        b += Ai @ C
    closest_point = np.linalg.solve(A, b)
    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    bbox_dims = bbox_max - bbox_min
    max_dim = bbox_dims.max()
    side_len = max_dim * 0.3
    xyz = (np.random.random((num_pts, 3)) - 0.5) * side_len + closest_point
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )
    ply_path = os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


def readSpringGausCaptureRealCameras(root_dir, obj_name, white_background):
    # static scene
    root_dir = Path(root_dir)
    static_dir = root_dir / "static"

    camdata = read_cameras_binary(static_dir / "colmap" / obj_name / "cameras.bin")
    imdata = read_images_binary(static_dir / "colmap" / obj_name / "images.bin")

    imdata = sorted(imdata.items(), reverse=False)

    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    static_cams = []

    # dynamic scene
    dynamic_dir = root_dir / "dynamic"
    cam_name = ["C0733", "C0787", "C0801"]

    if obj_name in ["burger", "dog"]:
        no_checkerboard = True
    else:  # ['bun', 'pig', 'potato']
        no_checkerboard = False

    backgrounds = []
    for name in cam_name:
        background = Image.open(
            os.path.join(
                root_dir,
                "dynamic",
                "video_uncompressed",
                name,
                "20800.jpg" if no_checkerboard else "00000.jpg",
            )
        )
        """
        background = 1.0 if white_background else 0.0
        """
        backgrounds.append(background)

    H, W, H_S, W_S = 1080, 1920, 2672, 4752

    FovX = focal2fov(camdata[1].params[1] * W / W_S, W)
    FovY = focal2fov(camdata[1].params[0] * H / H_S, H)

    with open(dynamic_dir / "cameras_calib.json", "r") as f:
        cam_calib = json.load(f)

    with open(dynamic_dir / "sequences" / obj_name / "0.json", "r") as fff:
        seq_info = json.load(fff)

    hit_frame = seq_info["hit_frame"]
    n_frames = len(seq_info[cam_name[0]])
    dynamic_cams = []

    for frame_id in tqdm(range(n_frames)):
        for cam_id, camera in enumerate(cam_name):
            rvecs = cam_calib[camera]["rvecs"]
            tvecs = cam_calib[camera]["tvecs"]

            rot_mat, _ = cv2.Rodrigues(np.array(rvecs))
            R = rot_mat
            T = np.array(tvecs).reshape(3)

            # SfM extrinsics are mis-scaled (effective gravity ~13.0); rescale to a reasonable scale
            W2C = np.eye(4)
            W2C[:3, :3] = R
            W2C[:3, 3] = T
            C2W = np.linalg.inv(W2C)
            C2W[:3, 3] *= 13.0 / 9.8

            W2C = np.linalg.inv(C2W)
            R = W2C[:3, :3]
            T = W2C[:3, 3]

            image_path = (
                dynamic_dir / "videos_images" / camera / seq_info[camera][frame_id]
            )
            mask_path = Path(
                str(image_path)
                .replace("/videos_images/", "/videos_masks/")
                .replace(".jpg", ".png")
            )
            image = Image.open(image_path)
            im_data = np.array(image)
            mask = Image.open(mask_path)
            mask = np.array(mask)[:, :, np.newaxis] / 255.0

            dynamic_cams.append(
                CameraInfo(
                    uid=cam_id,
                    fid=frame_id / 120,
                    R=np.transpose(R),
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=seq_info[camera][frame_id],
                    width=W,
                    height=H,
                    alpha=mask,
                    background=backgrounds[cam_id],
                )
            )

    return static_cams, dynamic_cams


def readVid2SimInfo(path, config_path, white_background):
    with open(os.path.join(config_path), "r") as f:
        cfg = json.load(f)

    print("Reading Vid2Sim data")
    # One pool of all cameras/frames; train = all cameras at the first n_frames,
    # test = everything. cam_idx picks the training camera downstream.
    cam_infos = readVid2SimData(path, white_background)

    fids = list(sorted(set([cam_info.fid for cam_info in cam_infos])))
    max_fid = fids[cfg["physics"]["n_frames"] - 1]
    train_cam_infos = [c for c in cam_infos if c.fid <= max_fid]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # No initial point cloud: see readJukeboxInfo.
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
    )
    return scene_info


def readVid2SimData(path, white_background, total_frames=24, dt=0.05):
    """
    Load every camera (training-pose + NVS) at the frames it actually has on
    disk, as one pool. Train/eval is selected downstream by cam_idx + frame (like
    the other datasets); NVS cameras simply lack the future frames, so the
    existence check naturally excludes them there.
    """
    cam_infos = []

    with open(os.path.join(path, "transforms_train.json")) as json_file:
        infos = json.load(json_file)

    with open(os.path.join(path, "transforms_train_nvs.json")) as json_file:
        infos_nvs = json.load(json_file)

    fov = infos["camera_angle_x"]
    cameras = infos["frames"] + infos_nvs["frames"]

    for cam in tqdm(cameras):
        cam_id = int(cam["file_path"].split(os.path.sep)[-1].split("_")[1])

        c2w = np.array(cam["transform_matrix"])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        for f in range(total_frames):
            image_path = os.path.join(path, "data", f"a_{cam_id}_{f}.png")
            if not os.path.exists(image_path):
                continue
            image_name = Path(image_path).stem
            image = np.asarray(Image.open(image_path))

            bg = 1.0 if white_background else 0.0
            norm_data = image[..., :3] / 255.0
            mask = (image[..., -1:] > 0).astype(float)
            arr = norm_data * mask + bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            cam_infos.append(
                CameraInfo(
                    uid=cam_id,
                    R=R,
                    T=T,
                    FovY=fov,
                    FovX=fov,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    fid=dt * f,
                    alpha=mask,
                    background=1.0 if white_background else 0.0,
                )
            )

    return cam_infos


def readPACNeRFInfo(
    path,
    config_path,
    white_background,
    eval_cam_id=0,
    load_fix_pcd=False,
    read_cam=True,
):
    with open(os.path.join(config_path), "r") as f:
        cfg = json.load(f)
    print("Reading data")
    if read_cam:
        cam_infos = readCamerasFromAllData(path, white_background)

        eval_cam_infos = cam_infos
        fids = list(sorted(set([cam_info.fid for cam_info in cam_infos])))
        max_fid = fids[cfg["physics"]["n_frames"] - 1]
        cam_infos = [c for c in cam_infos if c.fid <= max_fid]

        nerf_normalization = getNerfppNorm(cam_infos)
    else:
        eval_cam_infos = []
        cam_infos = []
        nerf_normalization = {}

    # No initial point cloud: see readJukeboxInfo.
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=cam_infos,
        test_cameras=eval_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
    )
    return scene_info


def readCamerasFromAllData(path, white_background):
    cam_infos = []

    with open(os.path.join(path, "all_data.json")) as json_file:
        frames = json.load(json_file)

        for idx, frame in enumerate(tqdm(frames)):
            cam_id, frame_id = (
                frame["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")
            )
            if frame_id == "-1":
                continue

            file_path = frame["file_path"].replace("r_", "m_")
            image_path = os.path.join(path, file_path)
            image_name = Path(image_path).stem
            image = np.asarray(Image.open(image_path))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = image / 255.0
            mask = (image.astype(int).sum(-1, keepdims=True) != 255 * 3).astype(float)
            arr = norm_data * mask + bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            frame_time = frame["time"]

            c2w = frame["c2w"]
            c2w.append([0.0, 0.0, 0.0, 1.0])
            matrix = np.linalg.inv(np.array(c2w))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            intrinsic = frame["intrinsic"]
            ori_h, ori_w = image.size[0], image.size[1]
            fovy = focal2fov(intrinsic[1][1], ori_h)
            fovx = focal2fov(intrinsic[0][0], ori_w)
            FovY = fovx
            FovX = fovy
            if ori_w != 800:
                image = image.resize((800, 800), Image.BILINEAR)
                if white_background:
                    mask = (
                        np.asarray(image).astype(int).sum(-1, keepdims=True) != 255 * 3
                    ).astype(float)
                else:
                    mask = (
                        np.asarray(image).astype(int).sum(-1, keepdims=True) != 0
                    ).astype(float)

            cam_infos.append(
                CameraInfo(
                    uid=int(cam_id),
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    fid=frame_time,
                    alpha=mask,
                    background=1.0 if white_background else 0.0,
                )
            )

    return cam_infos


def readSpringGausMPMSyntheticInfo(
    path, config_path, white_background, num_frame, eval_cam_id=0
):
    with open(os.path.join(config_path), "r") as f:
        cfg = json.load(f)
    print("Reading data")
    cam_infos = readCamerasFromFrameAndCamera(path, white_background)

    eval_cam_infos = cam_infos

    fids = list(sorted(set([cam_info.fid for cam_info in cam_infos])))
    max_fid = fids[cfg["physics"]["n_frames"] - 1]
    cam_infos = [c for c in cam_infos if c.fid <= max_fid]

    nerf_normalization = getNerfppNorm(cam_infos)

    # No initial point cloud: see readJukeboxInfo.
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=cam_infos,
        test_cameras=eval_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
    )
    return scene_info


def readCamerasFromFrameAndCamera(path, white_background):
    path = Path(path)
    cam_infos = []

    with open(path / "camera.json") as json_file:
        cam_list = json.load(json_file)
    with open(path / "frame.json") as json_file:
        fid_list = json.load(json_file)

    for cam_path in tqdm(path.glob("camera_*")):
        if not os.path.isdir(cam_path):
            continue

        for d in cam_list:
            if d["camera"] == cam_path.name:
                c2w = d["c2w"]
                intrinsic = d["K"]
                break
        c2w = np.asarray(c2w)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        cam_id = int(cam_path.name.split("_")[-1])
        for image_path in cam_path.glob("*"):
            img_id = int(image_path.name.split(".")[0])
            fid = list(fid_list[img_id].values())[0]

            image_name = image_path.stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]
            arr = norm_data[:, :, :3] * mask + bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            fovy = focal2fov(intrinsic[1][1], image.size[0])
            fovx = focal2fov(intrinsic[0][0], image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(
                CameraInfo(
                    uid=cam_id,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    fid=fid,
                    alpha=mask,
                    background=1.0 if white_background else 0.0,
                )
            )

    return cam_infos


def readJukeboxCameras(path, white_background, fps):
    path = Path(path)
    cam_infos = []

    with open(path / "camera.json") as f:
        cam_data = json.load(f)

    n_cameras = cam_data["n_cameras"]
    width = cam_data["width"]
    height = cam_data["height"]
    cameras = cam_data["cameras"]

    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

    for cam_id in range(n_cameras):
        cam = cameras[cam_id]
        K = np.array(cam["intrinsics"])
        fx, fy = K[0, 0], K[1, 1]
        c2w_per_frame = cam["c2w"]
        n_frames = len(c2w_per_frame)

        # fx/fy swapped, as in the SpringGaus reader and make_init_ply: the
        # three must share one convention, so changing it here alone breaks them.
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        for frame_id in range(n_frames):
            c2w = np.array(c2w_per_frame[frame_id])
            # Jukebox c2w is already OpenCV convention — no flip needed
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            img_name = f"{cam_id:03d}_{frame_id:03d}"
            image_path = path / "images" / f"{img_name}.png"
            mask_path = path / "masks" / f"{img_name}.png"

            image = Image.open(image_path)
            im_data = np.array(image.convert("RGB")) / 255.0

            mask = np.array(Image.open(mask_path).convert("L")) / 255.0
            mask = mask[:, :, np.newaxis]

            arr = im_data * mask + bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            fid = frame_id / fps

            cam_infos.append(
                CameraInfo(
                    uid=cam_id,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=str(image_path),
                    image_name=img_name,
                    width=width,
                    height=height,
                    fid=fid,
                    alpha=mask,
                    background=1.0 if white_background else 0.0,
                )
            )

    return cam_infos


def readJukeboxInfo(path, config_path, white_background):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    fps = cfg["physics"]["fps"]
    n_frames = cfg["physics"]["n_frames"]

    print("Reading Jukebox data")
    cam_infos = readJukeboxCameras(path, white_background, fps)

    # Split by frame: first n_frames → train, all frames available for test
    fids = sorted(set(c.fid for c in cam_infos))
    train_max_fid = fids[n_frames - 1]
    train_cam_infos = [c for c in cam_infos if c.fid <= train_max_fid]
    test_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # No initial point cloud: Gaussians come from the --init_ply reconstruction
    # (see make_init_ply.py) via scene_init.py, which replaces whatever the
    # Scene was built with.
    return SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
    )


sceneLoadTypeCallbacks = {
    # Colmap reader from official 3DGS [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Colmap": readColmapSceneInfo,
    # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "Blender": readNerfSyntheticInfo,
    "PacNeRF": readPACNeRFInfo,
    "Vid2Sim": readVid2SimInfo,
    "SpringGausMPMSynthetic": readSpringGausMPMSyntheticInfo,
    "SpringGausRealCapture": readSpringGausCaptureRealInfo,
    "Jukebox": readJukeboxInfo,
}
