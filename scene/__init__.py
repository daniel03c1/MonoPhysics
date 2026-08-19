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

import copy
import json
import os
import random
import torch
from arguments import ModelParams
from scene.dataset_readers import SceneInfo, sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from lib.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        shuffle=True,
        resolution_scales=[1.0],
        pcd=None,
        cam_info=None,
        n_train_frames=None,
    ):
        """Initialize Scene object."""
        self.model_path = args.model_path
        self.gaussians = gaussians

        self.cameras = {} if cam_info is None else cam_info.get("cameras")
        self.n_train_frames = (
            n_train_frames if cam_info is None else cam_info.get("n_train_frames")
        )
        read_cam = True if cam_info is None else False

        if not read_cam:
            print("use given cams")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(
            os.path.join(args.source_path, "transforms_train_nvs.json")
        ):
            print("Found transforms_train_nvs.json file, assuming Vid2Sim data set!")
            scene_info = sceneLoadTypeCallbacks["Vid2Sim"](
                args.source_path,
                args.config_path,
                args.white_background,
            )
        elif os.path.exists(os.path.join(args.source_path, "all_data.json")):
            print("Found all_data.json file, assuming PacNeRF data set!")
            scene_info = sceneLoadTypeCallbacks["PacNeRF"](
                args.source_path,
                args.config_path,
                args.white_background,
                read_cam=read_cam,
            )
        elif (
            os.path.exists(os.path.join(args.source_path, "camera.json"))
            and os.path.exists(os.path.join(args.source_path, "images"))
            and not os.path.exists(os.path.join(args.source_path, "frame.json"))
        ):
            print(
                "Found camera.json + images/ without frame.json, assuming Jukebox data set!"
            )
            scene_info = sceneLoadTypeCallbacks["Jukebox"](
                args.source_path,
                args.config_path,
                args.white_background,
            )
        elif os.path.exists(
            os.path.join(args.source_path, "camera.json")
        ) and os.path.exists(os.path.join(args.source_path, "frame.json")):
            print(
                "Found camera.json + frame.json, assuming SpringGaus MPM Synthetic data set!"
            )
            scene_info = sceneLoadTypeCallbacks["SpringGausMPMSynthetic"](
                args.source_path,
                args.config_path,
                args.white_background,
                args.num_frame,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        elif "real_capture" in args.source_path:
            print("Found real_capture, assuming Spring-Gaus Real Capture data set!")
            scene_info = sceneLoadTypeCallbacks["SpringGausRealCapture"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if read_cam:
            os.makedirs(self.model_path, exist_ok=True)
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle and read_cam:
            # Multi-res consistent random shuffling
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = (
            scene_info.nerf_normalization["radius"]
            if read_cam
            else cam_info.get("cameras_extent")
        )

        for resolution_scale in resolution_scales:
            if not read_cam:
                continue
            print("Loading Cameras")
            # One pool of unique views; train/test are derived from (cam_idx, frame).
            seen, infos = set(), []
            for c in scene_info.train_cameras + scene_info.test_cameras:
                if (c.uid, c.fid) not in seen:
                    seen.add((c.uid, c.fid))
                    infos.append(c)
            self.cameras[resolution_scale] = cameraList_from_camInfos(
                infos, resolution_scale, args
            )

        if pcd is not None:
            self.gaussians.create_from_pcd(pcd)
        elif scene_info.point_cloud is not None:
            self.gaussians.create_from_pcd(scene_info.point_cloud)
        # Readers that supply no initial cloud leave the Gaussians empty here;
        # scene_init.py fills them from the --init_ply reconstruction
        # (see make_init_ply.py).

        self.cam_idx = args.cam_idx

        all_fids = []
        for scale, cam_list in self.cameras.items():
            all_fids.extend([view.fid for view in cam_list])
        self.fids = (
            torch.unique(torch.stack(all_fids)) if all_fids else torch.tensor([])
        )

    def is_train_view(self, cam):
        """
        The single source of the train/test split: a view is training iff it is
        the selected camera and its frame is within the first n_train_frames.
        Everything else is test.
        """
        if self.n_train_frames and len(self.fids) > 0:
            max_fid = self.fids[min(self.n_train_frames, len(self.fids)) - 1]
            if cam.fid > max_fid:
                return False
        return cam.uid == self.cam_idx

    def getTrainCameras(
        self,
        scale=1.0,
        shuffle=True,
        max_fid=None,
        max_fid_idx=None,
        exact_fid=None,
    ):
        """
        Training views: the selected camera at the first n_train_frames frames.
        max_fid/max_fid_idx/exact_fid further restrict the frame range (ramp/init).
        """
        cameras = [c for c in self.cameras[scale] if self.is_train_view(c)]

        if exact_fid is not None:
            cameras = [cam for cam in cameras if cam.fid == exact_fid]
        elif max_fid is not None:
            cameras = [cam for cam in cameras if cam.fid <= max_fid]
        elif max_fid_idx is not None and len(self.fids) > max_fid_idx:
            cameras = [cam for cam in cameras if cam.fid <= self.fids[max_fid_idx]]

        if shuffle:
            random.shuffle(cameras)
        return cameras

    def getTestCameras(self, scale=1.0, shuffle=False):
        """
        Test views: the complement of the training set, i.e. the other cameras
        at any frame plus the training camera at frames >= n_train_frames.
        """
        cameras = [c for c in self.cameras[scale] if not self.is_train_view(c)]
        if shuffle:
            random.shuffle(cameras)
        return cameras

    def save(self, iteration, fix_pcd=False):
        """Save the point cloud to disk."""
        name = (
            "point_cloud/iteration_{}".format(iteration)
            if not fix_pcd
            else "point_cloud_fix_pcd/iteration_{}".format(iteration)
        )
        point_cloud_path = os.path.join(self.model_path, name)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
