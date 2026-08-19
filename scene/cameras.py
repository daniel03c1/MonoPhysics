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

import numpy as np
import torch
from torch import nn

from lib.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        fid=None,
        real_background=None,
        white_background=None,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.data_device = torch.device(data_device)

        self.gt_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.image_width = self.gt_image.shape[2]
        self.image_height = self.gt_image.shape[1]

        # Solid render background (white or black); original_image composites the
        # segmented foreground over it on demand.
        self.background = 1.0 if white_background else 0.0

        if gt_alpha_mask is None:
            self.gt_alpha_mask = None
        else:
            self.gt_alpha_mask = torch.tensor(
                gt_alpha_mask, dtype=torch.float32, device=self.data_device
            )

        # Realistic background plate (real datasets only), for representation
        # visualization; falls back to the solid color.
        self.real_background = (
            real_background if real_background is not None else self.background
        )

        self.zfar = 100.0
        self.znear = 1e-2

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def original_image(self):
        """
        Segmented foreground over the solid background — the train/refine/eval
        target. Equals gt_image when there is no mask.
        """
        if self.gt_alpha_mask is None:
            return self.gt_image
        return self.gt_image * self.gt_alpha_mask + self.background * (
            1 - self.gt_alpha_mask
        )

    def pw2pix(self, pw):
        assert pw.shape[1] == 3
        intrinsic = torch.from_numpy(self.intrinsic).to(pw)
        pc = self.pw2pc(pw)
        pix_coord = pc @ intrinsic.T
        pix_d = pix_coord[:, 2]
        pix_coord = pix_coord / pix_coord[:, 2:3]
        pix_coord = torch.round(pix_coord)
        pix_w, pix_h = pix_coord[:, 0].to(torch.int64), pix_coord[:, 1].to(torch.int64)
        return pix_w, pix_h, pix_d

    def pw2pc(self, pw):
        R_w2c = torch.from_numpy(self.R).to(pw)
        t_w2c = torch.from_numpy(self.T).to(pw)
        R_w2c = R_w2c.T
        pc = pw @ R_w2c.T + t_w2c.reshape(1, 3)
        return pc

    def is_in_view(self, pix_w, pix_h):
        in_mask = torch.logical_and(
            torch.logical_and(pix_w >= 0, pix_w < self.image_width),
            torch.logical_and(pix_h >= 0, pix_h < self.image_height),
        )
        return in_mask

    @property
    def intrinsic(self):
        fx = 0.5 * self.image_width / np.tan(self.FoVx / 2)
        fy = 0.5 * self.image_height / np.tan(self.FoVy / 2)
        cx = self.image_width / 2
        cy = self.image_height / 2
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return intrinsic
