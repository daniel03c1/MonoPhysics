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

import math
import numpy as np
import os
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from gsplat.relocation import compute_relocation
from knn import knn_idx

from lib.general_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from lib.graphics_utils import BasicPointCloud
from lib.sh_utils import RGB2SH
from lib.system_utils import mkdir_p
from lib.volume_utils import compute_induced_volumes

N_MAX_BINOM = 51
BINOMS = torch.zeros((N_MAX_BINOM, N_MAX_BINOM), device="cuda")
for n in range(N_MAX_BINOM):
    for k in range(n + 1):
        BINOMS[n, k] = math.comb(n, k)

# MCMC source-selection: mix of render (opacity*scale) and inverse-volume signals.
MCMC_VOL_ALPHA = 0.5


class GaussianModel:
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._dx = None  # Unified grid spacing; set externally before any volume routine is called
        self.max_radii2D = torch.empty(0)

        self.c2w = None  # 4x4 camera-to-world transform (fixed, non-learnable)
        self._c2w_inv = None  # Cached inverse for world->camera transform

        self.optimizer = None
        self.scale_optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self._scene_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32).requires_grad_(True)
        )

        self.adam_betas = (0.8, 0.98)  # (0.9, 0.99)

    def get_scaling(self):
        return self.scaling_activation(self._scaling) * self.get_scene_scale().detach()

    def get_rotation(self):
        return self.rotation_activation(self._rotation)  # (N,4)

    def get_xyz(self, detach_scene_scale=False):
        """Returns positions in world space."""
        scene_scale = self.get_scene_scale()
        if detach_scene_scale:
            scene_scale = scene_scale.detach()
        scaled_xyz = self._xyz * scene_scale

        if self.c2w is None:
            return scaled_xyz

        # camera -> world: xyz_world = xyz @ R^T + t
        R = self.c2w[:3, :3]
        t = self.c2w[:3, 3]

        xyz_world = scaled_xyz @ R.T + t
        return xyz_world

    @torch.no_grad()
    def set_c2w(self, c2w_matrix):
        """
        Set the 4x4 camera-to-world transform (or None). Does NOT transform
        existing _xyz positions; only affects future get_xyz calls.
        """
        if c2w_matrix is None:
            self.c2w = None
            self._c2w_inv = None
        else:
            assert c2w_matrix.shape == (
                4,
                4,
            ), f"c2w must be 4x4, got {c2w_matrix.shape}"
            device = self._xyz.device if hasattr(self._xyz, "device") else "cuda"
            self.c2w = c2w_matrix.to(device)
            self._c2w_inv = None

    def get_features(self):
        features_dc = self._features_dc
        # Higher-order SH forced off; empty at the sh_degree=0 used everywhere.
        features_rest = self._features_rest * 0.0
        return torch.cat((features_dc, features_rest), dim=1)

    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_volumes(self):
        return self.get_induced_volumes()

    def get_induced_volumes(self):
        """Always compute induced volumes from positions via iP2G normalization."""
        if self._dx is None:
            raise RuntimeError(
                "GaussianModel._dx is not set. Compute the unified dx via "
                "compute_optimal_dx and assign before calling volume routines."
            )
        return compute_induced_volumes(self.get_xyz(), dx=self._dx)

    def get_scene_scale(self):
        return self._scene_scale.abs()

    def feasible_scene_scale(self, ground_plane, eps=1e-8):
        """
        Largest scene_scale for which no particle lies below the ground plane.

        get_xyz is x_world = s * (R x_cam) + t with t the camera center, so scaling
        dilates the cloud about that center and slides every particle along its viewing
        ray. Writing d_i = n . (x_world_i - t) = s_cur * a_i, the constraint
        n . (x_world_i - p) >= 0 becomes s * a_i >= n . (p - t), so each particle gives a
        bound on s whose direction is set by the sign of d_i.

        Returns None when the scale is unconstrained (no c2w, no plane, camera center
        below the plane, or no binding bound), so callers leave the scale alone.
        """
        if self.c2w is None or ground_plane is None:
            return None

        with torch.no_grad():
            point, normal = ground_plane
            xyz_world = self.get_xyz()
            t = self.c2w[:3, 3]
            s_cur = self.get_scene_scale().detach()

            b = torch.dot(point - t, normal)

            # Camera center below the plane makes the system possibly infeasible;
            # skip rather than raise, since this runs every iteration.
            if b > 0:
                return None

            # a_i < 0 (particle on the far side of the camera along -n): larger s
            # pushes it further below, so it caps s from above.
            d = (xyz_world - t) @ normal  # = s_cur * a_i
            below = d < -eps
            if not below.any():
                return None

            upper = s_cur * (b / d[below]).min()
            return float(upper) if upper > 0 else None

    def clamp_scene_scale_to_ground(self, ground_plane, eps=1e-8):
        """
        Project scene_scale onto the non-penetration set.

        Scale-only: particle positions are never edited. Returns the factor applied
        (1.0 when inactive) so callers can keep dependent quantities consistent.
        """
        s_max = self.feasible_scene_scale(ground_plane, eps=eps)
        if s_max is None or not (s_max > 0):
            return 1.0

        with torch.no_grad():
            s_cur = float(self.get_scene_scale().detach())
            s_new = min(s_cur, s_max)
            if s_new <= 0 or abs(s_new - s_cur) <= eps * max(1.0, s_cur):
                return 1.0
            # get_scene_scale applies .abs(), so store a positive value.
            self._scene_scale.data.fill_(abs(s_new))

            # A .data write leaves Adam's moments pointing the old way, which would
            # ratchet the scale straight back through the bound next step.
            if self.scale_optimizer is not None:
                for group in self.scale_optimizer.param_groups:
                    for p in group["params"]:
                        st = self.scale_optimizer.state.get(p)
                        if st:
                            st["exp_avg"].zero_()
                            st["exp_avg_sq"].zero_()

            return s_new / s_cur

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling().repeat(1, 3), scaling_modifier, self._rotation
        )

    def create_from_pcd(self, pcd: BasicPointCloud):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        points = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        dist2 = (points - points[knn_idx(points)]).square().sum(dim=-1).clamp(min=1e-7)

        scales = torch.log(torch.sqrt(dist2))[..., None]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz().shape[0]), device="cuda")

    def training_setup(self, training_args, fix_pcd=False, freeze_geometry=False):
        self.adam_betas = (
            training_args.adam_beta_1,
            training_args.adam_beta_2,
        )
        self.spatial_lr_scale = 5

        params = [
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        x_params = {
            "params": [self._xyz],
            "lr": training_args.position_lr_init * self.spatial_lr_scale,
            "name": "xyz",
        }

        if not fix_pcd:
            params.append(x_params)

        self.optimizer = torch.optim.Adam(
            params, lr=0.0, betas=self.adam_betas, eps=1e-15, fused=True
        )

        if fix_pcd and not freeze_geometry:
            self.x_optimizer = torch.optim.Adam(
                [x_params], betas=self.adam_betas, fused=True
            )
        else:
            self.x_optimizer = None

        scale_lr = training_args.scale_lr
        if scale_lr > 0:
            scale_params = {
                "params": [self._scene_scale],
                "lr": scale_lr,
                "name": "scale",
            }
            self.scale_optimizer = torch.optim.Adam(
                [scale_params], betas=self.adam_betas, fused=True
            )
            print(f"Scale optimizer initialized with lr={scale_lr}")
        else:
            self.scale_optimizer = None

    def construct_list_of_attributes(self):
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attributes.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attributes.append("f_rest_{}".format(i))
        attributes.append("opacity")
        for i in range(self._scaling.shape[1]):
            attributes.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            attributes.append("rot_{}".format(i))
        return attributes

    @torch.no_grad()
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz().detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attributes_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    @torch.no_grad()
    def _zero_optimizer_moments(self, name):
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            stored_state = self.optimizer.state.get(group["params"][0])
            if not stored_state:  # absent, or an empty dict left by a state migration
                return
            stored_state["exp_avg"].zero_()
            stored_state["exp_avg_sq"].zero_()
            return

    @torch.no_grad()
    def reset_opacity(self, target_value=0.01):
        opacities_new = inverse_sigmoid(torch.full_like(self._opacity, target_value))
        self._opacity.data.copy_(opacities_new)
        self._zero_optimizer_moments("opacity")

    @torch.no_grad()
    def reset_scaling(self):
        points = self.get_xyz().detach()
        scene_scale = self.get_scene_scale().detach()
        dist2 = (points - points[knn_idx(points)]).square().sum(dim=-1).clamp(min=1e-7)
        scales = (torch.sqrt(dist2) / scene_scale)[..., None]
        self._scaling.data.copy_(self.scaling_inverse_activation(scales))
        self._zero_optimizer_moments("scaling")

    @torch.no_grad()
    def reset_color(self):
        self._features_dc.data.zero_()
        self._features_rest.data.zero_()
        self._zero_optimizer_moments("f_dc")
        self._zero_optimizer_moments("f_rest")

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.ascontiguousarray(
            np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        )

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

        num_points = self._xyz.shape[0]
        self.max_radii2D = torch.zeros((num_points,), device="cuda")

    @torch.no_grad()
    def get_dead_mask(self, dead_fraction: float = 0.01):
        """
        Return boolean mask (N,), True for particles to recycle: out of frustum
        (max_radii2D == 0), NaN/Inf in any raw parameter, or bottom dead_fraction
        by opacity * prod(scaling) or by volume.
        """
        N = self._xyz.shape[0]

        frustum_mask = self.max_radii2D == 0

        nan_inf_mask = torch.zeros(N, dtype=torch.bool, device=self._xyz.device)
        for param in [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ]:
            flat = param.data.reshape(N, -1)
            nan_inf_mask |= ~torch.isfinite(flat).all(dim=1)

        opacity = self.get_opacity().squeeze(-1)
        scaling = self.get_scaling().reshape(N, -1)
        if scaling.shape[1] == 1:
            scaling = scaling.expand(-1, 3)
        contrib = opacity * scaling.prod(dim=-1)
        contrib_mask = contrib <= torch.quantile(contrib, dead_fraction)

        volumes = self.get_volumes()
        volume_mask = volumes <= torch.quantile(volumes, dead_fraction)

        dead = frustum_mask | nan_inf_mask | contrib_mask | volume_mask

        return dead, {
            "frustum": frustum_mask,
            "nan_inf": nan_inf_mask,
            "contrib": contrib_mask,
            "volume": volume_mask,
        }

    @torch.no_grad()
    def reset_stats(self):
        """Zero out the per-particle max-radii tracker."""
        self.max_radii2D.zero_()

    # Physical-space setters and particle manipulation methods

    @torch.no_grad()
    def set_opacity_from_physical(self, physical_opacities, indices=None):
        """Set opacity from physical-space [0,1] values."""
        raw = self.inverse_opacity_activation(physical_opacities)
        if indices is not None:
            self._opacity.data[indices] = raw
        else:
            self._opacity.data[:] = raw

    @torch.no_grad()
    def set_scaling_from_physical(self, physical_scales, indices=None):
        """Set scaling from physical-space values (undoes scene_scale and activation)."""
        scene_scale = self.get_scene_scale()
        raw = self.scaling_inverse_activation(physical_scales / scene_scale)
        if indices is not None:
            self._scaling.data[indices] = raw
        else:
            self._scaling.data[:] = raw

    @torch.no_grad()
    def copy_particles(self, dst_indices, src_indices):
        """
        Copy all raw attributes from src to dst particles; raw parameter space,
        so no activation/deactivation is needed.
        """
        self._xyz.data[dst_indices] = self._xyz.data[src_indices]
        self._features_dc.data[dst_indices] = self._features_dc.data[src_indices]
        self._features_rest.data[dst_indices] = self._features_rest.data[src_indices]
        self._opacity.data[dst_indices] = self._opacity.data[src_indices]
        self._scaling.data[dst_indices] = self._scaling.data[src_indices]
        self._rotation.data[dst_indices] = self._rotation.data[src_indices]

    @torch.no_grad()
    def extend_particles(self, src_indices):
        """Append new particles by copying attributes from src_indices."""
        n_new = len(src_indices)

        def extend(param, new_data):
            return nn.Parameter(
                torch.cat([param.data, new_data], dim=0).requires_grad_(True)
            )

        attrs = [
            (
                "_xyz",
                self._xyz,
                self._xyz.data[src_indices],
                self.x_optimizer or self.optimizer,
            ),
            (
                "_features_dc",
                self._features_dc,
                self._features_dc.data[src_indices],
                self.optimizer,
            ),
            (
                "_features_rest",
                self._features_rest,
                self._features_rest.data[src_indices],
                self.optimizer,
            ),
            (
                "_opacity",
                self._opacity,
                self._opacity.data[src_indices],
                self.optimizer,
            ),
            (
                "_scaling",
                self._scaling,
                self._scaling.data[src_indices],
                self.optimizer,
            ),
            (
                "_rotation",
                self._rotation,
                self._rotation.data[src_indices],
                self.optimizer,
            ),
        ]

        for attr_name, old_param, new_data, opt in attrs:
            new_param = extend(old_param, new_data)
            setattr(self, attr_name, new_param)
            self.migrate_optimizer_state(opt, old_param, new_param, "append")

        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros(n_new, device="cuda")], dim=0
        )

        return n_new

    @torch.no_grad()
    def zero_optimizer_moments_at(self, indices):
        """Zero out Adam state at given indices for all parameters."""
        params_and_opts = [
            (self._xyz, self.optimizer),
            (self._features_dc, self.optimizer),
            (self._features_rest, self.optimizer),
            (self._scaling, self.optimizer),
            (self._rotation, self.optimizer),
            (self._opacity, self.optimizer),
            (self._xyz, self.x_optimizer),
        ]

        for param, opt in params_and_opts:
            self.migrate_optimizer_state(opt, param, param, "update", indices=indices)

    # MCMC relocation and growth

    @torch.no_grad()
    def migrate_optimizer_state(self, optimizer, param, new_param, mode, indices=None):
        """
        Sync Adam state when particle parameters are mutated: mode='update'
        zeros state at indices (relocated), mode='append' extends with zeros.
        """
        if optimizer is None:
            return

        if not any(
            p is param for group in optimizer.param_groups for p in group["params"]
        ):
            return

        state = optimizer.state[param]
        del optimizer.state[param]

        new_state = {}
        for key, v in state.items():
            if key == "step":
                new_state[key] = v
                continue

            if mode == "update":
                if indices is not None:
                    v[indices] = 0
                new_state[key] = v
            elif mode == "append":
                n_add = new_param.shape[0] - v.shape[0]
                zeros = torch.zeros(
                    (n_add, *v.shape[1:]), device=v.device, dtype=v.dtype
                )
                new_state[key] = torch.cat([v, zeros])

        optimizer.state[new_param] = new_state
        for group in optimizer.param_groups:
            try:
                idx = next(i for i, p in enumerate(group["params"]) if p is param)
                group["params"][idx] = new_param
            except StopIteration:
                continue

    @torch.no_grad()
    def compute_mcmc_probs(
        self, candidate_indices, opacities, scales, induced_volumes=None
    ):
        """
        Mixed MCMC source-selection probability: p_i ∝ (1 - MCMC_VOL_ALPHA) *
        p_render(i) + MCMC_VOL_ALPHA * p_volume(i), each independently normalized;
        falls back to render-only if induced_volumes is None or all-zero.
        """
        cand_opacity = opacities[candidate_indices].flatten()
        cand_scaling = scales[candidate_indices]
        if cand_scaling.shape[1] == 1:
            cand_scaling = cand_scaling.expand(-1, 3)
        p_render = cand_opacity * cand_scaling.prod(-1)
        p_render = p_render / (p_render.sum() + torch.finfo(torch.float32).eps)

        if MCMC_VOL_ALPHA > 0 and induced_volumes is not None:
            vols = induced_volumes[candidate_indices].flatten()
            p_sum = vols.sum()
            if p_sum > 0:
                p_vol = vols / p_sum
                return (1 - MCMC_VOL_ALPHA) * p_render + MCMC_VOL_ALPHA * p_vol

        return p_render

    @torch.no_grad()
    def rescale_split_sources(
        self, sampled_idxs, ratios, opacities, scales, min_opacity
    ):
        """
        Rescale source particles' opacity/scale in place via gsplat
        compute_relocation, so a source plus its `ratios` copies preserve the
        original contribution. Callers do the copying: relocate
        (clone-into-dead-slots) and add_new (clone-and-append).
        """
        is_isotropic = scales.shape[1] == 1
        selected_scales = scales[sampled_idxs]
        if is_isotropic:
            selected_scales = selected_scales.repeat(1, 3)

        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=selected_scales,
            ratios=ratios,
            binoms=BINOMS,
        )

        if is_isotropic and new_scales.shape[1] == 3:
            new_scales = new_scales.mean(dim=1, keepdim=True)

        eps = torch.finfo(torch.float32).eps
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

        self.set_opacity_from_physical(new_opacities, sampled_idxs)
        self.set_scaling_from_physical(new_scales, sampled_idxs)

    @torch.no_grad()
    def relocate(
        self,
        dead_fraction: float = 0.01,
        min_opacity: float = 0.005,
        induced_volumes=None,
    ):
        """Relocate dead Gaussians to live ones (in place)."""
        dead_mask, breakdown = self.get_dead_mask(dead_fraction=dead_fraction)
        n_dead = dead_mask.sum().item()
        n_total = self._xyz.shape[0]

        print(
            f"[MCMC Relocate] {n_dead}/{n_total} dead "
            f"(frustum={breakdown['frustum'].sum().item()}, "
            f"nan_inf={breakdown['nan_inf'].sum().item()}, "
            f"contrib={breakdown['contrib'].sum().item()}, "
            f"volume={breakdown['volume'].sum().item()})"
        )

        if n_dead == 0:
            return

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]

        opacities = self.get_opacity()
        scales = self.get_scaling()

        probs = self.compute_mcmc_probs(
            alive_indices, opacities, scales, induced_volumes
        )
        sampled_idxs_local = torch.topk(probs, n_dead).indices
        sampled_idxs = alive_indices[sampled_idxs_local]
        ratios = torch.bincount(sampled_idxs_local)[sampled_idxs_local] + 1

        self.rescale_split_sources(sampled_idxs, ratios, opacities, scales, min_opacity)
        self.copy_particles(dead_indices, sampled_idxs)
        self.zero_optimizer_moments_at(dead_indices)

    @torch.no_grad()
    def add_new(
        self,
        cap_max: int,
        min_opacity: float = 0.005,
        max_add_ratio: float = 0.05,
        induced_volumes=None,
    ):
        """Grow the particle cloud toward cap_max by sampling existing particles."""
        current_n = self._xyz.shape[0]
        if current_n >= cap_max:
            print(f"[MCMC Add] Cap reached. Current: {current_n}, Cap: {cap_max}")
            return

        n_target = min(cap_max, int((1 + max_add_ratio) * current_n))
        n_new = n_target - current_n
        print(f"[MCMC Add] Adding {n_new} particles. Total: {current_n} -> {n_target}")
        if n_new <= 0:
            return

        opacities = self.get_opacity()
        scales = self.get_scaling()

        all_indices = torch.arange(current_n, device=opacities.device)
        probs = self.compute_mcmc_probs(all_indices, opacities, scales, induced_volumes)
        sampled_idxs = torch.topk(probs, n_new).indices
        ratios = torch.bincount(sampled_idxs)[sampled_idxs] + 1

        self.rescale_split_sources(sampled_idxs, ratios, opacities, scales, min_opacity)
        self.extend_particles(sampled_idxs)
