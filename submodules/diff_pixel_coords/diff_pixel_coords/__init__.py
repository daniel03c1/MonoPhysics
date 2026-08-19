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

from typing import NamedTuple
import torch

from . import _C


def rasterize_pixel_coords(
    means3D,
    means2D,
    sh,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizePixelCoords.apply(
        means3D,
        means2D,
        sh,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizePixelCoords(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        num_rendered, pixel_coords, radii, geomBuffer, binningBuffer, imgBuffer = (
            _C.rasterize_pixel_coords(*args)
        )

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.means2D_ncols = means2D.shape[1]  # save so backward can match input shape
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            opacities,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return pixel_coords, radii

    @staticmethod
    def backward(ctx, grad_pixel_coords, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            opacities,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            means3D,
            radii,
            scales,
            opacities,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_pixel_coords,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        (
            grad_means2D,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_scales,
            grad_rotations,
            grad_opacities,
        ) = _C.rasterize_pixel_coords_backward(*args)

        # grad_means2D from CUDA is always [P, 3] (float3* internal format).
        # Narrow to match the actual means2D input shape (may be [P, 2] or [P, 3]).
        grad_means2D = grad_means2D.narrow(1, 0, ctx.means2D_ncols)

        grads = (
            grad_means3D,
            grad_means2D,
            None,  # sh
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # raster_settings
        )

        return grads


class PixelCoordSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool
