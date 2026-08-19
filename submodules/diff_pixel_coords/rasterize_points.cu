/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// ─── Per-pixel image-space coordinates ───────────────────────────────────────────

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePixelCoordsCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto float_opts = means3D.options().dtype(torch::kFloat32);

  // Output: [u, v, accumulated_alpha]
  torch::Tensor out_pixel_coords = torch::zeros({3, H, W}, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer   = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer    = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0)
  {
    int M = 0;
    if (sh.size(0) != 0)
      M = sh.size(1);

    // Colors precomp: null (not needed for UV render, but preprocess needs a dummy)
    // We pass sh and let preprocess compute RGB internally (stored but unused by UV kernel).
    rendered = CudaRasterizer::Rasterizer::forwardPixelCoords(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      W, H,
      means3D.contiguous().data<float>(),
      sh.contiguous().data_ptr<float>(),
      nullptr,   // colors_precomp: use SH path
      opacity.contiguous().data<float>(),
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(),
      viewmatrix.contiguous().data<float>(),
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_pixel_coords.contiguous().data<float>(),
      antialiasing,
      radii.contiguous().data<int>(),
      debug);
  }
  return std::make_tuple(rendered, out_pixel_coords, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePixelCoordsBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& opacities,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_pixel_coords,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_pixel_coords.size(1);
  const int W = dL_dout_pixel_coords.size(2);

  int M = 0;
  if (sh.size(0) != 0)
    M = sh.size(1);

  torch::Tensor dL_dmeans3D    = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D    = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcov3D      = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dscales     = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations  = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dsh         = torch::zeros({P, M, 3}, means3D.options());
  // Always zero: this backward carries no opacity gradient (opacity is detached
  // by render_pixel_coords). Kept so the returned tuple keeps its shape.
  torch::Tensor dL_dopacity    = torch::zeros({P, 1}, means3D.options());
  // Scratch for the per-Gaussian Cholesky gradient, from torch's caching allocator so
  // the backward avoids a synchronizing cudaMalloc/cudaFree per call.
  torch::Tensor dL_dL_chol     = torch::zeros({P, 4}, means3D.options());

  if (P != 0)
  {
    CudaRasterizer::Rasterizer::backwardPixelCoords(
      P, degree, M, R,
      W, H,
      means3D.contiguous().data<float>(),
      sh.contiguous().data<float>(),
      opacities.contiguous().data<float>(),
      scales.data_ptr<float>(),
      scale_modifier,
      rotations.data_ptr<float>(),
      (cov3D_precomp.size(0) == 0) ? nullptr : cov3D_precomp.contiguous().data<float>(),
      viewmatrix.contiguous().data<float>(),
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      radii.contiguous().data<int>(),
      reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
      dL_dout_pixel_coords.contiguous().data<float>(),
      dL_dmeans2D.contiguous().data<float>(),
      dL_dmeans3D.contiguous().data<float>(),
      dL_dcov3D.contiguous().data<float>(),
      dL_dsh.contiguous().data<float>(),
      dL_dscales.contiguous().data<float>(),
      dL_drotations.contiguous().data<float>(),
      dL_dL_chol.contiguous().data<float>(),
      antialiasing,
      debug);
  }

  // dL_dmeans2D is [P,3] internally (float3* for CUDA); the z column is always zero.
  // Return the full [P,3] so shapes match the [P,3] means2D (screenspace_points) input.
  return std::make_tuple(dL_dmeans2D, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations, dL_dopacity);
}
