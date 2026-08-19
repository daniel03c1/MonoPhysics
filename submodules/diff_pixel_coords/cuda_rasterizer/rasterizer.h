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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		// Per-pixel image-space coordinate forward pass.
		// Outputs a (3, H, W) tensor: [u, v, accumulated_alpha].
		// Gradients flow through means2D and L_chol (2D covariance Cholesky factor).
		static int forwardPixelCoords(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_pixel_coords,
			bool antialiasing,
			int* radii,
			bool debug);

		// Backward pass for the per-pixel image-space coordinates.
		// Propagates dL/d(out_pixel_coords) back to means3D, scales, rotations, cov3D_precomp.
		// dL_dL_chol_scratch is a caller-owned zeroed [P*4] device buffer.
		static void backwardPixelCoords(
			const int P, int D, int M, int R,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dpixel_coords,
			float* dL_dmean2D,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dL_chol_scratch,
			bool antialiasing,
			bool debug);
	};
};

#endif
