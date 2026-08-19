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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	// Backward through the per-pixel image-space coordinate kernel.
	// Produces dL/dmeans2D (in NDC gradient convention, [P]) and
	// dL/dL_chol (raw Cholesky gradient, [P*4]).
	void render_pixel_coords(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float4* conic_opacity,
		const float* L_chol,
		const float* dL_dpixel_coords,
		float3* dL_dmeans2D,
		float* dL_dL_chol);

	// Backward through Cholesky factorization + cov2D -> cov3D chain.
	// Takes dL/dL_chol and dL/dmeans2D and produces dL/dcov3D and dL/dmeans3D.
	void preprocess_pixel_coords(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dL_chol,
		const float* L_chol,
		glm::vec3* dL_dmeans,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
