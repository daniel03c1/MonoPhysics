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

#include "knn.h"
#include "spatial.h"

#include <cstdint>
#include <cuda_runtime.h> // Include the CUDA runtime header for cudaSetDevice()

torch::Tensor knn_idx(const torch::Tensor &points, int k) {
  const int P = points.size(0);

  // Determine which device the tensor is on and set it
  auto device = points.device();
  int device_index = device.index();
  cudaSetDevice(device_index);

  // Allocate output tensor of shape [P, k]
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor indices = torch::full({P, k}, 0, int_opts);

  KNN::knn_idx(P, (float3 *)points.contiguous().data_ptr<float>(),
               (uint32_t *)indices.contiguous().data_ptr<int>(), k);

  // Squeeze last dimension when k==1 for backward compatibility
  if (k == 1) {
    return indices.squeeze(-1);
  }
  return indices;
}

torch::Tensor knn_cross_2d(const torch::Tensor &source,
                           const torch::Tensor &target, int k) {
  const int M = source.size(0);
  const int N = target.size(0);

  // Determine which device the tensor is on and set it
  auto device = source.device();
  int device_index = device.index();
  cudaSetDevice(device_index);

  // Allocate output tensor of shape [N, k]
  auto int_opts = source.options().dtype(torch::kInt32);
  torch::Tensor indices = torch::full({N, k}, 0, int_opts);

  KNN::knn_cross_2d(M, (float2 *)source.contiguous().data_ptr<float>(), N,
                    (float2 *)target.contiguous().data_ptr<float>(),
                    (uint32_t *)indices.contiguous().data_ptr<int>(), k);

  if (k == 1) {
    return indices.squeeze(-1);
  }
  return indices;
}

torch::Tensor knn_cross_3d(const torch::Tensor &source,
                           const torch::Tensor &target, int k) {
  const int M = source.size(0);
  const int N = target.size(0);

  auto device = source.device();
  int device_index = device.index();
  cudaSetDevice(device_index);

  auto int_opts = source.options().dtype(torch::kInt32);
  torch::Tensor indices = torch::full({N, k}, 0, int_opts);

  KNN::knn_cross_3d(M, (float3 *)source.contiguous().data_ptr<float>(), N,
                    (float3 *)target.contiguous().data_ptr<float>(),
                    (uint32_t *)indices.contiguous().data_ptr<int>(), k);

  if (k == 1) {
    return indices.squeeze(-1);
  }
  return indices;
}
