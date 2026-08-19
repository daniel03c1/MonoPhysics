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

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "knn.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <vector>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin {
  __device__ __forceinline__ float3 operator()(const float3 &a,
                                               const float3 &b) const {
    return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
  }
};

struct CustomMax {
  __device__ __forceinline__ float3 operator()(const float3 &a,
                                               const float3 &b) const {
    return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
  }
};

// 2D versions for cross-group KNN
struct CustomMin2D {
  __device__ __forceinline__ float2 operator()(const float2 &a,
                                               const float2 &b) const {
    return {min(a.x, b.x), min(a.y, b.y)};
  }
};

struct CustomMax2D {
  __device__ __forceinline__ float2 operator()(const float2 &a,
                                               const float2 &b) const {
    return {max(a.x, b.x), max(a.y, b.y)};
  }
};

__host__ __device__ uint32_t prepMorton(uint32_t x) {
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn,
                                          float3 maxx) {
  uint32_t x =
      prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
  uint32_t y =
      prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
  uint32_t z =
      prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

  return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3 *points, float3 minn,
                             float3 maxx, uint32_t *codes) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax {
  float3 minn;
  float3 maxx;
};

struct MinMax2D {
  float2 minn;
  float2 maxx;
};

__device__ __host__ float distBoxPoint(const MinMax &box, const float3 &p) {
  float3 diff = {0, 0, 0};
  if (p.x < box.minn.x || p.x > box.maxx.x)
    diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
  if (p.y < box.minn.y || p.y > box.maxx.y)
    diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
  if (p.z < box.minn.z || p.z > box.maxx.z)
    diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
  return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

// 2D Morton code functions
__host__ __device__ uint32_t prepMorton2D(uint32_t x) {
  x = (x | (x << 16)) & 0x0000FFFF;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  return x;
}

__host__ __device__ uint32_t coord2Morton2D(float2 coord, float2 minn,
                                            float2 maxx) {
  uint32_t x =
      prepMorton2D(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 16) - 1));
  uint32_t y =
      prepMorton2D(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 16) - 1));
  return x | (y << 1);
}

__global__ void coord2Morton2DKernel(int P, const float2 *points, float2 minn,
                                     float2 maxx, uint32_t *codes) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;
  codes[idx] = coord2Morton2D(points[idx], minn, maxx);
}

__device__ __host__ float distBoxPoint2D(const MinMax2D &box, const float2 &p) {
  float2 diff = {0, 0};
  if (p.x < box.minn.x || p.x > box.maxx.x)
    diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
  if (p.y < box.minn.y || p.y > box.maxx.y)
    diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
  return diff.x * diff.x + diff.y * diff.y;
}

__global__ void buildBoxes2D(uint32_t P, float2 *points, uint32_t *indices,
                             MinMax2D *boxes) {
  uint32_t box_idx = blockIdx.x;
  uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
  if (box_idx >= num_boxes)
    return;

  // Initialize shared memory for reduction
  __shared__ float s_min_x[BOX_SIZE];
  __shared__ float s_min_y[BOX_SIZE];
  __shared__ float s_max_x[BOX_SIZE];
  __shared__ float s_max_y[BOX_SIZE];

  uint32_t point_idx = box_idx * BOX_SIZE + threadIdx.x;
  if (point_idx < P) {
    float2 pt = points[indices[point_idx]];
    s_min_x[threadIdx.x] = pt.x;
    s_min_y[threadIdx.x] = pt.y;
    s_max_x[threadIdx.x] = pt.x;
    s_max_y[threadIdx.x] = pt.y;
  } else {
    s_min_x[threadIdx.x] = FLT_MAX;
    s_min_y[threadIdx.x] = FLT_MAX;
    s_max_x[threadIdx.x] = -FLT_MAX;
    s_max_y[threadIdx.x] = -FLT_MAX;
  }
  __syncthreads();

  // Parallel reduction
  for (uint32_t stride = BOX_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_min_x[threadIdx.x] = min(s_min_x[threadIdx.x], s_min_x[threadIdx.x + stride]);
      s_min_y[threadIdx.x] = min(s_min_y[threadIdx.x], s_min_y[threadIdx.x + stride]);
      s_max_x[threadIdx.x] = max(s_max_x[threadIdx.x], s_max_x[threadIdx.x + stride]);
      s_max_y[threadIdx.x] = max(s_max_y[threadIdx.x], s_max_y[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    boxes[box_idx].minn = {s_min_x[0], s_min_y[0]};
    boxes[box_idx].maxx = {s_max_x[0], s_max_y[0]};
  }
}

__global__ void buildBoxes3D(uint32_t P, float3 *points, uint32_t *indices,
                             MinMax *boxes) {
  uint32_t box_idx = blockIdx.x;
  uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
  if (box_idx >= num_boxes)
    return;

  // Initialize shared memory for reduction
  __shared__ float s_min_x[BOX_SIZE];
  __shared__ float s_min_y[BOX_SIZE];
  __shared__ float s_min_z[BOX_SIZE];
  __shared__ float s_max_x[BOX_SIZE];
  __shared__ float s_max_y[BOX_SIZE];
  __shared__ float s_max_z[BOX_SIZE];

  uint32_t point_idx = box_idx * BOX_SIZE + threadIdx.x;
  if (point_idx < P) {
    float3 pt = points[indices[point_idx]];
    s_min_x[threadIdx.x] = pt.x;
    s_min_y[threadIdx.x] = pt.y;
    s_min_z[threadIdx.x] = pt.z;
    s_max_x[threadIdx.x] = pt.x;
    s_max_y[threadIdx.x] = pt.y;
    s_max_z[threadIdx.x] = pt.z;
  } else {
    s_min_x[threadIdx.x] = FLT_MAX;
    s_min_y[threadIdx.x] = FLT_MAX;
    s_min_z[threadIdx.x] = FLT_MAX;
    s_max_x[threadIdx.x] = -FLT_MAX;
    s_max_y[threadIdx.x] = -FLT_MAX;
    s_max_z[threadIdx.x] = -FLT_MAX;
  }
  __syncthreads();

  // Parallel reduction
  for (uint32_t stride = BOX_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_min_x[threadIdx.x] = min(s_min_x[threadIdx.x], s_min_x[threadIdx.x + stride]);
      s_min_y[threadIdx.x] = min(s_min_y[threadIdx.x], s_min_y[threadIdx.x + stride]);
      s_min_z[threadIdx.x] = min(s_min_z[threadIdx.x], s_min_z[threadIdx.x + stride]);
      s_max_x[threadIdx.x] = max(s_max_x[threadIdx.x], s_max_x[threadIdx.x + stride]);
      s_max_y[threadIdx.x] = max(s_max_y[threadIdx.x], s_max_y[threadIdx.x + stride]);
      s_max_z[threadIdx.x] = max(s_max_z[threadIdx.x], s_max_z[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    boxes[box_idx].minn = {s_min_x[0], s_min_y[0], s_min_z[0]};
    boxes[box_idx].maxx = {s_max_x[0], s_max_y[0], s_max_z[0]};
  }
}

template <int K>
__device__ void updateKBest2D(const float2 &ref, const float2 &point,
                              uint32_t pointIdx, float *knn_dist,
                              uint32_t *knn_idx) {
  // Skip if this point is already in the k-best list
  for (int j = 0; j < K; j++) {
    if (knn_idx[j] == pointIdx)
      return;
  }

  float2 d = {point.x - ref.x, point.y - ref.y};
  float dist = d.x * d.x + d.y * d.y;
  for (int j = 0; j < K; j++) {
    if (knn_dist[j] > dist) {
      // Swap distance
      float t_dist = knn_dist[j];
      knn_dist[j] = dist;
      dist = t_dist;

      // Swap index
      uint32_t t_idx = knn_idx[j];
      knn_idx[j] = pointIdx;
      pointIdx = t_idx;
    }
  }
}

template <int K>
__global__ void crossKNN2D(uint32_t M, float2 *source_points,
                           uint32_t *source_indices, MinMax2D *source_boxes,
                           uint32_t N, float2 *target_points,
                           uint32_t *nearest_indices) {
  int idx = cg::this_grid().thread_rank();
  if (idx >= N)
    return;

  float2 target = target_points[idx];
  float best_dist[K];
  uint32_t best_idx[K];
  for (int j = 0; j < K; j++) {
    best_dist[j] = FLT_MAX;
    best_idx[j] = 0;
  }

  uint32_t num_boxes = (M + BOX_SIZE - 1) / BOX_SIZE;

  // Search through source boxes
  for (uint32_t b = 0; b < num_boxes; b++) {
    MinMax2D box = source_boxes[b];
    float box_dist = distBoxPoint2D(box, target);
    if (box_dist > best_dist[K - 1])
      continue;

    // Search points in this box
    uint32_t box_start = b * BOX_SIZE;
    uint32_t box_end = min(M, (b + 1) * BOX_SIZE);
    for (uint32_t i = box_start; i < box_end; i++) {
      uint32_t src_idx = source_indices[i];
      updateKBest2D<K>(target, source_points[src_idx], src_idx, best_dist,
                       best_idx);
    }
  }
  for (int j = 0; j < K; j++) {
    nearest_indices[idx * K + j] = best_idx[j];
  }
}

template <int K>
__device__ void updateKBest(const float3 &ref, const float3 &point,
                            uint32_t pointIdx, float *knn_dist,
                            uint32_t *knn_idx) {
  // Skip if this point is already in the k-best list
  for (int j = 0; j < K; j++) {
    if (knn_idx[j] == pointIdx)
      return;
  }

  float3 d = {point.x - ref.x, point.y - ref.y, point.z - ref.z};
  float dist = d.x * d.x + d.y * d.y + d.z * d.z;
  for (int j = 0; j < K; j++) {
    if (knn_dist[j] > dist) {
      // Swap distance
      float t_dist = knn_dist[j];
      knn_dist[j] = dist;
      dist = t_dist;

      // Swap index
      uint32_t t_idx = knn_idx[j];
      knn_idx[j] = pointIdx;
      pointIdx = t_idx;
    }
  }
}

template <int K>
__global__ void boxNearestIdx(uint32_t P, float3 *points, uint32_t *indices,
                              MinMax *boxes, uint32_t *nearest_indices) {
  int idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  float3 point = points[indices[idx]];
  float best_dist[K];
  uint32_t best_idx[K];
  for (int j = 0; j < K; j++) {
    best_dist[j] = FLT_MAX;
    best_idx[j] = 0;
  }

  // 1. Check immediate neighbors in Morton order (fast heuristic)
  for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++) {
    if (i == idx)
      continue;
    updateKBest<K>(point, points[indices[i]], indices[i], best_dist, best_idx);
  }

  float reject = best_dist[K - 1];

  // 2. Search through all boxes
  for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++) {
    MinMax box = boxes[b];
    float dist = distBoxPoint(box, point);
    // If the box is further away than our K-th best, skip it
    if (dist > reject)
      continue;

    for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++) {
      if (i == idx)
        continue;
      updateKBest<K>(point, points[indices[i]], indices[i], best_dist,
                     best_idx);
    }
    reject = best_dist[K - 1];
  }
  // Store the indices of the K closest points for the original point index
  uint32_t orig_idx = indices[idx];
  for (int j = 0; j < K; j++) {
    nearest_indices[orig_idx * K + j] = best_idx[j];
  }
}

void KNN::knn_idx(int P, float3 *points, uint32_t *nearestIndices,
                        int K) {
  float3 *result;
  cudaMalloc(&result, sizeof(float3));
  size_t temp_storage_bytes;

  float3 init = {0, 0, 0}, minn, maxx;

  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P,
                            CustomMin(), init);
  thrust::device_vector<char> temp_storage(temp_storage_bytes);

  cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes,
                            points, result, P, CustomMin(), init);
  cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

  cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes,
                            points, result, P, CustomMax(), init);
  cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

  thrust::device_vector<uint32_t> morton(P);
  thrust::device_vector<uint32_t> morton_sorted(P);
  coord2Morton<<<(P + 255) / 256, 256>>>(P, points, minn, maxx,
                                         morton.data().get());

  thrust::device_vector<uint32_t> indices(P);
  thrust::sequence(indices.begin(), indices.end());
  thrust::device_vector<uint32_t> indices_sorted(P);

  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, morton.data().get(),
      morton_sorted.data().get(), indices.data().get(),
      indices_sorted.data().get(), P);
  temp_storage.resize(temp_storage_bytes);

  cub::DeviceRadixSort::SortPairs(
      temp_storage.data().get(), temp_storage_bytes, morton.data().get(),
      morton_sorted.data().get(), indices.data().get(),
      indices_sorted.data().get(), P);

  uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
  thrust::device_vector<MinMax> boxes(num_boxes);
  buildBoxes3D<<<num_boxes, BOX_SIZE>>>(P, points, indices_sorted.data().get(),
                                        boxes.data().get());
  switch (K) {
  case 1:
    boxNearestIdx<1><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 2:
    boxNearestIdx<2><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 3:
    boxNearestIdx<3><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 4:
    boxNearestIdx<4><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 5:
    boxNearestIdx<5><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 6:
    boxNearestIdx<6><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 7:
    boxNearestIdx<7><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 8:
    boxNearestIdx<8><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 9:
    boxNearestIdx<9><<<num_boxes, BOX_SIZE>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  case 16: {
    uint32_t knn_blocks = (P + 255) / 256;
    boxNearestIdx<16><<<knn_blocks, 256>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  }
  case 32: {
    uint32_t knn_blocks = (P + 255) / 256;
    boxNearestIdx<32><<<knn_blocks, 256>>>(
        P, points, indices_sorted.data().get(), boxes.data().get(),
        nearestIndices);
    break;
  }
  default:
    assert(false && "Unsupported K value (supported: 1-9, 16, 32)");
  }

  cudaFree(result);
}

template <int K>
__global__ void crossKNN3D(uint32_t M, float3 *source_points,
                           uint32_t *source_indices, MinMax *source_boxes,
                           uint32_t N, float3 *target_points,
                           uint32_t *nearest_indices) {
  int idx = cg::this_grid().thread_rank();
  if (idx >= N)
    return;

  float3 target = target_points[idx];
  float best_dist[K];
  uint32_t best_idx[K];
  for (int j = 0; j < K; j++) {
    best_dist[j] = FLT_MAX;
    best_idx[j] = 0;
  }

  uint32_t num_boxes = (M + BOX_SIZE - 1) / BOX_SIZE;

  for (uint32_t b = 0; b < num_boxes; b++) {
    MinMax box = source_boxes[b];
    float box_dist = distBoxPoint(box, target);
    if (box_dist > best_dist[K - 1])
      continue;

    uint32_t box_start = b * BOX_SIZE;
    uint32_t box_end = min(M, (b + 1) * BOX_SIZE);
    for (uint32_t i = box_start; i < box_end; i++) {
      uint32_t src_idx = source_indices[i];
      updateKBest<K>(target, source_points[src_idx], src_idx, best_dist,
                     best_idx);
    }
  }

  for (int j = 0; j < K; j++) {
    nearest_indices[idx * K + j] = best_idx[j];
  }
}

void KNN::knn_cross_3d(int M, float3 *source_points, int N,
                             float3 *target_points, uint32_t *nearestIndices,
                             int K) {
  if (M == 0 || N == 0)
    return;

  // 1. Compute bounding box for source points
  float3 *result;
  cudaMalloc(&result, sizeof(float3));
  size_t temp_storage_bytes;

  float3 init = {0, 0, 0}, minn, maxx;

  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, source_points, result,
                            M, CustomMin(), init);
  thrust::device_vector<char> temp_storage(temp_storage_bytes);

  cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes,
                            source_points, result, M, CustomMin(), init);
  cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

  cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes,
                            source_points, result, M, CustomMax(), init);
  cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

  // Handle degenerate bounding box
  if (maxx.x == minn.x)
    maxx.x = minn.x + 1.0f;
  if (maxx.y == minn.y)
    maxx.y = minn.y + 1.0f;
  if (maxx.z == minn.z)
    maxx.z = minn.z + 1.0f;

  // 2. Compute Morton codes for source points
  thrust::device_vector<uint32_t> morton(M);
  thrust::device_vector<uint32_t> morton_sorted(M);
  coord2Morton<<<(M + 255) / 256, 256>>>(M, source_points, minn, maxx,
                                          morton.data().get());

  // 3. Sort source by Morton code
  thrust::device_vector<uint32_t> indices(M);
  thrust::sequence(indices.begin(), indices.end());
  thrust::device_vector<uint32_t> indices_sorted(M);

  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, morton.data().get(),
      morton_sorted.data().get(), indices.data().get(),
      indices_sorted.data().get(), M);
  temp_storage.resize(temp_storage_bytes);

  cub::DeviceRadixSort::SortPairs(
      temp_storage.data().get(), temp_storage_bytes, morton.data().get(),
      morton_sorted.data().get(), indices.data().get(),
      indices_sorted.data().get(), M);

  // 4. Build bounding boxes for source
  uint32_t num_boxes = (M + BOX_SIZE - 1) / BOX_SIZE;
  thrust::device_vector<MinMax> boxes(num_boxes);
  buildBoxes3D<<<num_boxes, BOX_SIZE>>>(M, source_points,
                                        indices_sorted.data().get(),
                                        boxes.data().get());

  // 5. Launch crossKNN3D kernel (one thread per target point)
  uint32_t num_target_blocks = (N + 255) / 256;
  switch (K) {
  case 1:
    crossKNN3D<1><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 2:
    crossKNN3D<2><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 3:
    crossKNN3D<3><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 4:
    crossKNN3D<4><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 5:
    crossKNN3D<5><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 6:
    crossKNN3D<6><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 7:
    crossKNN3D<7><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 8:
    crossKNN3D<8><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 9:
    crossKNN3D<9><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 16:
    crossKNN3D<16><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 32:
    crossKNN3D<32><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  default:
    assert(false && "Unsupported K value (supported: 1-9, 16, 32)");
  }

  cudaFree(result);
}

void KNN::knn_cross_2d(int M, float2 *source_points, int N,
                             float2 *target_points, uint32_t *nearestIndices, int K) {
  // Handle edge case: no source points
  if (M == 0 || N == 0)
    return;

  // 1. Compute bounding box for source points
  float2 *result2d;
  cudaMalloc(&result2d, sizeof(float2));
  size_t temp_storage_bytes;

  float2 init2d = {0, 0}, minn2d, maxx2d;

  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, source_points,
                            result2d, M, CustomMin2D(), init2d);
  thrust::device_vector<char> temp_storage(temp_storage_bytes);

  cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes,
                            source_points, result2d, M, CustomMin2D(), init2d);
  cudaMemcpy(&minn2d, result2d, sizeof(float2), cudaMemcpyDeviceToHost);

  cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes,
                            source_points, result2d, M, CustomMax2D(), init2d);
  cudaMemcpy(&maxx2d, result2d, sizeof(float2), cudaMemcpyDeviceToHost);

  // Handle degenerate bounding box (all points at same location)
  if (maxx2d.x == minn2d.x)
    maxx2d.x = minn2d.x + 1.0f;
  if (maxx2d.y == minn2d.y)
    maxx2d.y = minn2d.y + 1.0f;

  // 2. Compute Morton codes for source points
  thrust::device_vector<uint32_t> morton(M);
  thrust::device_vector<uint32_t> morton_sorted(M);
  coord2Morton2DKernel<<<(M + 255) / 256, 256>>>(M, source_points, minn2d,
                                                  maxx2d, morton.data().get());

  // 3. Sort source by Morton code
  thrust::device_vector<uint32_t> indices(M);
  thrust::sequence(indices.begin(), indices.end());
  thrust::device_vector<uint32_t> indices_sorted(M);

  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, morton.data().get(),
      morton_sorted.data().get(), indices.data().get(),
      indices_sorted.data().get(), M);
  temp_storage.resize(temp_storage_bytes);

  cub::DeviceRadixSort::SortPairs(
      temp_storage.data().get(), temp_storage_bytes, morton.data().get(),
      morton_sorted.data().get(), indices.data().get(),
      indices_sorted.data().get(), M);

  // 4. Build bounding boxes for source
  uint32_t num_boxes = (M + BOX_SIZE - 1) / BOX_SIZE;
  thrust::device_vector<MinMax2D> boxes(num_boxes);
  buildBoxes2D<<<num_boxes, BOX_SIZE>>>(M, source_points,
                                        indices_sorted.data().get(),
                                        boxes.data().get());

  // 5. Launch crossKNN2D kernel (one thread per target point)
  uint32_t num_target_blocks = (N + 255) / 256;
  switch (K) {
  case 1:
    crossKNN2D<1><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 2:
    crossKNN2D<2><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 3:
    crossKNN2D<3><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 4:
    crossKNN2D<4><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 5:
    crossKNN2D<5><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 6:
    crossKNN2D<6><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 7:
    crossKNN2D<7><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 8:
    crossKNN2D<8><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 9:
    crossKNN2D<9><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 16:
    crossKNN2D<16><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  case 32:
    crossKNN2D<32><<<num_target_blocks, 256>>>(
        M, source_points, indices_sorted.data().get(), boxes.data().get(), N,
        target_points, nearestIndices);
    break;
  default:
    assert(false && "Unsupported K value (supported: 1-9, 16, 32)");
  }

  cudaFree(result2d);
}
