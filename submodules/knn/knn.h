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

#ifndef KNN_H_INCLUDED
#define KNN_H_INCLUDED

#include <cstdint>

class KNN {
public:
  static void knn_idx(int P, float3 *points, uint32_t *nearestIndices,
                      int K = 1);
  static void knn_cross_2d(int M, float2 *source_points, int N,
                           float2 *target_points, uint32_t *nearestIndices,
                           int K = 1);
  static void knn_cross_3d(int M, float3 *source_points, int N,
                           float3 *target_points, uint32_t *nearestIndices,
                           int K = 1);
};

#endif
