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

#include <torch/extension.h>
#include "spatial.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_idx", &knn_idx, "KNN index", py::arg("points"), py::arg("k") = 1);
  m.def("knn_cross_2d", &knn_cross_2d,
        "Cross-group KNN for 2D points (variable K). Returns index into source "
        "for each target point.",
        py::arg("source"), py::arg("target"), py::arg("k") = 1);
  m.def("knn_cross_3d", &knn_cross_3d,
        "Cross-group KNN for 3D points (variable K). Returns index into source "
        "for each target point.",
        py::arg("source"), py::arg("target"), py::arg("k") = 1);
}
