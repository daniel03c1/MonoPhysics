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
import os
import trimesh as tm
from os import makedirs


def mkdir_p(folder_path):
    """Create a directory and its parents, like `mkdir -p`."""
    makedirs(folder_path, exist_ok=True)


def write_particles(particles, idx, path, name="", vertex_colors=None):
    if type(particles) == np.ndarray:
        numpy_array = particles
    else:
        numpy_array = particles.cpu().detach().numpy()
    if not os.path.exists(os.path.join(path, "mpm")):
        mkdir_p(os.path.join(path, "mpm"))
    if vertex_colors is None:
        tm.Trimesh(numpy_array).export(os.path.join(path, f"mpm/{name}_{idx}.ply"))
    else:
        tm.Trimesh(numpy_array, vertex_colors=vertex_colors).export(
            os.path.join(path, f"mpm/{name}_{idx}.ply")
        )
