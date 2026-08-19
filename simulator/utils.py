"""
Utility functions for MPM simulation.

Contains mathematical helper functions that are reusable across different
parts of the simulator and not tied to specific material models.
"""

import taichi as ti


@ti.func
def norm(x, eps=1e-4):
    """L2 norm of x with an eps floor, keeping the gradient finite at x = 0."""
    return ti.sqrt(x.dot(x) + eps * eps)


@ti.func
def make_matrix_from_diag_3d(d):
    """3x3 diagonal matrix from a 3-vector (dtype inferred from d)."""
    return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
