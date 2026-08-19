"""
Per-particle volume estimation via Taichi-accelerated iterative P2G normalization.

ρ-free; uses the same quadratic B-spline kernel as MPM and caps volumes at dx³.
"""

import math

import taichi as ti
import torch
from knn import knn_idx


@torch.no_grad()
def compute_optimal_dx(xyz, k=8, percentile=50, min_dx=1e-3):
    """
    Unified grid spacing dx = (4pi/3)^(1/3) * percentile of k-th NN distance,
    shared by iP2G volume estimation and the MPM grid (k=8 targets ppc ~= 8).
    """
    indices = knn_idx(xyz.detach(), k=k)
    if indices.dim() == 1:  # knn_idx returns [N] for k=1, [N,k] for k>1
        indices = indices.unsqueeze(1)
    dx_vec = xyz.detach()[:, None, :] - xyz.detach()[indices]
    r = (dx_vec.square().sum(dim=-1) + 1e-12).sqrt()
    d_k = r.amax(dim=-1)
    factor = (4.0 * math.pi / 3.0) ** (1.0 / 3.0)
    q = torch.quantile(d_k.float(), percentile / 100.0).item()
    return max(min_dx, factor * q)


@ti.data_oriented
class TaichiVolumeComputer:
    """
    Taichi-accelerated P2G iterative volume computation. Fields are allocated
    once and reused; differentiable fields use needs_grad=True for Taichi autodiff.
    """

    def __init__(self, max_n_particles, max_grid_size=256, dtype=ti.f32):
        self.max_n_particles = max_n_particles
        self.max_grid_size = max_grid_size
        self.ti_dtype = dtype

        # Scalar parameters
        self.n_particles = ti.field(ti.i32, shape=())
        self.dx_val = ti.field(dtype, shape=())
        self.inv_dx_val = ti.field(dtype, shape=())
        self.dx3_val = ti.field(dtype, shape=())
        self.base_min = ti.Vector.field(3, dtype=ti.i32, shape=())

        # Particle fields
        self.pos = ti.Vector.field(
            3, dtype=dtype, shape=max_n_particles, needs_grad=True
        )
        self.volumes = ti.field(dtype=dtype, shape=max_n_particles, needs_grad=True)
        self.weight_sum = ti.field(dtype=dtype, shape=max_n_particles, needs_grad=True)
        self.n_eff = ti.field(dtype=dtype, shape=max_n_particles, needs_grad=True)

        self.base = ti.Vector.field(3, dtype=ti.i32, shape=max_n_particles)

        self.fx = ti.Vector.field(
            3, dtype=dtype, shape=max_n_particles, needs_grad=True
        )

        # w[p, dim] = vec3(w0, w1, w2) for that dimension
        self.w = ti.Vector.field(
            3, dtype=dtype, shape=(max_n_particles, 3), needs_grad=True
        )

        # Grid fields (dense 3D)
        gs = max_grid_size
        self.grid_count = ti.field(dtype=dtype, shape=(gs, gs, gs), needs_grad=True)
        self.grid_vol = ti.field(dtype=dtype, shape=(gs, gs, gs), needs_grad=True)
        self.node_corr = ti.field(dtype=dtype, shape=(gs, gs, gs), needs_grad=True)

    # Data transfer kernels

    @ti.kernel
    def _set_positions(self, xyz: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                self.pos[p][d] = xyz[p, d]

    @ti.kernel
    def _get_volumes(self, out: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            out[p] = self.volumes[p]

    @ti.kernel
    def _set_volume_grads(self, grad: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            self.volumes.grad[p] = grad[p]

    @ti.kernel
    def _get_pos_grads(self, out: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                out[p, d] = self.pos.grad[p][d]

    @ti.kernel
    def _restore_volumes(self, snap: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            self.volumes[p] = snap[p]

    # Clear kernels

    @ti.kernel
    def _clear_grid_count(self):
        for I in ti.grouped(self.grid_count):
            self.grid_count[I] = 0.0

    @ti.kernel
    def _clear_grid_vol(self):
        for I in ti.grouped(self.grid_vol):
            self.grid_vol[I] = 0.0

    @ti.kernel
    def _clear_node_corr(self):
        for I in ti.grouped(self.node_corr):
            self.node_corr[I] = 0.0

    @ti.kernel
    def _clear_particle_fields(self):
        for p in range(self.n_particles[None]):
            self.weight_sum[p] = 0.0
            self.n_eff[p] = 0.0
            self.volumes[p] = 0.0
            self.fx[p] = ti.Vector.zero(self.ti_dtype, 3)
            self.base[p] = ti.Vector.zero(ti.i32, 3)
            for d in ti.static(range(3)):
                self.w[p, d] = ti.Vector.zero(self.ti_dtype, 3)

    @ti.kernel
    def _clear_all_grads(self):
        for p in range(self.n_particles[None]):
            self.pos.grad[p] = ti.Vector.zero(self.ti_dtype, 3)
            self.volumes.grad[p] = 0.0
            self.weight_sum.grad[p] = 0.0
            self.n_eff.grad[p] = 0.0
            self.fx.grad[p] = ti.Vector.zero(self.ti_dtype, 3)
            for d in ti.static(range(3)):
                self.w.grad[p, d] = ti.Vector.zero(self.ti_dtype, 3)

    @ti.kernel
    def _clear_grid_grads(self):
        for I in ti.grouped(self.grid_count):
            self.grid_count.grad[I] = 0.0
        for I in ti.grouped(self.grid_vol):
            self.grid_vol.grad[I] = 0.0
        for I in ti.grouped(self.node_corr):
            self.node_corr.grad[I] = 0.0

    # Forward kernels

    @ti.kernel
    def compute_base_and_weights(self):
        """Compute base grid indices, fractional positions, and B-spline weights."""
        for p in range(self.n_particles[None]):
            xp_scaled = self.pos[p] * self.inv_dx_val[None]
            base_raw = ti.floor(xp_scaled - 0.5).cast(int)
            self.base[p] = base_raw - self.base_min[None] + 1

            fx = xp_scaled - base_raw.cast(self.ti_dtype)
            self.fx[p] = fx

            # Quadratic B-spline weights per dimension
            for d in ti.static(range(3)):
                self.w[p, d][0] = 0.5 * (1.5 - fx[d]) ** 2
                self.w[p, d][1] = 0.75 - (fx[d] - 1.0) ** 2
                self.w[p, d][2] = 0.5 * (fx[d] - 0.5) ** 2

            ws = self.ti_dtype(0.0)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                ws += self.w[p, 0][i] * self.w[p, 1][j] * self.w[p, 2][k]
            self.weight_sum[p] = ws

    @ti.kernel
    def p2g_count(self):
        """P2G: scatter particle weights to grid_count."""
        for p in range(self.n_particles[None]):
            b = self.base[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                weight = self.w[p, 0][i] * self.w[p, 1][j] * self.w[p, 2][k]
                self.grid_count[b[0] + i, b[1] + j, b[2] + k] += weight

    @ti.kernel
    def g2p_neff(self):
        """G2P: gather effective particle count from grid_count."""
        for p in range(self.n_particles[None]):
            b = self.base[p]
            acc = self.ti_dtype(0.0)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                weight = self.w[p, 0][i] * self.w[p, 1][j] * self.w[p, 2][k]
                acc += weight * self.grid_count[b[0] + i, b[1] + j, b[2] + k]
            self.n_eff[p] = acc / ti.max(self.weight_sum[p], 1e-2)

    @ti.kernel
    def init_volumes(self):
        """Initialize volumes: V_i = clamp(dx^3 / n_eff_i, max=dx^3)."""
        for p in range(self.n_particles[None]):
            dx3 = self.dx3_val[None]
            self.volumes[p] = ti.min(dx3 / ti.max(self.n_eff[p], 0.1), dx3)

    @ti.kernel
    def p2g_volumes(self):
        """P2G: scatter weighted volumes to grid_vol."""
        for p in range(self.n_particles[None]):
            b = self.base[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                weight = self.w[p, 0][i] * self.w[p, 1][j] * self.w[p, 2][k]
                self.grid_vol[b[0] + i, b[1] + j, b[2] + k] += weight * self.volumes[p]

    @ti.kernel
    def compute_node_correction(self):
        """Capped node correction: clamp(dx^3 / grid_vol, max=1.0)."""
        for I in ti.grouped(self.grid_vol):
            dx3 = self.dx3_val[None]
            gv = ti.max(self.grid_vol[I], 1e-2 * dx3)
            self.node_corr[I] = ti.min(dx3 / gv, 1.0)

    @ti.kernel
    def g2p_correction(self):
        """G2P: gather correction factor and update volumes."""
        for p in range(self.n_particles[None]):
            dx3 = self.dx3_val[None]
            b = self.base[p]
            acc = self.ti_dtype(0.0)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                weight = self.w[p, 0][i] * self.w[p, 1][j] * self.w[p, 2][k]
                acc += weight * self.node_corr[b[0] + i, b[1] + j, b[2] + k]
            f = acc / ti.max(self.weight_sum[p], 1e-2)
            self.volumes[p] = ti.min(self.volumes[p] * f, dx3)

    # High-level forward / backward

    def run_forward(self, xyz, dx, inv_dx, dx3, base_min, n_iter):
        """
        Full forward pass. n_iter must match run_backward (backward indexes the
        returned snapshots). Returns a list of n_iter [N] tensors holding the
        volume state before each refinement iteration.
        """
        N = xyz.shape[0]
        self.n_particles[None] = N
        self.dx_val[None] = dx
        self.inv_dx_val[None] = inv_dx
        self.dx3_val[None] = dx3
        self.base_min[None] = [int(base_min[0]), int(base_min[1]), int(base_min[2])]

        self._clear_particle_fields()
        self._clear_grid_count()

        self._set_positions(xyz)

        self.compute_base_and_weights()
        self.p2g_count()
        self.g2p_neff()
        self.init_volumes()

        # Iterative refinement with snapshots
        vol_snapshots = []
        for _t in range(n_iter):
            snap = torch.empty(N, dtype=torch.float32, device="cuda")
            self._get_volumes(snap)
            vol_snapshots.append(snap)

            self._clear_grid_vol()
            self.p2g_volumes()
            self.compute_node_correction()
            self.g2p_correction()

        return vol_snapshots

    def run_backward(self, grad_volumes, vol_snapshots, n_iter):
        """
        Full backward pass; n_iter must match run_forward.
        Returns grad_xyz as an [N, 3] tensor of position gradients.
        """
        N = grad_volumes.shape[0]

        self._clear_all_grads()
        self._clear_grid_grads()

        self._set_volume_grads(grad_volumes)

        for t in reversed(range(n_iter)):
            self._restore_volumes(vol_snapshots[t])

            # Re-run forward kernels for this iteration to reconstruct grid state
            self._clear_grid_vol()
            self._clear_node_corr()
            self.p2g_volumes()
            self.compute_node_correction()

            # Clear grid grads before backward kernels
            self._clear_grid_grads()

            # Backward kernels in reverse of forward order
            self.g2p_correction.grad()
            self.compute_node_correction.grad()
            self.p2g_volumes.grad()

        self.init_volumes.grad()
        self.g2p_neff.grad()
        self.p2g_count.grad()
        self.compute_base_and_weights.grad()

        grad_xyz = torch.empty(N, 3, dtype=torch.float32, device="cuda")
        self._get_pos_grads(grad_xyz)
        return grad_xyz


# Holds at most one lazily-built computer, grown on demand and reused when
# compute_induced_volumes is called without a computer.
computer_cache: "list[TaichiVolumeComputer]" = []


class TaichiInducedVolumes(torch.autograd.Function):
    """PyTorch autograd bridge for Taichi volume computation."""

    @staticmethod
    def forward(ctx, xyz, computer, n_iter, dx):
        """Compute induced volumes via Taichi; returns [N] per-particle volumes."""
        N = xyz.shape[0]

        with torch.no_grad():
            inv_dx = 1.0 / dx
            dx3 = dx**3

            xp_scaled = xyz.detach() * inv_dx
            base_raw = torch.floor(xp_scaled - 0.5).long()
            base_min = base_raw.min(dim=0).values

            base_max = base_raw.max(dim=0).values
            grid_needed = (base_max - base_min + 5).max().item()
            if grid_needed > computer.max_grid_size:
                raise RuntimeError(
                    f"Volume grid needs {grid_needed} cells but max is "
                    f"{computer.max_grid_size}. Increase max_grid_size."
                )

        vol_snapshots = computer.run_forward(xyz, dx, inv_dx, dx3, base_min, n_iter)

        volumes = torch.empty(N, dtype=torch.float32, device="cuda")
        computer._get_volumes(volumes)

        ctx.computer = computer
        ctx.n_iter = n_iter
        ctx.vol_snapshots = vol_snapshots
        ctx.save_for_backward(xyz)
        ctx.dx = dx
        ctx.inv_dx = inv_dx
        ctx.dx3 = dx3
        ctx.base_min = base_min

        return volumes

    @staticmethod
    def backward(ctx, grad_volumes):
        """Compute gradients w.r.t. positions via Taichi backward."""
        computer = ctx.computer
        (xyz,) = ctx.saved_tensors

        # Re-run forward to restore Taichi field state (needed for .grad() kernels)
        computer.run_forward(xyz, ctx.dx, ctx.inv_dx, ctx.dx3, ctx.base_min, ctx.n_iter)

        grad_xyz = computer.run_backward(
            grad_volumes.contiguous(), ctx.vol_snapshots, ctx.n_iter
        )

        return grad_xyz, None, None, None


def compute_induced_volumes(xyz, dx, n_iter=5, computer=None):
    """
    Differentiable per-particle volumes ([N]) via Taichi iterative P2G.
    dx must be the unified spacing from compute_optimal_dx, computed once —
    do not recompute per call site.
    """
    if computer is None:
        n = xyz.shape[0]
        if not computer_cache or computer_cache[0].max_n_particles < n:
            computer_cache[:] = [TaichiVolumeComputer(max_n_particles=n + 8192)]
        computer = computer_cache[0]

    return TaichiInducedVolumes.apply(xyz, computer, n_iter, dx)
