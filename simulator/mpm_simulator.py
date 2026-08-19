import math
import taichi as ti
import torch

import simulator.materials as materials


@ti.data_oriented
class MPMSimulator:
    # Surfaces
    surface_sticky = 0
    surface_slip = 1
    surface_separate = 2
    surface_friction = 3

    # Coulomb mu for surface_friction; critical slope is atan(mu) = 21.8 deg
    default_floor_friction = 0.4

    elasticity = 10
    viscous_fluid = 11
    plasticine = 12
    drucker_prager = 13
    neo_hookean = 14
    non_newtonian = 15
    sand = 17
    plasticine_corotated = 18

    # Pre-compute torch dtype mappings to avoid Taichi scope violations
    _TORCH_FLOAT32 = torch.float32
    _TORCH_FLOAT64 = torch.float64

    def __init__(
        self,
        dtype,
        dt,
        frame_dt,
        dx,
        max_n_particles,
        args,
        gravity=[0, -9.8, 0],
        material=elasticity,
        cuda_chunk_size=200,
        **kargs,
    ):
        dim = self.dim = 3
        self.dtype = dtype
        self.material = material
        self.particle = ti.root.dense(ti.i, max_n_particles)
        self.n_particles = ti.field(ti.i32, shape=())
        # Compile-time loop bound for p2g/g2p (see comment in p2g).
        self.max_n_particles = max_n_particles

        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())
        self.inv_dx_mul_4 = ti.field(self.dtype, shape=())
        self.inv_dx_sq_mul_4 = ti.field(self.dtype, shape=())
        self.voxel_offset = ti.field(self.dtype, shape=())
        self.grid_vol = ti.field(self.dtype, shape=())
        self.max_vol = ti.field(self.dtype, shape=())
        self.mass_eps = ti.field(self.dtype, shape=())
        self.cfl_v_threshold = ti.field(self.dtype, shape=())

        self.frame_dt = frame_dt
        self.args = args
        self.cfl_satisfy = ti.field(ti.i8, shape=())
        self.v_max_observed = ti.field(self.dtype, shape=())

        # Read-only diagnostic: sigma components clamped at SIG_MIN/SIG_MAX in
        # project_F, accumulated per forward rollout (reset in reset()); only counted
        # when count_sig_clamps_enabled (zero GPU cost otherwise).
        # _clamp_substep_count is the denominator.
        self.n_sig_clamped_lo = ti.field(ti.i32, shape=())
        self.n_sig_clamped_hi = ti.field(ti.i32, shape=())
        self.count_sig_clamps_enabled = False
        self._clamp_substep_count = 0

        self.dt = ti.field(self.dtype, shape=())
        self.dt_ori = ti.field(self.dtype, shape=())
        self.inv_dt = ti.field(self.dtype, shape=())
        self.n_substeps = ti.field(ti.i32, shape=())
        self.cuda_chunk_size = cuda_chunk_size

        print("cuda_chunk_size: ", cuda_chunk_size)

        self.step_particle = self.particle.dense(ti.j, cuda_chunk_size + 1)
        self.x = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)
        self.v = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)
        self.C = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.F = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)

        self.step_particle.place(
            self.x,
            self.x.grad,
            self.v,
            self.v.grad,
            self.C,
            self.C.grad,
            self.F,
            self.F.grad,
        )

        # material
        self.vol = ti.field(self.dtype, needs_grad=True)
        self.mass = ti.field(self.dtype, needs_grad=True)
        self.rho = ti.field(self.dtype)

        self.lam = ti.field(dtype=self.dtype, needs_grad=True)
        self.mu = ti.field(dtype=self.dtype, needs_grad=True)
        self.yield_stress = ti.field(dtype=self.dtype, needs_grad=True)
        self.plastic_viscosity = ti.field(dtype=self.dtype, needs_grad=True)
        self.friction_alpha = ti.field(dtype=self.dtype, needs_grad=True)
        self.Jp = ti.field(dtype=self.dtype, needs_grad=True)

        self.F_tmp = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.U = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.V = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.sig = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.sig_out = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)

        self.gravity = ti.Vector.field(dim, self.dtype, shape=(), needs_grad=True)

        self.particle.place(
            self.vol,
            self.vol.grad,
            self.mass,
            self.mass.grad,
            self.rho,
            self.lam,
            self.lam.grad,
            self.mu,
            self.mu.grad,
            self.yield_stress,
            self.yield_stress.grad,
            self.plastic_viscosity,
            self.plastic_viscosity.grad,
            self.friction_alpha,
            self.friction_alpha.grad,
            self.Jp,
            self.Jp.grad,
            self.F_tmp,
            self.F_tmp.grad,
            self.U,
            self.U.grad,
            self.V,
            self.V.grad,
            self.sig,
            self.sig.grad,
            self.sig_out,
            self.sig_out.grad,
        )

        grid_size = 4096
        offset = self.offset = tuple(-grid_size // 2 for _ in range(3))
        self.offset_vec = ti.Vector(list(offset), ti.i32)
        grid_block_size = 128
        leaf_block_size = 4

        grid = self.grid = ti.root.pointer(ti.ijk, grid_size // grid_block_size)
        block = grid.pointer(ti.ijk, grid_block_size // leaf_block_size)

        self.grid_m = ti.field(dtype=self.dtype, needs_grad=True)
        self.grid_v_in = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)
        self.grid_v_out = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)

        def block_component(c):
            block.dense(ti.ijk, leaf_block_size).place(c, c.grad, offset=offset)

        block_component(self.grid_m)
        block_component(self.grid_v_in)
        block_component(self.grid_v_out)

        self.dt[None] = dt
        self.n_substeps[None] = round(frame_dt / dt)
        self.dt_ori[None] = dt

        # Seeded so the field is never empty; Estimator.initialize() overwrites it
        # each rollout from the learnable gravity parameter.
        self.gravity[None] = gravity

        self.ground_response = None
        self.cached_states = []

        # Single ground plane, field-based so it stays differentiable w.r.t. the boundary
        self.collider_point = ti.Vector.field(
            dim, dtype=self.dtype, shape=(), needs_grad=True
        )
        self.collider_normal = ti.Vector.field(
            dim, dtype=self.dtype, shape=(), needs_grad=True
        )

        self.set_dx(dx)
        self.set_colliders(args)
        self.voxel_offset[None] = self.compute_voxel_offset()
        self.collision_detection = ti.field(ti.i8, shape=())

        # Material function pointers — dispatch resolved here, no runtime branching.
        self._project_F_func = materials.PROJECT_F_FUNCTIONS.get(material)
        self._compute_stress_func = materials.COMPUTE_STRESS_FUNCTIONS.get(material)

        if self._project_F_func is None:
            raise ValueError(f"Unknown material type: {material}")
        if self._compute_stress_func is None:
            raise ValueError(f"Unknown material type: {material}")

        self.n_particles[None] = 0

    def reset(self):
        self.grid.deactivate_all()
        self.cfl_satisfy[None] = True
        self.v_max_observed[None] = 0.0

        self.collision_detection[None] = 0

        self.cached_states.clear()
        self.clear_grads()

        self.n_sig_clamped_lo[None] = 0
        self.n_sig_clamped_hi[None] = 0
        self._clamp_substep_count = 0

    def set_dx(self, dx):
        self.dx[None] = dx
        self.inv_dx[None] = 1.0 / dx
        self.inv_dx_mul_4[None] = 4.0 / dx
        self.inv_dx_sq_mul_4[None] = 4.0 / dx / dx
        self.grid_vol[None] = self.dx[None] ** 3
        self.max_vol[None] = self.grid_vol[None]

        # Grid-mass floor from a reference rho of 1000 (not the learnable rho).
        self.mass_eps[None] = 1e-4 * 1000.0 * self.grid_vol[None]

        self._update_cfl_threshold()

    def compute_voxel_offset(self):
        """
        Align the grid with the ground plane: grid node I sits at world pos
        (I - voxel_offset) * dx, and offset = -frac puts a node exactly at the
        boundary (dist = 0) for grid_op's hard wall.
        """
        point = self.collider_point[None]
        normal = self.collider_normal[None]

        abs_normal = [abs(float(normal[i])) for i in range(3)]
        dominant_axis = abs_normal.index(max(abs_normal))

        boundary_pos = float(point[dominant_axis])
        dx = float(self.dx[None])

        # always in [0, 1)
        frac_part = (boundary_pos / dx) - math.floor(boundary_pos / dx)

        return -frac_part

    def check_collision(self):
        """True if any particle has touched a collider since the last reset()."""
        return bool(self.collision_detection[None])

    def set_dt(self, dt):
        self.dt[None] = dt
        self.inv_dt[None] = 1 / dt
        self.n_substeps[None] = round(self.frame_dt / dt)
        self._update_cfl_threshold()

    def _update_cfl_threshold(self):
        """
        Velocity CFL threshold at Courant number 0.5: v_max^2 < (dx / (2*dt))^2.
        Depends on both dx and dt, so both setters refresh it.
        """
        self.cfl_v_threshold[None] = (self.dx[None] * 0.5 / self.dt[None]) ** 2

    def set_particles(
        self,
        particles,
        velocities,
        particle_rho,
        volumes,
        particle_mu,
        particle_lam,
        particle_yield_stress,
        particle_plastic_viscosity,
        particle_friction_alpha,
        dx,
    ):
        """
        Upload per-particle state from torch tensors into Taichi fields.
        dx must be the unified grid spacing computed once via compute_optimal_dx
        and stored on phys_args.voxel_size.
        """
        self.set_dx(dx)
        self.voxel_offset[None] = self.compute_voxel_offset()
        self._set_particles_kernel(
            particles,
            velocities,
            particle_rho,
            volumes,
            particle_mu,
            particle_lam,
            particle_yield_stress,
            particle_plastic_viscosity,
            particle_friction_alpha,
        )

    @ti.kernel
    def _set_particles_kernel(
        self,
        particles: ti.types.ndarray(),
        velocities: ti.types.ndarray(),
        particle_rho: ti.types.ndarray(),
        volumes: ti.types.ndarray(),
        particle_mu: ti.types.ndarray(),
        particle_lam: ti.types.ndarray(),
        particle_yield_stress: ti.types.ndarray(),
        particle_plastic_viscosity: ti.types.ndarray(),
        particle_friction_alpha: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                self.x[p, 0][d] = particles[p, d]
                self.v[p, 0][d] = velocities[p, d]
            self.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            self.F[p, 0] = ti.Matrix.identity(self.dtype, 3)

            self.rho[p] = particle_rho[p]
            self.vol[p] = volumes[p]

            self.mu[p] = particle_mu[p]
            self.lam[p] = particle_lam[p]

            self.yield_stress[p] = particle_yield_stress[p]
            self.plastic_viscosity[p] = particle_plastic_viscosity[p]
            self.friction_alpha[p] = particle_friction_alpha[p]
            self.Jp[p] = 0.0

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.n_particles[None]):
            self.mass[p] = self.rho[p] * self.vol[p]

    def clear_grads(self):
        self.x.grad.fill(0)
        self.v.grad.fill(0)
        self.C.grad.fill(0)
        self.F.grad.fill(0)

        self.vol.grad.fill(0)
        self.mass.grad.fill(0)

        self.gravity.grad.fill(0)

        self.lam.grad.fill(0)
        self.mu.grad.fill(0)
        self.yield_stress.grad.fill(0)
        self.plastic_viscosity.grad.fill(0)
        self.friction_alpha.grad.fill(0)
        self.Jp.grad.fill(0)

    def clear_svd_grads(self):
        self.F_tmp.grad.fill(0)
        self.U.grad.fill(0)
        self.V.grad.fill(0)
        self.sig.grad.fill(0)
        self.sig_out.grad.fill(0)

    @ti.kernel
    def compute_F_tmp(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            if ti.static(self.material == self.viscous_fluid):
                self.F_tmp[p][0, 0] = (
                    1.0 + self.dt[None] * self.C[p, s].trace()
                ) * self.F[p, s][0, 0]
            else:
                self.F_tmp[p] = (
                    ti.Matrix.identity(self.dtype, self.dim)
                    + self.dt[None] * self.C[p, s]
                ) @ self.F[p, s]

    @ti.kernel
    def svd(self):
        for p in range(self.n_particles[None]):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p])

    @ti.kernel
    def svd_grad(self):
        for p in range(self.n_particles[None]):
            self.F_tmp.grad[p] += self.backward_svd(
                self.U.grad[p],
                self.sig.grad[p],
                self.V.grad[p],
                self.U[p],
                self.sig[p],
                self.V[p],
            )
            assert not ti.math.isnan(self.F_tmp.grad[p]).any()

    @ti.kernel
    def project_F(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            F_new, sig_out, Jp_new = ti.static(self._project_F_func)(
                self.F_tmp[p],
                self.U[p],
                self.sig[p],
                self.V[p],
                self.mu[p],
                self.lam[p],
                self.yield_stress[p],
                self.plastic_viscosity[p],
                self.friction_alpha[p],
                self.dt[None],
                self.Jp[p],
            )

            self.F[p, s + 1] = F_new
            self.sig_out[p] = sig_out
            self.Jp[p] = Jp_new

    @ti.kernel
    def count_sig_clamps(self):
        """
        Diagnostic: tally sigma components clamped by clamp_sig this substep (same
        pre-clamp singular values project_F sees); never differentiated. Counts from
        pop_from_memory's replay land after the fwd report, reset on initialize().
        """
        for p in range(self.n_particles[None]):
            for d in ti.static(range(self.dim)):
                s_d = self.sig[p][d, d]
                if s_d < materials.SIG_MIN:
                    self.n_sig_clamped_lo[None] += 1
                elif s_d > materials.SIG_MAX:
                    self.n_sig_clamped_hi[None] += 1

    @ti.func
    def smoothed_reciprocal(self, a, eps=1e-3):
        # Lorentzian-smoothed reciprocal: bounded at 1/(2*eps), vanishes as a -> 0.
        # Lets backward_svd safely invert (s_j - s_i) for near-degenerate singular values.
        return a / (a * a + eps * eps)

    @ti.func
    def backward_svd(self, gu, gsigma, gv, u, sig, v):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = v.transpose()
        ut = u.transpose()
        sigma_term = u @ gsigma @ vt
        s = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]) ** 2
        inv_sig_gap = ti.Matrix.zero(self.dtype, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i != j:
                inv_sig_gap[i, j] = self.smoothed_reciprocal(s[j] - s[i])
        u_term = u @ ((inv_sig_gap * (ut @ gu - gu.transpose() @ u)) @ sig) @ vt
        v_term = u @ (sig @ ((inv_sig_gap * (vt @ gv - gv.transpose() @ v)) @ vt))
        return u_term + v_term + sigma_term

    @ti.func
    def _base_in_grid(self, base):
        # Grid indices valid: [-2048, 2047]. Stencil accesses base..base+2.
        # So base must be in [-2048, 2045] per dimension.
        lo = self.offset_vec
        hi = -self.offset_vec - 3
        return (base >= lo).all() and (base <= hi).all()

    @ti.kernel
    def p2g(self, s: ti.i32):
        # Loop bound must be compile-time constant: with a runtime bound
        # (n_particles[None]) Taichi launches kernels that atomically scatter
        # into the sparse grid ~12x slower (p2g fwd and g2p's autodiff reverse).
        for p in range(self.max_n_particles):
            if p < self.n_particles[None]:
                xp_scaled = self.x[p, s] * self.inv_dx[None] + self.voxel_offset[None]
                base = ti.floor(xp_scaled - 0.5).cast(int)
                if self._base_in_grid(base):
                    fx = xp_scaled - base.cast(self.dtype)

                    # Quadratic kernels [http://mpm.graphics Eqn. 123]
                    w = [
                        0.5 * (1.5 - fx) ** 2,
                        0.75 - (fx - 1) ** 2,
                        0.5 * (fx - 0.5) ** 2,
                    ]
                    new_F = self.F[p, s + 1]

                    stress = ti.static(self._compute_stress_func)(
                        new_F,
                        self.C[p, s],
                        self.U[p],
                        self.sig_out[p],
                        self.mu[p],
                        self.lam[p],
                        self.dt[None],
                    )

                    stress = (
                        -self.dt[None] * self.vol[p] * self.inv_dx_sq_mul_4[None]
                    ) * stress
                    affine = stress + self.mass[p] * self.C[p, s]

                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            for k in ti.static(range(3)):
                                offset = ti.Vector([i, j, k])
                                dpos = (
                                    ti.cast(ti.Vector([i, j, k]), self.dtype) - fx
                                ) * self.dx[None]
                                weight = w[i][0] * w[j][1] * w[k][2]
                                idx = base + offset

                                self.grid_v_in[idx] += weight * (
                                    self.mass[p] * self.v[p, s] + affine @ dpos
                                )
                                self.grid_m[idx] += weight * self.mass[p]

    @ti.kernel
    def grid_op(self, s: ti.i32):
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]

            if m > 0:
                inv_m = 1 / (m + self.mass_eps[None])
                v_out = self.grid_v_in[I] * inv_m + self.dt[None] * self.gravity[None]

                v_out = self.ground_response(I, v_out)

                self.grid_v_out[I] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        # Compile-time loop bound: see comment in p2g (matters here for the
        # autodiff reverse kernel, which scatters into grid_v_out.grad).
        for p in range(self.max_n_particles):
            if p < self.n_particles[None]:
                xp_scaled = self.x[p, f] * self.inv_dx[None] + self.voxel_offset[None]
                base = ti.floor(xp_scaled - 0.5).cast(int)
                if self._base_in_grid(base):
                    fx = xp_scaled - base.cast(self.dtype)
                    w = [
                        0.5 * (1.5 - fx) ** 2,
                        0.75 - (fx - 1.0) ** 2,
                        0.5 * (fx - 0.5) ** 2,
                    ]
                    new_v = ti.Vector.zero(self.dtype, self.dim)
                    new_C = ti.Matrix.zero(self.dtype, self.dim, self.dim)
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            for k in ti.static(range(3)):
                                dpos = ti.cast(ti.Vector([i, j, k]), self.dtype) - fx

                                g_v = self.grid_v_out[
                                    base[0] + i, base[1] + j, base[2] + k
                                ]
                                weight = w[i][0] * w[j][1] * w[k][2]
                                new_v += weight * g_v
                                new_C += (
                                    weight
                                    * g_v.outer_product(dpos)
                                    * self.inv_dx_mul_4[None]
                                )

                    self.v[p, f + 1] = new_v
                    self.x[p, f + 1] = self.x[p, f] + self.dt[None] * self.v[p, f + 1]
                    self.C[p, f + 1] = new_C
                else:
                    # Particle outside grid — freeze in place with zero velocity
                    self.v[p, f + 1] = ti.Vector.zero(self.dtype, self.dim)
                    self.x[p, f + 1] = self.x[p, f]
                    self.C[p, f + 1] = ti.Matrix.zero(self.dtype, self.dim, self.dim)

    @ti.kernel
    def check_cfl(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            if ti.math.isnan(self.v[p, s]).any():
                self.cfl_satisfy[None] = ti.cast(0, ti.i8)
            v_sq = self.v[p, s].dot(self.v[p, s])
            if v_sq > self.cfl_v_threshold[None]:
                self.cfl_satisfy[None] = ti.cast(0, ti.i8)
                ti.atomic_max(self.v_max_observed[None], ti.sqrt(v_sq))

    @ti.kernel
    def detect_collision(self, s: ti.i32):
        """
        Set the sticky collision flag if any particle is within 0.5*dx of the ground.
        Pure control-flow signal gating material-opt scheduling (_first_collision_frame):
        reads x, writes a needs_grad=False flag, never runs in substep_grad — no gradient.
        """
        thresh = 0.5 * self.dx[None]
        for p in range(self.n_particles[None]):
            dist = (self.x[p, s] - self.collider_point[None]).dot(
                self.collider_normal[None]
            )
            if dist < thresh:
                self.collision_detection[None] = ti.cast(1, ti.i8)

    @ti.kernel
    def zero_grid(self):
        # Zero active cells instead of grid.deactivate_all() per substep (deactivation
        # churn ~10% of an iteration). Cells stay active — bounded by the trajectory
        # envelope, freed in reset() — and grid_op skips m == 0 cells, so results match.
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0
            self.grid_m.grad[I] = 0
            for d in ti.static(range(3)):
                self.grid_v_in[I][d] = 0
                self.grid_v_in.grad[I][d] = 0
                self.grid_v_out[I][d] = 0
                self.grid_v_out.grad[I][d] = 0

    def substep(self, s, cache=True):
        """Execute one forward substep (s = global substep index)."""
        local_index = s % self.cuda_chunk_size
        self.zero_grid()
        self.compute_F_tmp(local_index)
        self.svd()
        self.project_F(local_index)
        if self.count_sig_clamps_enabled:
            self.count_sig_clamps()
            self._clamp_substep_count += 1

        self.p2g(local_index)
        self.grid_op(local_index)
        self.g2p(local_index)
        self.check_cfl(local_index + 1)
        self.detect_collision(local_index + 1)

        if (local_index == self.cuda_chunk_size - 1) and cache:
            self.push_to_memory()

    def substep_grad(self, s):
        """Execute one backward substep (s = global substep index)."""
        local_index = s % self.cuda_chunk_size
        if local_index == self.cuda_chunk_size - 1:
            self.pop_from_memory()
        self.zero_grid()
        self.compute_F_tmp(local_index)
        self.svd()
        self.project_F(local_index)
        self.p2g(local_index)
        self.grid_op(local_index)

        self.clear_svd_grads()
        self.g2p.grad(local_index)
        self.grid_op.grad(local_index)
        self.p2g.grad(local_index)
        self.project_F.grad(local_index)
        self.svd_grad()
        self.compute_F_tmp.grad(local_index)

    @ti.kernel
    def get_x(self, f: ti.i32, x: ti.types.ndarray()):
        local_index = (f * self.n_substeps[None]) % self.cuda_chunk_size
        for i in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                x[i, d] = ti.cast(self.x[i, local_index][d], ti.f32)

    @ti.kernel
    def read_x_grad_slot(self, s: ti.i32, out: ti.types.ndarray()):
        """
        Read x.grad at substep slot s into an [N, 3] f32 tensor. Read-only
        diagnostic: extracts one slot of the chunked grad field, modifies nothing.
        """
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                out[p, d] = ti.cast(self.x.grad[p, s][d], ti.f32)

    @ti.kernel
    def read_xv_grad_slot0(self, x_out: ti.types.ndarray(), v_out: ti.types.ndarray()):
        """
        Read x.grad and v.grad at slot 0 into [N, 3] f32 tensors, avoiding a
        full-field to_torch() that would materialize all cuda_chunk_size+1 slots.
        """
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                x_out[p, d] = ti.cast(self.x.grad[p, 0][d], ti.f32)
                v_out[p, d] = ti.cast(self.v.grad[p, 0][d], ti.f32)

    def advance(self, f):
        """Advance simulation by one frame (forward pass)."""
        for i in range(self.n_substeps[None] * (f - 1), self.n_substeps[None] * f):
            if self.cfl_satisfy[None]:
                self.substep(i, cache=True)

    def advance_grad(self, f):
        """Advance simulation by one frame (backward pass)."""
        for i in reversed(
            range(self.n_substeps[None] * (f - 1), self.n_substeps[None] * f)
        ):
            if self.cfl_satisfy[None]:
                self.substep_grad(i)

    @ti.kernel
    def get_state_chunk(
        self,
        x: ti.types.ndarray(),
        v: ti.types.ndarray(),
        C: ti.types.ndarray(),
        F: ti.types.ndarray(),
        Jp: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            Jp[p] = self.Jp[p]
            for i in ti.static(range(self.dim)):
                x[p, i] = self.x[p, 0][i]
                v[p, i] = self.v[p, 0][i]

                for j in ti.static(range(self.dim)):
                    C[p, i, j] = self.C[p, 0][i, j]
                    F[p, i, j] = self.F[p, 0][i, j]

        for p in range(self.n_particles[None]):
            self.x[p, 0] = self.x[p, self.cuda_chunk_size]
            self.v[p, 0] = self.v[p, self.cuda_chunk_size]
            self.C[p, 0] = self.C[p, self.cuda_chunk_size]
            self.F[p, 0] = self.F[p, self.cuda_chunk_size]

    @ti.kernel
    def set_state_chunk(
        self,
        x: ti.types.ndarray(),
        v: ti.types.ndarray(),
        C: ti.types.ndarray(),
        F: ti.types.ndarray(),
        Jp: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            self.Jp[p] = Jp[p]
            for i in ti.static(range(self.dim)):
                self.x[p, 0][i] = x[p, i]
                self.v[p, 0][i] = v[p, i]

                for j in ti.static(range(self.dim)):
                    self.C[p, 0][i, j] = C[p, i, j]
                    self.F[p, 0][i, j] = F[p, i, j]

    @ti.kernel
    def cache_gradient(
        self,
        x_grad: ti.types.ndarray(),
        v_grad: ti.types.ndarray(),
        C_grad: ti.types.ndarray(),
        F_grad: ti.types.ndarray(),
        Jp_grad: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            Jp_grad[p] = self.Jp.grad[p]
            for i in ti.static(range(self.dim)):
                x_grad[p, i] = self.x.grad[p, 0][i]
                v_grad[p, i] = self.v.grad[p, 0][i]

                for j in ti.static(range(self.dim)):
                    C_grad[p, i, j] = self.C.grad[p, 0][i, j]
                    F_grad[p, i, j] = self.F.grad[p, 0][i, j]

    @ti.kernel
    def prepare_gradient(
        self,
        x_grad: ti.types.ndarray(),
        v_grad: ti.types.ndarray(),
        C_grad: ti.types.ndarray(),
        F_grad: ti.types.ndarray(),
        Jp_grad: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            self.Jp.grad[p] = Jp_grad[p]
            for i in ti.static(range(self.dim)):
                self.x.grad[p, self.cuda_chunk_size][i] = x_grad[p, i]
                self.v.grad[p, self.cuda_chunk_size][i] = v_grad[p, i]

                for j in ti.static(range(self.dim)):
                    self.C.grad[p, self.cuda_chunk_size][i, j] = C_grad[p, i, j]
                    self.F.grad[p, self.cuda_chunk_size][i, j] = F_grad[p, i, j]

    def push_to_memory(self):
        if self.dtype == ti.f32:
            dtype = self._TORCH_FLOAT32
        else:
            dtype = self._TORCH_FLOAT64

        x = torch.empty([self.n_particles[None], self.dim], dtype=dtype)
        v = torch.empty([self.n_particles[None], self.dim], dtype=dtype)
        C = torch.empty([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        F = torch.empty([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        Jp = torch.empty([self.n_particles[None]], dtype=dtype)

        self.get_state_chunk(x, v, C, F, Jp)
        state = dict(x=x, v=v, C=C, F=F, Jp=Jp)
        self.cached_states.append(state)

    def pop_from_memory(self):
        if self.dtype == ti.f32:
            dtype = self._TORCH_FLOAT32
        else:
            dtype = self._TORCH_FLOAT64

        x_grad = torch.empty([self.n_particles[None], self.dim], dtype=dtype)
        v_grad = torch.empty([self.n_particles[None], self.dim], dtype=dtype)
        C_grad = torch.empty([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        F_grad = torch.empty([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        Jp_grad = torch.empty([self.n_particles[None]], dtype=dtype)

        self.cache_gradient(x_grad, v_grad, C_grad, F_grad, Jp_grad)

        self.x.grad.fill(0)
        self.v.grad.fill(0)
        self.C.grad.fill(0)
        self.F.grad.fill(0)
        self.Jp.grad.fill(0)

        state = self.cached_states.pop()
        self.set_state_chunk(
            state["x"],
            state["v"],
            state["C"],
            state["F"],
            state["Jp"],
        )
        for i in range(self.cuda_chunk_size):
            self.substep(i, False)
        self.prepare_gradient(x_grad, v_grad, C_grad, F_grad, Jp_grad)

    def set_colliders(self, phys_args):
        """Register the ground plane from bc; re-registering replaces it."""
        if "ground" not in phys_args.bc:
            raise ValueError(
                f"bc must define a 'ground' plane, got keys {list(phys_args.bc)}."
            )
        point, normal, bc_style = phys_args.bc["ground"][:3]
        friction = (
            phys_args.bc["ground"][3]
            if len(phys_args.bc["ground"]) > 3
            else self.default_floor_friction
        )
        self.set_surface_collider(point, normal, bc_style, friction)

    def set_surface_collider(
        self, point, normal, surface=surface_sticky, friction=default_floor_friction
    ):
        """Set the ground plane using field-based storage (differentiable)."""
        point = list(point)
        friction = float(friction)

        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal_normalized = [x * normal_scale for x in normal]

        self.collider_point[None] = point
        self.collider_normal[None] = normal_normalized

        @ti.func
        def get_velocity(I, v):
            offset = (I.cast(self.dtype) - self.voxel_offset[None]) * self.dx[
                None
            ] - ti.Vector(point)
            n = ti.Vector(normal_normalized)

            # 1e-6, not 0: voxel_offset lands a node on the plane only in exact
            # arithmetic, and rounding it to +1e-9 would disable the floor.
            if offset.dot(n) <= 1e-6:
                v_normal = n.dot(v)

                if ti.static(surface == self.surface_sticky):
                    v = ti.Vector.zero(self.dtype, self.dim)
                elif ti.static(surface == self.surface_slip):
                    v = v - n * v_normal
                elif ti.static(surface == self.surface_separate):
                    v = v - n * ti.min(v_normal, 0)
                else:
                    # Coulomb: ||v_t'|| = max(0, ||v_t|| - friction * |v_normal|),
                    # normal component removed, no restitution.
                    if v_normal < 0:
                        v_t = v - n * v_normal
                        v = v_t * ti.max(
                            0, 1 - friction * (-v_normal) / v_t.norm(1e-12)
                        )
            return v

        self.ground_response = get_velocity
