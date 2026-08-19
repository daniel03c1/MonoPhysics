import numpy as np
import os
import taichi as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from simulator import MPMSimulator
from simulator.base_simulator import BaseSimulator
from simulator.loss_controller import LossController
from simulator.material_utils import (
    activate_friction_alpha,
    compute_E_nu_from_K_G,
    compute_K_G_from_E_nu,
    compute_lame_from_E_nu,
    compute_lame_from_mu_kappa,
    constraint,
    constraint_inv,
)
from lib.center_metric import per_frame_center_inside
from lib.volume_utils import compute_induced_volumes


CACHE_KEYS = {
    "F_flow",
    "F_uncertainty",
    "f_flow",
    "f_uncertainty",
    "b_flow",
    "b_uncertainty",
    "B_flow",
    "B_uncertainty",
}


def build_ground_plane(bc, device):
    """
    The "ground" collider from a bc dict as a (point, unit normal) GPU tensor pair,
    read the same way MPMSimulator.set_colliders reads it. None when bc is absent.
    """
    if bc is None:
        return None

    if "ground" not in bc:
        raise ValueError(f"bc must define a 'ground' plane, got keys {list(bc)}.")

    point, normal = bc["ground"][0], bc["ground"][1]
    point_t = torch.tensor(point, dtype=torch.float32, device=device)
    normal_t = F.normalize(
        torch.tensor(normal, dtype=torch.float32, device=device), dim=0
    )
    return point_t, normal_t


@ti.data_oriented
class Estimator(BaseSimulator, torch.nn.Module):
    """
    Joint parameter and state estimation via gradient-based optimization
    (material parameters, initial velocity/gravity, multi-modal losses).
    """

    def __init__(
        self,
        phys_args,
        dtype,
        positions,
        cuda_chunk_size=800,
        dynamic_scene=None,
        image_scale=1.0,
        pipeline=None,
        image_op=None,
        background=None,
    ):
        super().__init__()
        self.scene = dynamic_scene

        self.pipeline = pipeline
        self.image_op = image_op

        self.image_scale = image_scale
        self.positions = positions

        self.device = positions.device
        self.dtype = ti.f64 if dtype == "float64" else ti.f32
        self.frame_dt = 1.0 / phys_args.fps

        self.pos_grad_seq = []
        # Debug-only per-loss breakdown of pos_grad_seq, keyed by loss name
        # ("Img"/"Flow"/"Sil"): per-frame detached grad snapshots for comparing
        # relative magnitudes; NOT used for optimization.
        self.pos_grad_debug = {}

        self.bc = getattr(phys_args, "bc", None)
        self._cache_boundary_tensors()

        self.loss_controller = LossController(phys_args, self)

        self.phys_args = phys_args
        self.mat = getattr(phys_args, "mat_params", {})

        self._setup_views()

        if background is None:
            self.background = torch.tensor(
                [0, 0, 0], dtype=torch.float32, device="cuda"
            )
        else:
            assert isinstance(background, torch.Tensor)
            self.background = background

        self.adam_betas = (
            image_op.adam_beta_1,
            image_op.adam_beta_2,
        )
        self._setup_material_params(phys_args)
        self._setup_state_params(phys_args)

        max_n_particles = phys_args.n_max_particles
        if max_n_particles < 1:
            max_n_particles = len(self.scene.gaussians.get_xyz())

        self.sim = MPMSimulator(
            dtype=self.dtype,
            dt=self.frame_dt / phys_args.mpm_iter_cnt,
            frame_dt=self.frame_dt,
            max_n_particles=max_n_particles,
            material=phys_args.material,
            dx=phys_args.voxel_size,
            args=phys_args,
            gravity=phys_args.gravity,
            cuda_chunk_size=cuda_chunk_size,
        )

        self._x_cache = None
        self._x_cache_scene_detached = None
        self._v_cache = None
        self._rho_cache = None
        self._vol_cache = None
        self._gravity_cache = None

        self.init_yield_stress = None
        self.init_plastic_viscosity = None
        self.init_friction_alpha = None

        self.training = True

        # for optical flow computation
        self.prev_xyz = None

        # per-frame position cache for appearance refinement
        self.cached_positions = []
        # per-frame appearance-invariant geometry gate (set by compute_center_inside)
        self.per_frame_center_inside = []

    def _cache_boundary_tensors(self):
        """Cache the ground plane as GPU tensors; fixed during optimization."""
        self.ground_plane = build_ground_plane(self.bc, self.device)

    """    Core APIs    """

    def eval(self):
        self.training = False

    def initialize(self):
        self.sim.reset()

        ti.sync()
        torch.cuda.synchronize()

        self.pos_grad_seq.clear()
        self.pos_grad_debug.clear()

        # Zero Gaussian parameter gradients: per-frame loss.backward() accumulates
        # into these and neither sim.reset() nor pos_grad_seq.clear() clears them,
        # so a failed pass (dt halved + retry) would otherwise double-count.
        gs = self.scene.gaussians
        for opt in [
            gs.optimizer,
            gs.x_optimizer,
            gs.scale_optimizer,
        ]:
            if opt is not None:
                opt.zero_grad()

        # freeze_geometry: _xyz is in no optimizer, so clear its stray grad here.
        gs._xyz.grad = None

        self.loss_controller.reset()
        self.prev_xyz = None
        self._first_collision_frame = None
        self.cached_positions = []

        # Fetch fresh positions to get the current computation graph (avoids a
        # stale graph when _scale or _xyz are modified by optimizers).
        current_xyz = self.scene.gaussians.get_xyz()
        n_particles = current_xyz.shape[0]

        if self._use_E_nu:
            E = self.get_E()
            nu = self.get_nu()
            mu, lam = compute_lame_from_E_nu(E, nu)
        else:
            mu = self.get_mu()
            kappa = self.get_kappa()
            mu, lam = compute_lame_from_mu_kappa(mu, kappa)
        self.init_mu = mu.repeat(n_particles).contiguous()
        self.init_lam = lam.repeat(n_particles).contiguous()

        yield_stress = self.get_yield_stress()
        eta = self.get_plastic_viscosity()
        friction_alpha_activated = activate_friction_alpha(self.get_friction_angle())

        self._x_cache = current_xyz * 1.0
        self._x_cache = self._x_cache.requires_grad_(True).contiguous()
        self._x_cache_scene_detached = self.scene.gaussians.get_xyz(
            detach_scene_scale=True
        )

        if torch.isnan(self._x_cache).any():
            n_nan = torch.isnan(self._x_cache).any(dim=1).sum()
            print(f"[NaN DEBUG] _x_cache has {n_nan} NaN particles")
            print(f"  scene_scale = {self.scene.gaussians._scene_scale.item()}")
            print(
                f"  _xyz NaN count = {torch.isnan(self.scene.gaussians._xyz).any(dim=1).sum()}"
            )

        self._v_cache = (
            self.init_vel.repeat(n_particles).reshape(n_particles, -1).contiguous()
        ).requires_grad_(True)

        self._rho_cache = self.global_rho.repeat(n_particles).detach().contiguous()

        vol_raw = compute_induced_volumes(
            self._x_cache_scene_detached,
            dx=self.phys_args.voxel_size,
        )
        self._vol_cache = torch.clamp(vol_raw, max=self.sim.max_vol[None]).contiguous()

        if torch.isnan(self._vol_cache).any():
            n_nan = torch.isnan(self._vol_cache).sum()
            print(
                f"[NaN DEBUG] _vol_cache has {n_nan} NaN values out of {self._vol_cache.numel()}"
            )
            print(f"  vol_raw NaN: {torch.isnan(vol_raw).sum()}")
            print(f"  max_vol = {self.sim.max_vol[None]}")
            raise RuntimeError("NaN in volumes")

        # Optionally detach so volume gradients don't flow back to positions
        if self.phys_args.detach_induced_volumes:
            self._vol_cache = self._vol_cache.detach()

        self.init_yield_stress = (
            yield_stress.repeat(n_particles)
            if yield_stress.numel() == 1
            else yield_stress
        ).contiguous()
        self.init_plastic_viscosity = (
            eta.repeat(n_particles) if eta.numel() == 1 else eta
        ).contiguous()
        self.init_friction_alpha = (
            friction_alpha_activated.repeat(n_particles)
            if friction_alpha_activated.numel() == 1
            else friction_alpha_activated
        ).contiguous()

        self.sim.n_particles[None] = n_particles
        self.sim.set_particles(
            self._x_cache,
            self._v_cache,
            self._rho_cache,
            self._vol_cache,
            self.init_mu,
            self.init_lam,
            self.init_yield_stress,
            self.init_plastic_viscosity,
            self.init_friction_alpha,
            dx=self.phys_args.voxel_size,
        )

        ti.sync()
        torch.cuda.synchronize()

        self._gravity_cache = self.gravity
        self.sim.gravity[None] = self._gravity_cache.detach().cpu().tolist()

        self.sim.compute_particle_mass()

    def forward(self, f, backward=True, skip_loss=False):
        xyz = torch.zeros(
            [self.get_n_particles(), 3],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        if f > 0:
            self.sim.advance(f)
            if self._first_collision_frame is None and self.sim.check_collision():
                self._first_collision_frame = f

        if not self.succeed():
            return xyz

        self.sim.get_x(f, xyz)
        assert xyz.requires_grad

        if not skip_loss:
            self.loss_controller.compute_all(self, f, xyz, backward)

        # Cache current position for next frame's optical flow computation
        self.prev_xyz = xyz.detach().clone()
        self.prev_xyz.requires_grad_(True)

        self.cache_frame_position(f, xyz)

        return xyz

    def cache_frame_position(self, f, x):
        assert len(self.cached_positions) == f, (
            f"cache_frame_position: expected frame {f}, "
            f"but cached_positions has {len(self.cached_positions)} entries"
        )
        self.cached_positions.append(x.detach().clone())

    def compute_losses_on_cached(self, max_f):
        """
        Compute per-frame losses on the cached rollout positions, refilling
        pos_grad_seq for the adjoint backward. Must run after a skip_loss rollout
        (initialize() leaves pos_grad_seq/pos_grad_debug cleared and the loss
        controller reset). prev_xyz is rebuilt per frame from the cache so
        FlowLoss's cross-frame gradients land in the same slots as the fused path.
        """
        self.prev_xyz = None
        for f in range(max_f):
            if f > 0:
                self.prev_xyz = (
                    self.cached_positions[f - 1].detach().clone().requires_grad_(True)
                )
            xyz_f = self.cached_positions[f].detach().clone().requires_grad_(True)
            self.loss_controller.compute_all(self, f, xyz_f, backward=True)

    def compute_center_inside(self, max_f):
        """
        Per-frame fraction of Gaussian centers projecting inside the GT mask
        (appearance-invariant geometry gate). Stored in per_frame_center_inside.
        """
        views = [self.views[f] for f in range(max_f)]
        self.per_frame_center_inside = per_frame_center_inside(
            self.cached_positions[:max_f], views
        )

    def get_loss_summary(self):
        """Get loss summary from controller."""
        return self.loss_controller.get_summary()

    def backward(self, f):
        if len(self.pos_grad_seq) > f:
            grad = self.pos_grad_seq[f]
            nan_count = torch.isnan(grad).sum().item()
            inf_count = torch.isinf(grad).sum().item()
            if nan_count > 0 or inf_count > 0:
                print(
                    f"[NaN] pos_grad_seq[{f}]: {nan_count} NaN, {inf_count} Inf"
                    " — NaN originates from rendering side"
                )

            pos_grad_clamp = self.phys_args.pos_grad_clamp
            if pos_grad_clamp is not None and pos_grad_clamp > 0:
                norms = grad.norm(dim=-1, keepdim=True)  # [N, 1]
                scale = (pos_grad_clamp / (norms + 1e-8)).clamp(max=1.0)  # [N, 1]
                grad.mul_(scale)

            self.set_pos_grad(f, grad)

        if f > 0:
            self.sim.advance_grad(f)
            return True
        else:
            self.sim.compute_particle_mass.grad()

            n_particles = self.get_n_particles()

            kwargs = {"dtype": torch.float32, "device": self.device}

            # Slot-0 read via kernel; to_torch() would copy all cuda_chunk_size+1
            # slots of the chunked grad field just to slice out slot 0.
            x_grad = torch.empty((n_particles, 3), **kwargs)  # [N, 3]
            v_grad = torch.empty((n_particles, 3), **kwargs)  # [N, 3]
            self.sim.read_xv_grad_slot0(x_grad, v_grad)

            vol_grad = self.sim.vol.grad.to_torch(device=self.device)[:n_particles]

            mu_grad = self.sim.mu.grad.to_torch(device=self.device)[:n_particles]
            lam_grad = self.sim.lam.grad.to_torch(device=self.device)[:n_particles]

            yield_stress_grad = self.sim.yield_stress.grad.to_torch(device=self.device)[
                :n_particles
            ]
            viscosity_grad = self.sim.plastic_viscosity.grad.to_torch(
                device=self.device
            )[:n_particles]
            friction_alpha_grad = self.sim.friction_alpha.grad.to_torch(
                device=self.device
            )[:n_particles]

            # Read-only, verbose-only diagnostics: how per-particle gradients
            # aggregate into the shared scalar material params.
            if self.phys_args.verbose:
                # Pre-aggregation per-particle grad-norm distribution (tail heaviness).
                for nm, g in (
                    ("x", x_grad),
                    ("v", v_grad),
                    ("vol", vol_grad),
                    ("mu", mu_grad),
                    ("lam", lam_grad),
                    ("yield_stress", yield_stress_grad),
                    ("viscosity", viscosity_grad),
                    ("friction_alpha", friction_alpha_grad),
                ):
                    self._log_grad_percentiles(nm, g)

                self._log_grad_reduction("yield_stress", yield_stress_grad)
                if (
                    self._use_E_nu
                    and self.nu.requires_grad
                    and self.init_mu.requires_grad
                ):
                    # nu feeds BOTH Lame params: particle i's net vote is
                    # mu_grad_i * dmu/dnu + lam_grad_i * dlam/dnu (Jacobians shared
                    # across particles — init_mu/init_lam repeat one scalar; graph
                    # still alive via retain_graph).
                    j_mu = torch.autograd.grad(
                        self.init_mu[0], self.nu, retain_graph=True
                    )[0]
                    j_lam = torch.autograd.grad(
                        self.init_lam[0], self.nu, retain_graph=True
                    )[0]
                    nu_contrib = mu_grad * j_mu.detach() + lam_grad * j_lam.detach()
                    self._log_grad_reduction("nu", nu_contrib)

            grad_info = {
                "x_grad": x_grad,
                "v_grad": v_grad,
                "vol_grad": vol_grad,
                "mu_grad": mu_grad,
                "lam_grad": lam_grad,
                "yield_stress_grad": yield_stress_grad,
                "viscosity_grad": viscosity_grad,
                "friction_alpha_grad": friction_alpha_grad,
            }

            # Single combined GPU reduction; one host sync on the happy path
            # instead of 18 (.item() per grad × 2 for nan/inf).
            any_bad_flag = torch.zeros((), dtype=torch.bool, device=self.device)
            for grad in grad_info.values():
                any_bad_flag = (
                    any_bad_flag | torch.isnan(grad).any() | torch.isinf(grad).any()
                )

            has_nan = False
            has_inf = False
            error_msg = "\n" + "=" * 80 + "\n"
            error_msg += "GRADIENT NaN/Inf DETECTED IN BACKWARD PASS (frame 0)\n"
            error_msg += "=" * 80 + "\n"

            if any_bad_flag.item():
                for name, grad in grad_info.items():
                    nan_count = torch.isnan(grad).sum().item()
                    inf_count = torch.isinf(grad).sum().item()

                    if nan_count > 0 or inf_count > 0:
                        has_nan = has_nan or (nan_count > 0)
                        has_inf = has_inf or (inf_count > 0)

                        grad_finite = grad[torch.isfinite(grad)]
                        error_msg += f"\n{name}:\n"
                        error_msg += f"  Shape: {grad.shape}\n"
                        error_msg += f"  NaN count: {nan_count} / {grad.numel()}\n"
                        error_msg += f"  Inf count: {inf_count} / {grad.numel()}\n"

                        if grad_finite.numel() > 0:
                            error_msg += f"  Finite values - min: {grad_finite.min().item():.6e}, "
                            error_msg += f"max: {grad_finite.max().item():.6e}, "
                            error_msg += f"mean: {grad_finite.mean().item():.6e}\n"

            if has_nan or has_inf:
                error_msg += "\n" + "=" * 80 + "\n"
                error_msg += (
                    "This indicates numerical instability in the backward pass.\n"
                )
                error_msg += "Common causes:\n"
                error_msg += "  - Deformation gradient F has extreme values (check CFL condition)\n"
                error_msg += "  - Material parameters (mu, lam) may be too extreme\n"
                error_msg += "  - Loss gradients may be too large\n"
                error_msg += "=" * 80
                print(error_msg)
                return False

            self._x_cache.backward(
                retain_graph=True,
                gradient=x_grad - self.pos_grad_seq[0],
            )
            self._x_cache_scene_detached.backward(
                retain_graph=True,
                gradient=self.pos_grad_seq[0],
            )
            self._v_cache.backward(
                retain_graph=True,
                gradient=v_grad,
            )

            if self._vol_cache.grad_fn is not None:
                self._vol_cache.backward(retain_graph=True, gradient=vol_grad)

            self.init_mu.backward(retain_graph=True, gradient=mu_grad)
            self.init_lam.backward(retain_graph=True, gradient=lam_grad)

            self.init_yield_stress.backward(
                retain_graph=True,
                gradient=yield_stress_grad,
            )
            self.init_plastic_viscosity.backward(
                retain_graph=True,
                gradient=viscosity_grad,
            )
            self.init_friction_alpha.backward(
                retain_graph=True,
                gradient=friction_alpha_grad,
            )

            gravity_grad = torch.empty([3], **kwargs)
            for i in range(3):
                gravity_grad[i] = self.sim.gravity.grad[None][i]
            self._gravity_cache.backward(
                retain_graph=True,
                gradient=gravity_grad,
            )
            return True

    @staticmethod
    @torch.no_grad()
    def _log_grad_percentiles(name, g):
        """
        Log p50/p95/p99/p100 of the per-particle gradient NORM for one feature
        (pre-aggregation). Per-particle norm = vector norm for [N, *], |g| for
        [N]. Read-only; verbose-only.
        """
        norms = g.reshape(g.shape[0], -1).norm(dim=1)  # [N]
        qs = torch.quantile(
            norms, torch.tensor([0.5, 0.95, 0.99, 1.0], device=norms.device)
        )
        p50, p95, p99, p100 = (v.item() for v in qs)
        print(
            f"[grad-dist] {name}: p50={p50:.3e} p95={p95:.3e} "
            f"p99={p99:.3e} p100={p100:.3e}"
        )

    @staticmethod
    @torch.no_grad()
    def _log_grad_reduction(name, contrib):
        """
        Read-only, verbose-only: how a per-particle gradient [N] sums into a
        shared scalar param. Reports sum (post-reduction gradient), top1%_mass
        (fraction of total |contribution| in the top 1% of particles; ~0.01 =
        bulk-dominated so clamping is moot), and clip@p99 (signed sum after
        clamping each |contribution| at its p99, ratio vs. raw, sign flip).
        """
        c = contrib.detach().reshape(-1)
        absc = c.abs()
        total = absc.sum()
        raw = c.sum().item()

        k = max(1, c.numel() // 100)
        top_mass = (torch.topk(absc, k).values.sum() / (total + 1e-30)).item()

        thr = torch.quantile(absc, 0.99)
        clipped = c.clamp(min=-thr, max=thr).sum().item()

        if abs(raw) < 1e-30:
            ratio_s, flip_s = "nan", "n/a"
        else:
            ratio_s = f"{clipped / raw:.3f}"
            flip_s = str(raw * clipped < 0)

        print(
            f"[grad-reduce] {name}: sum={raw:.3e} top1%_mass={top_mass:.3f} "
            f"clip@p99: sum={clipped:.3e} ratio={ratio_s} sign_flip={flip_s}"
        )

    def check_collision_with_delay(self, actual_frames, delay=1):
        """
        True if collision occurred and at least `delay` frames ran after it.
        `actual_frames` is a 1-based count; `_first_collision_frame` is 0-based
        (delay=1 requires actual_frames >= _first_collision_frame + 2).
        """
        if self._first_collision_frame is None:
            return False
        return actual_frames >= self._first_collision_frame + 1 + delay

    def accumulate_pos_grad(self, name, frame, xyz):
        """
        Inject a loss's xyz.grad into pos_grad_seq[frame] (append or in-place add),
        mirroring it into the per-loss debug breakdown.
        """
        if xyz.grad is None:
            xyz.grad = torch.zeros_like(xyz)
        grad = xyz.grad.clone()
        if len(self.pos_grad_seq) > frame:
            self.pos_grad_seq[frame].add_(grad)
        else:
            self.pos_grad_seq.append(grad)
        self.add_pos_grad_debug(name, frame, grad)

    def add_pos_grad_debug(self, name, frame, grad):
        """
        Mirror one loss's per-frame position-gradient contribution into a
        debug-only list (does NOT affect pos_grad_seq). Stores a detached clone so
        later in-place .add_() on shared grad storage cannot corrupt the snapshot;
        frame-indexed to stay aligned with pos_grad_seq (Flow also hits earlier frames).
        """
        seq = self.pos_grad_debug.setdefault(name, [])
        while len(seq) <= frame:
            seq.append(None)
        snap = grad.detach().clone()
        seq[frame] = snap if seq[frame] is None else seq[frame] + snap

    def clear_grads(self):
        self.sim.clear_grads()

    # setters and getters
    def get_E(self):
        return 10**self.E

    def get_nu(self):
        return constraint(self.nu, self.nu_bound)

    def get_mu(self):
        return 10**self.global_mu

    def get_kappa(self):
        return 10**self.global_kappa

    def get_yield_stress(self):
        return 10**self.yield_stress

    def get_plastic_viscosity(self):
        return 10**self.plastic_viscosity

    def get_friction_angle(self):
        return self.friction_angle

    def get_derived_E(self):
        """Get Young's modulus — primary if E/nu, derived from K/G otherwise."""
        if self._use_E_nu:
            return self.get_E()
        E, _ = compute_E_nu_from_K_G(self.get_kappa(), self.get_mu())
        return E

    def get_derived_nu(self):
        """Get Poisson's ratio — primary if E/nu, derived from K/G otherwise."""
        if self._use_E_nu:
            return self.get_nu()
        _, nu = compute_E_nu_from_K_G(self.get_kappa(), self.get_mu())
        return nu

    def get_derived_K(self):
        """Get bulk modulus K — primary if K/G, derived from E/nu otherwise."""
        if self._use_E_nu:
            K, _ = compute_K_G_from_E_nu(self.get_E(), self.get_nu())
            return K
        return self.get_kappa()

    def get_derived_G(self):
        """Get shear modulus G — primary if K/G, derived from E/nu otherwise."""
        if self._use_E_nu:
            _, G = compute_K_G_from_E_nu(self.get_E(), self.get_nu())
            return G
        return self.get_mu()

    """    Internal Functions    """

    # These shouldn't be called outside the Estimator
    def _setup_views(self):
        views = self.scene.getTrainCameras(scale=self.image_scale)
        t_ls = torch.unique(torch.stack([view.fid for view in views if view.fid >= 0]))
        t_ls, _ = torch.sort(t_ls.cpu())
        all_views = []
        for t in t_ls:
            views_by_t = [v for v in views if torch.abs(v.fid.cpu() - t) < 1e-7]
            all_views.append(views_by_t)

        self.views = all_views
        self.view_maps = {
            (v.fid.item(), v.uid): v for views_t in all_views for v in views_t
        }

        if self.phys_args.w_flow > 0.0:
            self._cache_optical_flows()

    def _setup_optical_flow(self):
        """Initialize SEA-RAFT optical flow estimator."""
        from lib.flow import FlowEstimator, is_hf_repo_id

        model_path = self.phys_args.flow_model_path
        config_path = self.phys_args.flow_config_path
        if not os.path.exists(model_path) and not is_hf_repo_id(model_path):
            raise FileNotFoundError(
                f"SEA-RAFT weights not found: {model_path}. Download them (see the "
                "'Optical flow' section of the README) or pass --flow_model_path."
            )

        print("[Optical Flow] Initializing SEA-RAFT estimator...")
        print(f"  Config: {config_path}")
        print(f"  Weights: {model_path}")

        self.flow_estimator = FlowEstimator(
            config_path=config_path,
            model_path=model_path,
            device="cuda",
            iters=getattr(self.phys_args, "flow_iters", 32),
        )

        print("[Optical Flow] Initialized successfully")

    def _cache_optical_flows(self):
        """
        Load optical flows from disk cache (model_path/optical_flow_cache/, one
        .pt per (fid, uid)), computing and saving misses; the flow estimator is only
        initialized on a miss and freed afterwards. Builds self.optical_flows:
        (fid, uid) -> {"F": 0→t, "f": (t-1)→t, "b": t→(t-1), "B": t→0}, each (flow, uncertainty).
        """
        cache_dir = os.path.join(self.scene.model_path, "optical_flow_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[Optical Flow] Cache directory: {cache_dir}")
        self.optical_flows = {}
        init_view_map = {v.uid: v for v in self.views[0]}
        flow_estimator_initialized = False

        for f in tqdm(
            range(1, len(self.views)), desc="Loading/computing optical flows"
        ):
            fid = self.views[f][0].fid.item()
            prev_view_map = {v.uid: v for v in self.views[f - 1]}

            for curr_view in self.views[f]:
                uid = curr_view.uid
                cache_path = os.path.join(cache_dir, f"flow_fid{fid:.4f}_uid{uid}.pt")

                cache_loaded = False
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    if CACHE_KEYS.issubset(data.keys()):
                        self.optical_flows[(fid, uid)] = {
                            "F": (
                                data["F_flow"].to(self.device),
                                data["F_uncertainty"].to(self.device),
                            ),
                            "f": (
                                data["f_flow"].to(self.device),
                                data["f_uncertainty"].to(self.device),
                            ),
                            "b": (
                                data["b_flow"].to(self.device),
                                data["b_uncertainty"].to(self.device),
                            ),
                            "B": (
                                data["B_flow"].to(self.device),
                                data["B_uncertainty"].to(self.device),
                            ),
                        }
                        cache_loaded = True
                    # else: old cache format — recompute below

                if not cache_loaded:
                    if not flow_estimator_initialized:
                        self._setup_optical_flow()
                        flow_estimator_initialized = True

                    init_view = init_view_map[uid]
                    prev_view = prev_view_map[uid]

                    F_flow, F_uncertainty = self.flow_estimator.compute_flow(
                        init_view.gt_image,
                        curr_view.gt_image,
                        return_numpy=False,
                        return_uncertainty=True,
                    )
                    f_flow, f_uncertainty = self.flow_estimator.compute_flow(
                        prev_view.gt_image,
                        curr_view.gt_image,
                        return_numpy=False,
                        return_uncertainty=True,
                    )
                    b_flow, b_uncertainty = self.flow_estimator.compute_flow(
                        curr_view.gt_image,
                        prev_view.gt_image,
                        return_numpy=False,
                        return_uncertainty=True,
                    )
                    B_flow, B_uncertainty = self.flow_estimator.compute_flow(
                        curr_view.gt_image,
                        init_view.gt_image,
                        return_numpy=False,
                        return_uncertainty=True,
                    )

                    self.optical_flows[(fid, uid)] = {
                        "F": (F_flow.to(self.device), F_uncertainty.to(self.device)),
                        "f": (f_flow.to(self.device), f_uncertainty.to(self.device)),
                        "b": (b_flow.to(self.device), b_uncertainty.to(self.device)),
                        "B": (B_flow.to(self.device), B_uncertainty.to(self.device)),
                    }

                    torch.save(
                        {
                            "F_flow": F_flow.cpu(),
                            "F_uncertainty": F_uncertainty.cpu(),
                            "f_flow": f_flow.cpu(),
                            "f_uncertainty": f_uncertainty.cpu(),
                            "b_flow": b_flow.cpu(),
                            "b_uncertainty": b_uncertainty.cpu(),
                            "B_flow": B_flow.cpu(),
                            "B_uncertainty": B_uncertainty.cpu(),
                        },
                        cache_path,
                    )

                    # Flow visualizations, saved alongside each cached .pt
                    from lib.flow import flow_to_image
                    from PIL import Image

                    for letter, flow_tensor in [
                        ("F", F_flow),
                        ("f", f_flow),
                        ("b", b_flow),
                        ("B", B_flow),
                    ]:
                        flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
                        Image.fromarray(flow_to_image(flow_np)).save(
                            cache_path.replace(".pt", f"_{letter}.png")
                        )

        if flow_estimator_initialized:
            del self.flow_estimator
            torch.cuda.empty_cache()

        n_cached = len(self.optical_flows)
        print(f"[Optical Flow] Cached {n_cached} flow fields (cache dir: {cache_dir})")

    def get_optical_flow(self, fid, uid):
        """
        Cached flows for (fid, uid): {"F": 0→t, "f": (t-1)→t, "b": t→(t-1),
        "B": t→0}, each (flow, uncertainty); None if uncached.
        """
        if not hasattr(self, "optical_flows"):
            return None
        return self.optical_flows.get((fid, uid), None)

    # Materials that use E/nu parameterization instead of K/G
    _E_NU_MATERIALS = {
        MPMSimulator.elasticity,
        MPMSimulator.neo_hookean,
        MPMSimulator.plasticine_corotated,
    }

    def _setup_material_params(self, phys_args):
        kwargs = {"device": self.device, "dtype": torch.float32}
        self.global_rho = nn.Parameter(torch.tensor(phys_args.rho, **kwargs))

        self.yield_stress = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_yield_stress", 0.0)], **kwargs)
        )
        self.plastic_viscosity = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_plastic_viscosity", -1e6)], **kwargs)
        )
        # Degrees; activate_friction_alpha converts it to Drucker-Prager alpha at
        # each initialize(). Config key stays init_friction_alpha, as in GIC.
        self.friction_angle = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_friction_alpha", 0.0)], **kwargs)
        )

        # Materials outside _E_NU_MATERIALS (viscous fluid, von Mises, Drucker-Prager)
        # are natively parameterized by bulk/shear modulus, so they keep the K/G path
        # and the get_derived_* accessors.
        self._use_E_nu = phys_args.material in self._E_NU_MATERIALS

        if self._use_E_nu:
            # E/nu parameterization (materials 10, 14, 18): E in log10 space,
            # nu via tanh constraint to [0, 0.49].
            self.nu_bound = getattr(phys_args, "nu_bound", [0.0, 0.49])

            init_E = getattr(phys_args, "init_E", None)
            if init_E is not None:
                self.E = nn.Parameter(torch.tensor([init_E], **kwargs))
            else:
                self.E = nn.Parameter(
                    torch.tensor([np.log10(getattr(phys_args, "E", 1e5))], **kwargs)
                )

            init_nu = getattr(phys_args, "init_nu", None)
            if init_nu is not None:
                nu_tensor = torch.tensor([init_nu], **kwargs)
            else:
                nu_tensor = torch.tensor([0.25], **kwargs)
            self.nu = nn.Parameter(constraint_inv(nu_tensor, self.nu_bound))
        else:
            # K/G parameterization for other materials (11, 15, etc.).
            # Shear modulus G (log10 space). Accepts init_G (log10) or mu (raw).
            init_G = getattr(phys_args, "init_G", None)
            if init_G is not None:
                self.global_mu = nn.Parameter(torch.tensor([init_G], **kwargs))
            else:
                self.global_mu = nn.Parameter(
                    torch.tensor([np.log10(getattr(phys_args, "mu", 1.0))], **kwargs)
                )

            # Bulk modulus K (log10 space). Accepts init_K (log10) or kappa (raw).
            init_K = getattr(phys_args, "init_K", None)
            if init_K is not None:
                self.global_kappa = nn.Parameter(torch.tensor([init_K], **kwargs))
            else:
                self.global_kappa = nn.Parameter(
                    torch.tensor([np.log10(getattr(phys_args, "kappa", 1.0))], **kwargs)
                )

        specs = {
            "Yield stress": (self.yield_stress, phys_args.yield_stress_lr),
            "plastic viscosity": (
                self.plastic_viscosity,
                phys_args.plastic_viscosity_lr,
            ),
            "friction angle": (self.friction_angle, phys_args.friction_angle_lr),
        }
        if self._use_E_nu:
            specs["Youngs modulus"] = (self.E, phys_args.youngs_modulus_lr)
            specs["Poisson ratio"] = (self.nu, phys_args.poisson_ratio_lr)
        else:
            specs["bulk modulus"] = (self.global_kappa, phys_args.bulk_modulus_lr)
            specs["shear modulus"] = (self.global_mu, phys_args.shear_modulus_lr)

        params = []
        for param_name, info in phys_args.params.items():
            if param_name in specs:
                param, default_lr = specs[param_name]
                params.append(
                    {
                        "params": param,
                        "lr": info.get("init_lr", default_lr),
                        "name": param_name,
                    }
                )

        self.material_optimizer = torch.optim.Adam(
            [*params], betas=self.adam_betas, amsgrad=False, fused=True
        )

    def _setup_state_params(self, phys_args):
        self.gravity = nn.Parameter(torch.tensor(phys_args.gravity, device=self.device))

        self._init_vel = nn.Parameter(
            torch.tensor(phys_args.init_vel, device=self.device)
        )
        self.state_optimizer = torch.optim.Adam(
            [
                {
                    "params": self._init_vel,
                    "lr": phys_args.vel_lr,
                    "name": "velocity",
                },
                {
                    "params": self.gravity,
                    "lr": getattr(phys_args, "gravity_lr", 0.0),
                    "name": "gravity",
                },
            ],
            betas=self.adam_betas,
            fused=True,
        )

    @property
    def init_vel(self):
        """
        World-space initial velocity, scene_scale-multiplied (ready for the
        simulator). scene_scale defaults to 1.0, so this is a no-op when scale
        optimization is not used.
        """
        return self._init_vel * self.scene.gaussians.get_scene_scale().detach()

    """    Taichi Kernels    """

    @ti.kernel
    def set_pos_grad(self, f: ti.i32, dLdpo: ti.types.ndarray()):
        s = (f * self.sim.n_substeps[None]) % self.sim.cuda_chunk_size
        for p in range(self.sim.n_particles[None]):
            for d in ti.static(range(3)):
                self.sim.x.grad[p, s][d] += dLdpo[p, d]
