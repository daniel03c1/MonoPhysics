import taichi as ti
import torch
from simulator import MPMSimulator
from simulator.base_simulator import BaseSimulator
from lib.volume_utils import compute_induced_volumes


@ti.data_oriented
class Simulator(BaseSimulator):
    """
    Lightweight simulator for forward-pass inference only.

    Used for:
    - Generating trajectories with known parameters
    - Inference and prediction
    - Validation

    Does not support gradient computation or optimization.
    """

    def __init__(self, phys_args, positions, particle_volumes=None, device="cuda"):
        """
        Forward-only simulator over `positions` [N, 3]. particle_volumes defaults
        to compute_induced_volumes at phys_args.voxel_size.
        """
        self.device = device
        self.dtype = ti.f32  # No gradients needed for inference

        # Time stepping
        frame_dt = 1.0 / phys_args.fps
        dt = frame_dt / phys_args.mpm_iter_cnt

        max_n_particles = phys_args.n_max_particles
        if max_n_particles < 1:
            max_n_particles = len(positions)

        # Initialize MPM simulator
        self.sim = MPMSimulator(
            dtype=self.dtype,
            dt=dt,
            frame_dt=frame_dt,
            dx=phys_args.voxel_size,
            max_n_particles=max_n_particles,
            args=phys_args,
            gravity=phys_args.gravity,
            material=phys_args.mat_params["material"],
            cuda_chunk_size=100,
        )

        # Particle state
        self.positions = positions  # Particle positions [N, 3]
        if particle_volumes is None:
            particle_volumes = compute_induced_volumes(
                positions, dx=phys_args.voxel_size
            )
        self.particle_volumes = particle_volumes.clamp(
            max=self.sim.max_vol[None]
        )  # Per-particle volumes [N]

        # Physical parameters
        self.phys_args = phys_args
        self.mat = getattr(phys_args, "mat_params", {})

        # State parameters (velocity, gravity, density) - required fields
        self.vel = torch.tensor(phys_args.init_vel, device=self.device)
        self.gravity = torch.tensor(phys_args.gravity, device=self.device)
        self.sim.gravity[None] = self.gravity
        self.rho = torch.tensor([phys_args.rho], device=self.device)

    def forward(self, f, *args, **kwargs):
        """Particle positions [N, 3] at frame f (0-indexed); forward pass only."""
        xyz = torch.zeros(
            [self.sim.n_particles[None], 3],
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        if f > 0:
            self.sim.advance(f)

        if not self.succeed():
            return xyz

        self.sim.get_x(f, xyz)
        return xyz

    def reload(self, phys_args=None):
        """Reload particle and material params; phys_args replaces the stored one."""
        if phys_args:
            self.phys_args = phys_args
            self.mat = getattr(phys_args, "mat_params", {})

        self._load_particles_params()
        self._load_material_params()
        self.sim.set_colliders(self.phys_args)

    def initialize(self, phys_args=None):
        """Upload particle state to Taichi; phys_args replaces the stored one."""
        if phys_args is None:
            phys_args = self.phys_args

        self.reload(phys_args)
        n_particles = self.positions.shape[0]

        # Compute velocities
        velocities = self.vel.repeat(n_particles).reshape(n_particles, -1)

        # Prepare material parameters
        mu, lam, yield_stress, plastic_viscosity, friction_alpha = (
            self._prepare_material_params()
        )

        # Density
        rho = self.rho.repeat(n_particles)

        # Transfer to Taichi
        self.sim.n_particles[None] = n_particles
        self.sim.set_particles(
            self.positions,
            velocities,
            rho,
            self.particle_volumes,
            mu,
            lam,
            yield_stress,
            plastic_viscosity,
            friction_alpha,
            dx=self.phys_args.voxel_size,
        )
        self.sim.gravity[None] = self.gravity

        # Compute masses
        self.sim.compute_particle_mass()

        # Reset CFL flag
        self.sim.cfl_satisfy[None] = True
        self.sim.v_max_observed[None] = 0.0
