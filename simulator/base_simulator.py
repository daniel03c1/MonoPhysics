import taichi as ti
import torch

from simulator.material_utils import (
    activate_friction_alpha,
    compute_lame_from_E_nu,
    compute_lame_from_mu_kappa,
)


@ti.data_oriented
class BaseSimulator:
    """
    Mixin with methods shared by Simulator (inference) and Estimator
    (training/optimization). Subclasses must define self.sim (MPMSimulator),
    self.scene, and self.device.
    """

    material_attr_names = [
        "E",
        "nu",
        "yield_stress",
        "plastic_viscosity",
        "mu",
        "kappa",
        "friction_alpha",
    ]

    """    Core API    """

    # Must be implemented or available in all subclasses.
    def initialize(self, *args, **kwargs):
        """Initialize simulation state; must be called before forward()."""
        raise NotImplementedError("Subclasses must implement initialize()")

    def forward(self, f):
        """Return particle positions [N, 3] at frame f (0-indexed)."""
        raise NotImplementedError("Subclasses must implement forward()")

    def succeed(self):
        """True if the simulation succeeded (CFL condition satisfied)."""
        return self.sim.cfl_satisfy[None]

    def check_collision(self):
        """True if any particle has touched a collider since the last reset()."""
        return self.sim.check_collision()

    def get_n_particles(self):
        if hasattr(self, "scene"):
            return len(self.scene.gaussians._xyz)
        elif hasattr(self, "positions"):
            return len(self.positions)
        raise ValueError("No scene nor positions")

    """    Shared Utilities    """

    # Parameter loading and processing.
    def _load_particles_params(self):
        """Load particle state parameters (velocity, density) from phys_args."""
        self.vel = torch.tensor(self.phys_args.init_vel, device=self.device)
        self.rho = torch.tensor([self.phys_args.rho], device=self.device)

    def _load_material_params(self):
        """Load material parameters from mat dictionary."""
        for attr_name in self.material_attr_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        report_msg = ""
        for attr_name, value in self.mat.items():
            report_msg += f"{attr_name}: {value} "
            if attr_name == "material":
                self.material = value
                self.sim.material = value
            else:
                setattr(self, attr_name, torch.tensor([value], device=self.device))
        print("Material info: " + report_msg)

    def _prepare_material_params(self):
        """
        Expand material parameters to per-particle arrays; returns
        (mu, lam, yield_stress, plastic_viscosity, friction_alpha).
        """
        n_particles = self.get_n_particles()

        if (
            getattr(self, "E", None) is not None
            and getattr(self, "nu", None) is not None
        ):
            mu, lam = compute_lame_from_E_nu(self.E, self.nu)
        elif (
            getattr(self, "kappa", None) is not None
            and getattr(self, "mu", None) is not None
        ):
            mu, lam = compute_lame_from_mu_kappa(self.mu, self.kappa)
        else:
            raise ValueError(
                "Material parameters undefined! Need (E, nu) or (kappa, mu)"
            )

        mu = mu.repeat(n_particles)
        lam = lam.repeat(n_particles)

        # Friction angle (Drucker-Prager)
        if getattr(self, "friction_alpha", None) is not None:
            friction_alpha_val = activate_friction_alpha(self.friction_alpha)
            friction_alpha = friction_alpha_val.repeat(n_particles)
        else:
            friction_alpha = torch.zeros(n_particles, device=self.device)

        # Yield stress (plasticity)
        if getattr(self, "yield_stress", None) is not None:
            yield_stress = self.yield_stress.repeat(n_particles)
        else:
            yield_stress = torch.zeros(n_particles, device=self.device)

        # Plastic viscosity (non-Newtonian)
        if getattr(self, "plastic_viscosity", None) is not None:
            plastic_viscosity = self.plastic_viscosity.repeat(n_particles)
        else:
            plastic_viscosity = torch.zeros(n_particles, device=self.device)

        return mu, lam, yield_stress, plastic_viscosity, friction_alpha
