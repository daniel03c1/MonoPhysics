"""Material parameter helpers: pure, stateless transformation functions."""

import numpy as np
import torch


def compute_lame_from_E_nu(E, nu):
    """Lame parameters (mu, lam) from Young's modulus E and Poisson ratio nu."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def compute_lame_from_mu_kappa(mu, kappa):
    """Lame parameters (mu, lam) from shear modulus mu and bulk modulus kappa."""
    lam = kappa - 2.0 / 3.0 * mu
    return mu, lam


def compute_K_G_from_E_nu(E, nu):
    """Bulk and shear modulus (K, G) from Young's modulus and Poisson ratio."""
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return K, G


def constraint(x, bound):
    """Map unconstrained parameter to bounded range via tanh."""
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return y_scale * torch.tanh(x_scale * x) + (bound[0] + y_scale)


def constraint_inv(y, bound):
    """Inverse of constraint — map physical value to unconstrained space."""
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale


def compute_E_nu_from_K_G(K, G):
    """Young's modulus and Poisson ratio (E, nu) from bulk and shear modulus."""
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    return E, nu


def activate_friction_alpha(friction_angle_degrees):
    """Convert friction angle (degrees) to the Drucker-Prager alpha parameter."""
    sin_phi = torch.sin(friction_angle_degrees / 180 * np.pi)
    return np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
