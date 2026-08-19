"""
Material constitutive models for MPM. All functions share identical signatures
to enable function-pointer dispatch (material selected at initialization, no
runtime branching); dimension is fixed to 3D.
"""

import taichi as ti

from simulator.utils import make_matrix_from_diag_3d, norm


@ti.func
def heaviside(y):
    """
    Hard Heaviside step: 0 for y < 0, else 1. For a differentiable yield
    transition use 0.5 * (1.0 + ti.tanh(y / (2.0 * eps))) with smoothing width eps.
    """
    return ti.select(y < 0, 0.0, 1.0)


# Material type constants (matching MPMSimulator)
ELASTICITY = 10
VISCOUS_FLUID = 11
PLASTICINE = 12
DRUCKER_PRAGER = 13
NEO_HOOKEAN = 14
NON_NEWTONIAN = 15
SAND = 17
PLASTICINE_COROTATED = 18

# Principal stretch limits.
SIG_MIN = 0.1
SIG_MAX = 10.0


@ti.func
def clamp_sig(v):
    """Clamp all three principal stretches to [SIG_MIN, SIG_MAX]."""
    return ti.Vector(
        [
            ti.min(ti.max(v[0], SIG_MIN), SIG_MAX),
            ti.min(ti.max(v[1], SIG_MIN), SIG_MAX),
            ti.min(ti.max(v[2], SIG_MIN), SIG_MAX),
        ]
    )


# F projection functions — identical signature for function-pointer compatibility:
# (F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp)
# -> (F_new, sig_out, Jp_new); dtype inferred from input tensors.


@ti.func
def project_F_elasticity(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Pure elastic — clamp sigma and reconstruct F so J stays bounded downstream."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )
    sig_out = in_bounds * sig_orig + (1 - in_bounds) * sig_clamped
    F_new = in_bounds * F_tmp + (1 - in_bounds) * (
        U @ make_matrix_from_diag_3d(sig_clamped) @ V.transpose()
    )
    return F_new, sig_out, Jp


@ti.func
def project_F_neo_hookean(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Pure elastic — clamp sigma and reconstruct F so J stays bounded downstream."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )
    sig_out = in_bounds * sig_orig + (1 - in_bounds) * sig_clamped
    F_new = in_bounds * F_tmp + (1 - in_bounds) * (
        U @ make_matrix_from_diag_3d(sig_clamped) @ V.transpose()
    )
    return F_new, sig_out, Jp


@ti.func
def project_F_viscous_fluid(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Viscous fluid: Clamp J to prevent excessive compression."""
    F_new = ti.Matrix(
        [[ti.max(F_tmp[0, 0], 0.05), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    sig_out = ti.Vector([1.0, 1.0, 1.0])
    return F_new, sig_out, Jp


@ti.func
def project_F_plasticine(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Plasticine material (von Mises plasticity, uses E and nu)."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    F_clamped = U @ make_matrix_from_diag_3d(sig_clamped) @ V.transpose()
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )

    eps = ti.log(sig_clamped)
    trace_eps = eps.sum()
    avg_eps = trace_eps / 3.0
    eps_hat = eps - avg_eps

    eps_hat_norm = norm(eps_hat)
    y = eps_hat_norm - 0.5 * yield_stress / mu

    H = eps - y * eps_hat / eps_hat_norm

    condition = heaviside(y)
    sig_plastic = ti.exp(H)
    fallback_sig = in_bounds * sig_orig + (1 - in_bounds) * sig_clamped
    fallback_F = in_bounds * F_tmp + (1 - in_bounds) * F_clamped
    sig_out = sig_plastic * condition + (1 - condition) * fallback_sig
    F_new = (U @ make_matrix_from_diag_3d(sig_plastic) @ V.transpose()) * condition + (
        1 - condition
    ) * fallback_F

    return F_new, sig_out, Jp


@ti.func
def project_F_plasticine_corotated(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Plasticine corotated (von Mises plasticity) — singular values clamped from below to SIG_MIN."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    F_clamped = U @ make_matrix_from_diag_3d(sig_clamped) @ V.transpose()
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )

    eps = ti.log(sig_clamped)
    trace_eps = eps.sum()
    avg_eps = trace_eps / 3.0
    eps_hat = eps - avg_eps

    eps_hat_norm = norm(eps_hat)
    y = eps_hat_norm - 0.5 * yield_stress / mu

    H = eps - y * eps_hat / eps_hat_norm

    condition = heaviside(y)
    sig_plastic = ti.exp(H)
    fallback_sig = in_bounds * sig_orig + (1 - in_bounds) * sig_clamped
    fallback_F = in_bounds * F_tmp + (1 - in_bounds) * F_clamped
    sig_out = sig_plastic * condition + (1 - condition) * fallback_sig
    F_new = (U @ make_matrix_from_diag_3d(sig_plastic) @ V.transpose()) * condition + (
        1 - condition
    ) * fallback_F

    return F_new, sig_out, Jp


@ti.func
def project_F_non_newtonian(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Non-Newtonian fluid (von Mises model, uses kappa and mu)."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    F_clamped = U @ make_matrix_from_diag_3d(sig_clamped) @ V.transpose()
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )

    eps = ti.log(sig_clamped)
    trace_eps = eps.sum()
    avg_eps = trace_eps / 3.0
    eps_hat = eps - avg_eps

    s_trial = 2 * mu * eps_hat
    s_trial_norm = norm(s_trial)
    y = s_trial_norm - ti.sqrt(2.0 / 3) * yield_stress

    mu_hat = mu * (sig_clamped**2).sum() / 3.0
    s_new_norm = s_trial_norm - y / (1 + plastic_viscosity / (2 * mu_hat * dt))
    s_new = (s_new_norm / s_trial_norm) * s_trial
    H = s_new / (2 * mu) + trace_eps / 3.0

    condition = heaviside(y)
    sig_plastic = ti.exp(H)
    fallback_sig = in_bounds * sig_orig + (1 - in_bounds) * sig_clamped
    fallback_F = in_bounds * F_tmp + (1 - in_bounds) * F_clamped
    sig_out = sig_plastic * condition + (1 - condition) * fallback_sig
    F_new = (U @ make_matrix_from_diag_3d(sig_plastic) @ V.transpose()) * condition + (
        1 - condition
    ) * fallback_F
    return F_new, sig_out, Jp


@ti.func
def project_F_drucker_prager(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Drucker-Prager plasticity for granular materials (simplified, no Jp tracking)."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )

    eps = ti.log(sig_clamped)
    trace_eps = eps.sum()

    eps_hat = eps - trace_eps / 3.0
    eps_hat_norm = norm(eps_hat)

    delta_gamma = eps_hat_norm + (1.5 * lam / mu + 1) * friction_alpha * trace_eps

    condition = ti.cast(trace_eps < 0, ti.f32)
    sig_out = ti.exp(
        condition * (eps - ti.max(delta_gamma, 0.0) * eps_hat / eps_hat_norm)
    )

    # Use F_tmp only for elastic compression (inside cone + in bounds).
    # Tension reset and plastic projection always need reconstruction.
    use_F_tmp = condition * ti.cast(delta_gamma <= 0, ti.f32) * in_bounds
    F_new = use_F_tmp * F_tmp + (1 - use_F_tmp) * (
        U @ make_matrix_from_diag_3d(sig_out) @ V.transpose()
    )
    return F_new, sig_out, Jp


@ti.func
def project_F_sand(
    F_tmp, U, sig, V, mu, lam, yield_stress, plastic_viscosity, friction_alpha, dt, Jp
):
    """Drucker-Prager with plastic volume tracking (Klar 2016)."""
    sig_orig = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sig_clamped = clamp_sig(sig_orig)
    in_bounds = (
        ti.cast(sig_orig[0] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[1] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[2] >= SIG_MIN, ti.f32)
        * ti.cast(sig_orig[0] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[1] <= SIG_MAX, ti.f32)
        * ti.cast(sig_orig[2] <= SIG_MAX, ti.f32)
    )

    eps = ti.log(sig_clamped)
    trace_eps = eps.sum()
    eps_hat = eps - trace_eps / 3.0
    eps_hat_norm = norm(eps_hat)

    # Include accumulated plastic dilation history
    effective_trace = trace_eps + Jp

    # Drucker-Prager cone check (meaningful in compression only)
    delta_gamma = eps_hat_norm + (1.5 * lam / mu + 1) * friction_alpha * effective_trace

    in_compression = ti.cast(effective_trace < 0, ti.f32)

    sig_compression = ti.exp(eps - ti.max(delta_gamma, 0.0) * eps_hat / eps_hat_norm)

    sig_out = in_compression * sig_compression + (1.0 - in_compression) * 1.0
    Jp_new = in_compression * Jp + (1.0 - in_compression) * (Jp + trace_eps)

    # Use F_tmp only for elastic compression (inside cone + in bounds).
    # Tension reset and plastic projection always need reconstruction.
    use_F_tmp = in_compression * ti.cast(delta_gamma <= 0, ti.f32) * in_bounds
    F_new = use_F_tmp * F_tmp + (1 - use_F_tmp) * (
        U @ make_matrix_from_diag_3d(sig_out) @ V.transpose()
    )
    return F_new, sig_out, Jp_new


# Stress computation functions — identical signature for function-pointer
# compatibility: (F, C, U, sig_out, mu, lam, dt) -> stress; dtype inferred from inputs.


@ti.func
def compute_stress_elasticity(F, C, U, sig_out, mu, lam, dt):
    """Hencky (logarithmic) elasticity stress: tau = mu*B + (lam*log(J) - mu)*I."""
    J = F.determinant()
    scale = lam * ti.log(J) - mu
    identity = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stress = mu * (F @ F.transpose()) + scale * identity
    return stress


@ti.func
def compute_stress_neo_hookean(F, C, U, sig_out, mu, lam, dt):
    """
    Neo-Hookean Kirchhoff stress matching Kaolin/Simplicits (used by Vid2Sim):
    psi = mu/2*(I1-3) + lam/2*(J-1)^2 - mu*(J-1) => tau = mu*F@F^T + (lam*(J-1) - mu)*J*I.
    The -mu*J (not -mu) volumetric term is intentional for parity with that model;
    it linearizes to lam_eff = lam - mu.
    """
    J = F.determinant()
    scale = (lam * (J - 1) - mu) * J
    identity = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stress = mu * (F @ F.transpose()) + scale * identity
    return stress


@ti.func
def compute_stress_viscous_fluid(F, C, U, sig_out, mu, lam, dt):
    """Newtonian viscous fluid stress."""
    J = ti.max(F[0, 0], 1e-2)
    kappa = (2.0 / 3.0) * mu + lam
    identity = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    epsilon = 0.5 * (C + C.transpose())
    stress = kappa * identity * (J - 1 / (J**6)) + mu * epsilon * J
    return stress


@ti.func
def compute_stress_plasticine(F, C, U, sig_out, mu, lam, dt):
    """Kirchhoff stress for plasticine and non-Newtonian materials."""
    log_sig = ti.log(sig_out)
    tau = 2 * mu * log_sig + lam * log_sig.sum()
    stress = U @ make_matrix_from_diag_3d(tau) @ U.transpose()
    return stress


@ti.func
def compute_stress_drucker_prager(F, C, U, sig_out, mu, lam, dt):
    """Kirchhoff stress for Drucker-Prager plasticity."""
    log_sig = ti.log(sig_out)
    tau = 2 * mu * log_sig + lam * log_sig.sum()
    stress = U @ make_matrix_from_diag_3d(tau) @ U.transpose()
    return stress


@ti.func
def compute_stress_corotated(F, C, U, sig_out, mu, lam, dt):
    """Fixed corotated stress for plasticine with corotated elasticity."""
    # Recover R = U @ V^T from F, U, sig_out: V^T = diag(1/sig) @ U^T @ F
    inv_sig = 1.0 / sig_out
    R = U @ make_matrix_from_diag_3d(inv_sig) @ U.transpose() @ F
    J = F.determinant()
    identity = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stress = 2.0 * mu * (F - R) @ F.transpose() + lam * J * (J - 1.0) * identity
    return stress


# Material function registry

PROJECT_F_FUNCTIONS = {
    ELASTICITY: project_F_elasticity,
    NEO_HOOKEAN: project_F_neo_hookean,
    VISCOUS_FLUID: project_F_viscous_fluid,
    PLASTICINE: project_F_plasticine,
    NON_NEWTONIAN: project_F_non_newtonian,
    DRUCKER_PRAGER: project_F_drucker_prager,
    SAND: project_F_sand,
    PLASTICINE_COROTATED: project_F_plasticine_corotated,
}

COMPUTE_STRESS_FUNCTIONS = {
    ELASTICITY: compute_stress_elasticity,
    NEO_HOOKEAN: compute_stress_neo_hookean,
    VISCOUS_FLUID: compute_stress_viscous_fluid,
    PLASTICINE: compute_stress_plasticine,
    NON_NEWTONIAN: compute_stress_plasticine,
    DRUCKER_PRAGER: compute_stress_drucker_prager,
    SAND: compute_stress_drucker_prager,
    PLASTICINE_COROTATED: compute_stress_corotated,
}
