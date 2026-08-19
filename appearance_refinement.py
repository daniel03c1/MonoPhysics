"""
Appearance refinement: the appearance-only optimization phases of training.

during - refine_round: per-iteration refinement on the cached rollout positions.
post   - refine_post: post-training polish over the whole cached rollout.

Optimization constants live in inv_problem.py and are passed in as arguments.
"""

import random

import torch
from tqdm import tqdm

from gaussian_renderer import render
from lib.loss_utils import alpha_loss_value, image_loss_value
from lib.training_utils import step_optimizer


def eligible_frames(max_f, threshold, per_frame_score):
    """Rolled-out frames whose score clears `threshold`; frame 0 always included."""
    frames = list(range(max_f))
    if threshold is not None and per_frame_score is not None:
        assert len(per_frame_score) >= max_f, (len(per_frame_score), max_f)
        frames = [f for f in frames if per_frame_score[f] >= threshold]
    if 0 not in frames:
        frames.append(0)
    return frames


def detached_render_offset(gaussians, target_xyz):
    """
    d_xyz whose live get_xyz() term cancels its own graph, so rendering at
    means3D = get_xyz() + d_xyz backpropagates only through `target_xyz`.
    """
    return target_xyz - gaussians.get_xyz()


def refine_frame(estimator, frame, w_img):
    """
    One appearance step on `frame` at its cached rollout position. Positions are
    simulator outputs and never receive gradient; only appearance params step.
    """
    gs = estimator.scene.gaussians

    d_xyz = detached_render_offset(gs, estimator.cached_positions[frame])
    view = estimator.views[frame][0]

    results = render(view, gs, estimator.pipeline, estimator.background, d_xyz)

    target_rgb = view.original_image.cuda()
    target_alpha = view.gt_alpha_mask

    frame_loss = torch.tensor(0.0, device=estimator.device)
    if w_img > 0:
        frame_loss = frame_loss + w_img * image_loss_value(
            results["render"], target_rgb, estimator.image_op.lambda_dssim
        )
    rendering_loss = estimator.loss_controller.rendering_loss
    if rendering_loss.w_alp > 0:
        frame_loss = frame_loss + rendering_loss.w_alp * alpha_loss_value(
            results["alpha"], target_alpha
        )
    frame_loss.backward()

    step_optimizer(gs.optimizer)


def refine_round(estimator, actual_f, n_steps, threshold):
    """Per-round refinement on the frames that clear the geometry gate."""
    w_img = estimator.loss_controller.rendering_loss.w_img
    frames = eligible_frames(actual_f, threshold, estimator.per_frame_center_inside)
    for _ in range(n_steps):
        refine_frame(estimator, random.choice(frames), w_img)


def refine_post(estimator, n_steps, w_img):
    """Post-training polish over the cached rollout."""
    frames = eligible_frames(len(estimator.cached_positions), None, None)
    for _ in tqdm(range(n_steps), desc="Color refinement"):
        refine_frame(estimator, random.choice(frames), w_img)


def snapshot_gs_state(gaussians):
    """Clone appearance params so a dt-halving retry can roll refinement back."""
    return [
        p.detach().clone()
        for group in gaussians.optimizer.param_groups
        for p in group["params"]
    ]


def restore_gs_state(gaussians, snapshot):
    params = [p for group in gaussians.optimizer.param_groups for p in group["params"]]
    with torch.no_grad():
        for p, saved in zip(params, snapshot):
            p.data.copy_(saved)
