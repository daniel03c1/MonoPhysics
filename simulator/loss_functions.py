"""
Loss classes for Estimator; all store per-frame UNWEIGHTED values in self.values.
No imports from estimator.py (avoids circular imports).
"""

import torch
from abc import ABC, abstractmethod
from geomloss import SamplesLoss

from gaussian_renderer import (
    render,
    render_optical_flow,
    render_pixel_coords,
)
from lib.loss_utils import (
    alpha_loss_value,
    image_loss_value,
    knn_distribution_loss,
    optical_flow_loss_value,
)


class BaseLoss(ABC):
    """Abstract base for all loss functions."""

    def __init__(self, name, weight, config=None):
        self.name = name
        self.weight = weight
        self.values = []
        self._pending = []  # 0-D GPU tensors; drained in get_summary

    @abstractmethod
    def should_compute(self, frame, training):
        """Return True if this loss should compute at this frame."""
        pass

    @abstractmethod
    def compute(self, estimator, frame, xyz, **kwargs):
        """Compute loss, call backward, accumulate gradients."""
        pass

    def reset(self):
        """Reset per-iteration state. Override to clear additional state."""
        self.values.clear()
        self._pending.clear()

    def drain_pending(self):
        """Convert deferred 0-D tensors to Python floats with one host sync."""
        if self._pending:
            self.values.extend(torch.stack(self._pending).cpu().tolist())
            self._pending.clear()


class RenderingLoss(BaseLoss):
    """Combined RGB image and alpha mask rendering loss."""

    def __init__(self, w_img, w_alp, config):
        super().__init__("Img", max(w_img, w_alp), config)
        self.w_img = w_img
        self.w_alp = w_alp

        self.alpha_iou_values = []

    def reset(self):
        super().reset()
        self.alpha_iou_values.clear()

    def should_compute(self, frame, training):
        return training

    def compute(self, estimator, frame, xyz, backward=True, **kwargs):
        gaussians = estimator.scene.gaussians
        views = estimator.views[frame]
        d_xyz = xyz - gaussians.get_xyz()

        loss_img = torch.tensor(0.0, device=estimator.device, requires_grad=True)
        loss_alp = torch.tensor(0.0, device=estimator.device, requires_grad=True)

        iou_sum = 0.0
        iou_count = 0

        for view in views:
            gt_image = view.original_image.cuda()
            gt_alpha = view.gt_alpha_mask

            results = render(
                view,
                gaussians,
                estimator.pipeline,
                estimator.background,
                d_xyz,
            )
            image = results["render"]
            alpha = results["alpha"]

            # Hard IoU at 0.05 threshold; reported per iteration, not a loss term
            with torch.no_grad():
                pred_mask = alpha > 0.05
                gt_mask = gt_alpha > 0.05
                intersection = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                iou_sum += (intersection / union.clamp(min=1.0)).item()
                iou_count += 1

            # Update max_radii2D for MCMC pixel-dead detection
            with torch.no_grad():
                vis = results["visibility_filter"]
                rads = results["radii"].float()
                gaussians.max_radii2D[vis] = torch.max(
                    gaussians.max_radii2D[vis], rads[vis]
                )

            if self.w_img > 0.0:
                loss_img = loss_img + image_loss_value(
                    image, gt_image, estimator.image_op.lambda_dssim
                )

            if self.w_alp > 0.0:
                loss_alp = loss_alp + alpha_loss_value(alpha, gt_alpha)

        self.alpha_iou_values.append(iou_sum / iou_count)

        unweighted_loss = (loss_img + loss_alp) / len(views)
        self._pending.append(unweighted_loss.detach())

        loss = (self.w_img * loss_img + self.w_alp * loss_alp) / len(views)

        if backward:
            if xyz.grad is not None:
                xyz.grad = None
            loss.backward()

        estimator.accumulate_pos_grad(self.name, frame, xyz)


class FlowLoss(BaseLoss):
    """Optical flow loss (separated from image/alpha)."""

    def __init__(self, weight, config):
        super().__init__("Flow", weight, config)
        self.flow_code = config["flow_code"]

    def should_compute(self, frame, training):
        return training and frame > 0  # Skip frame 0 (no previous frame)

    def compute(self, estimator, frame, xyz, backward=True, **kwargs):
        """
        Average the per-component losses over flow_code chars: F: forward 0→t
        (rendered at frame-0 positions), f: forward (t-1)→t (at prev positions),
        B: backward t→0, b: backward t→(t-1) (both at current positions).
        """
        if estimator.prev_xyz is None:
            self._pending.append(torch.zeros((), device=estimator.device))
            return

        xyz.grad = None
        estimator.prev_xyz.grad = None

        gaussians = estimator.scene.gaussians
        views = estimator.views[frame]

        d_xyz = xyz - gaussians.get_xyz()
        zero_d_xyz = torch.zeros_like(d_xyz).requires_grad_(True)

        n_components = len(self.flow_code)
        total_loss = torch.tensor(0.0, device=estimator.device, requires_grad=True)

        prev_d_xyz = estimator.prev_xyz - gaussians.get_xyz()
        prev_uid_map = {v.uid: v for v in estimator.views[frame - 1]}

        for view in views:
            curr_fid = view.fid.item()
            uid = view.uid

            flows = estimator.get_optical_flow(curr_fid, uid)
            gt_alpha = view.gt_alpha_mask

            for i, char in enumerate(self.flow_code):
                gt_flow, uncertainty = flows[char]

                if char == "F":
                    result = render_optical_flow(
                        viewpoint_camera=view,
                        pc=gaussians,
                        velocities_world=d_xyz,
                        pipe=estimator.pipeline,
                        d_xyz=zero_d_xyz,
                    )
                    mask = estimator.view_maps[(0.0, uid)].gt_alpha_mask
                elif char == "f":
                    result = render_optical_flow(
                        viewpoint_camera=view,
                        pc=gaussians,
                        velocities_world=xyz - estimator.prev_xyz,
                        pipe=estimator.pipeline,
                        d_xyz=prev_d_xyz,
                    )
                    mask = prev_uid_map[uid].gt_alpha_mask
                elif char == "B":
                    result = render_optical_flow(
                        viewpoint_camera=view,
                        pc=gaussians,
                        velocities_world=zero_d_xyz - d_xyz,
                        pipe=estimator.pipeline,
                        d_xyz=d_xyz,
                    )
                    mask = gt_alpha
                elif char == "b":
                    result = render_optical_flow(
                        viewpoint_camera=view,
                        pc=gaussians,
                        velocities_world=estimator.prev_xyz - xyz,
                        pipe=estimator.pipeline,
                        d_xyz=d_xyz,
                    )
                    mask = gt_alpha
                else:
                    raise ValueError(
                        f"Unknown flow_code character '{char}'. Valid: F, f, B, b."
                    )

                component_loss = optical_flow_loss_value(
                    result["flow"],
                    gt_flow,
                    uncertainty,
                    mask,
                )

                total_loss = total_loss + component_loss

        n_total = len(views) * n_components
        unweighted_loss = total_loss / n_total
        self._pending.append(unweighted_loss.detach())

        loss = self.weight * total_loss / n_total
        if backward:
            loss.backward()

        # FlowLoss runs first, so this typically appends the frame's entry; later
        # losses add to it.
        estimator.accumulate_pos_grad(self.name, frame, xyz)

        if zero_d_xyz.grad is None:
            zero_d_xyz.grad = torch.zeros_like(zero_d_xyz)
        estimator.pos_grad_seq[0].add_(zero_d_xyz.grad)
        estimator.add_pos_grad_debug("Flow", 0, zero_d_xyz.grad)

        # Add prev_xyz.grad to PREVIOUS FRAME using explicit index (not [-2])
        if estimator.prev_xyz.grad is not None and frame > 0:
            prev_grad = estimator.prev_xyz.grad.clone()
            estimator.pos_grad_seq[frame - 1].add_(prev_grad)
            estimator.add_pos_grad_debug("Flow", frame - 1, prev_grad)


class SilhouetteLoss(BaseLoss):
    """
    2D silhouette loss: entropic-regularized Sinkhorn (geomloss) between the
    alpha-weighted rendered pixel coordinates and the GT mask pixels.
    """

    def __init__(self, weight, config):
        super().__init__("Sil", weight, config)
        self.com_distance_values = []

        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=config["sinkhorn_blur"],
            debias=True,
            scaling=0.9,
        )

    def reset(self):
        super().reset()
        self.com_distance_values.clear()

    def should_compute(self, frame, training):
        return training

    def accumulate_position_grad(self, estimator, frame, xyz, backward=True):
        """
        Render the pixel-coordinate silhouette for `frame`, accumulate the Sinkhorn
        position gradient into xyz.grad, return (total_unweighted_loss, com_distance).
        Touches neither estimator state nor this loss's accumulators; `compute` adds
        that bookkeeping.
        """
        gaussians = estimator.scene.gaussians
        d_xyz = xyz - gaussians.get_xyz()

        views = estimator.views[frame]

        total_unweighted_loss = torch.tensor(
            0.0, device=estimator.device, requires_grad=True
        )
        com_dist_sum = 0.0
        com_dist_count = 0

        if len(views) == 0:
            raise RuntimeError(
                f"[SilhouetteLoss] frame {frame} has no views to render."
            )

        loss_weight = self.weight / len(views)

        for view in views:
            gt_alpha = view.gt_alpha_mask  # [1, H, W]

            if gt_alpha is None:
                raise RuntimeError(
                    f"[SilhouetteLoss] frame {frame}, view {view.uid}: gt_alpha_mask "
                    "is None. Every training view must carry a GT silhouette."
                )

            with torch.no_grad():
                fg_mask = gt_alpha[0] > 0
                fg_indices = torch.nonzero(fg_mask, as_tuple=False)

            if fg_indices.shape[0] == 0:
                raise RuntimeError(
                    f"[SilhouetteLoss] frame {frame}, view {view.uid}: GT silhouette "
                    "has no foreground pixels (all-zero mask)."
                )

            gt_pix = torch.stack(
                [fg_indices[:, 1].float() + 0.5, fg_indices[:, 0].float() + 0.5],
                dim=-1,
            ).to(estimator.device)

            coord_result = render_pixel_coords(
                view,
                gaussians,
                estimator.pipeline,
                d_xyz=d_xyz,
            )
            alpha_map = coord_result["alpha"][0]  # [H, W]
            active_mask = alpha_map > 0.0
            rendered_points = coord_result["pixel_coords"].permute(1, 2, 0)[
                active_mask
            ]  # [n, 2]
            rendered_alphas = alpha_map[active_mask]

            gt_alphas = gt_alpha.squeeze(0)  # [H, W]
            gt_alphas = gt_alphas[gt_alphas > 0.0]

            if rendered_alphas.numel() == 0:
                raise RuntimeError(
                    f"[SilhouetteLoss] frame {frame}, view {view.uid}: pixel-coords "
                    f"rasterizer returned an all-zero alpha."
                )

            # Diagnostic alpha-weighted CoM distance (pixel units), logged via
            # com_distance_values. Not a loss term.
            if rendered_alphas.sum() > 0:
                rendered_com = (rendered_alphas[:, None] * rendered_points).sum(
                    0
                ) / rendered_alphas.sum()
                gt_com = (gt_alphas[:, None] * gt_pix).sum(0) / gt_alphas.sum()

                com_dist_sum += (rendered_com - gt_com).detach().norm().item()
                com_dist_count += 1

            # Sinkhorn (uses normalized alphas; manual autograd.grad avoids
            # backwarding through Sinkhorn's internals directly).
            sum_alphas = gt_alphas.sum()
            gt_alphas = gt_alphas / sum_alphas
            rendered_alphas = rendered_alphas / sum_alphas

            unweighted_loss = self.sinkhorn(
                rendered_alphas,
                rendered_points,
                gt_alphas,
                gt_pix,
            )
            view_loss = loss_weight * unweighted_loss

            # Only the point gradient is propagated: the alpha branch terminates at the
            # pixel-coords rasterizer, which detaches opacity, so backwarding it is a no-op.
            if backward:
                (rendered_points_grad,) = torch.autograd.grad(
                    view_loss, [rendered_points]
                )
                rendered_points.backward(
                    retain_graph=True, gradient=rendered_points_grad
                )

            total_unweighted_loss = total_unweighted_loss + unweighted_loss

        com_dist = com_dist_sum / com_dist_count if com_dist_count > 0 else float("nan")
        return total_unweighted_loss, com_dist

    def compute(self, estimator, frame, xyz, backward=True, **kwargs):
        xyz.grad = None

        total_unweighted_loss, com_dist = self.accumulate_position_grad(
            estimator, frame, xyz, backward=backward
        )

        self._pending.append(total_unweighted_loss.detach())
        self.com_distance_values.append(com_dist)

        estimator.accumulate_pos_grad(self.name, frame, xyz)


class DistributionLoss(BaseLoss):
    """
    Two-sided KNN spacing regularizer: for each particle's K nearest neighbors,
    penalizes distances outside [alpha_lower * dx, alpha_upper * dx] — a lower-bound
    hinge repels overcrowded pairs, an upper-bound hinge attracts strays.
    """

    def __init__(self, weight, config):
        super().__init__("Dist", weight, config)
        self.k = config.get("distribution_k", 3)
        self.alpha_lower = config.get("distribution_alpha_lower", 0.3)
        self.alpha_upper = config.get("distribution_alpha_upper", 0.8)

    def should_compute(self, frame, training):
        return training and (frame == 0)

    def compute(self, estimator, frame, xyz, backward=True, **kwargs):
        # DistributionLoss shouldn't affect scene_scale.
        x0 = estimator.scene.gaussians.get_xyz(detach_scene_scale=True)
        dx = float(estimator.sim.dx[None])
        unweighted_loss, _ = knn_distribution_loss(
            x0,
            k=self.k,
            r_lower=self.alpha_lower * dx,
            r_upper=self.alpha_upper * dx,
        )
        self._pending.append(unweighted_loss.detach())
        loss = self.weight * unweighted_loss
        if backward:
            loss.backward()
