"""
Loss controller; Estimator only needs to import LossController.
No cyclic dependencies: controller -> loss_functions -> utils.
"""

import math

from simulator.loss_functions import (
    DistributionLoss,
    FlowLoss,
    RenderingLoss,
    SilhouetteLoss,
)


class LossController:
    """Manages all loss computation for Estimator."""

    def __init__(self, phys_args, estimator):
        config = vars(phys_args)

        # Only instantiate losses with weight > 0. Ordering matters: each loss
        # harvests xyz.grad into pos_grad_seq via accumulate_pos_grad, and
        # RenderingLoss clears xyz.grad before its own backward, so it must run last.
        self.pytorch_losses = []

        w_flow = phys_args.w_flow
        flow_code = config["flow_code"]  # empty flow_code disables flow
        if w_flow > 0 and flow_code:
            self.pytorch_losses.append(FlowLoss(w_flow, config))

        w_distribution = phys_args.w_distribution
        if w_distribution > 0:
            self.pytorch_losses.append(DistributionLoss(w_distribution, config))

        w_silhouette = phys_args.w_silhouette
        self.silhouette_loss = None
        if w_silhouette > 0:
            self.silhouette_loss = SilhouetteLoss(w_silhouette, config)
            self.pytorch_losses.append(self.silhouette_loss)

        w_img = phys_args.w_img
        w_alp = phys_args.w_alp

        self.rendering_loss = RenderingLoss(
            w_img,
            w_alp,
            config,
        )
        self.pytorch_losses.append(self.rendering_loss)

    def reset(self):
        for loss in self.pytorch_losses:
            loss.reset()

    def get_mean_alpha_iou(self):
        """Return the mean alpha IoU across all frames rendered this iteration, or None."""
        if self.rendering_loss and self.rendering_loss.alpha_iou_values:
            vals = self.rendering_loss.alpha_iou_values
            return sum(vals) / len(vals)
        return None

    def get_mean_com_distance(self):
        """Return mean CoM distance across frames this iteration, or None if unavailable."""
        if self.silhouette_loss and self.silhouette_loss.com_distance_values:
            vals = [
                v for v in self.silhouette_loss.com_distance_values if not math.isnan(v)
            ]
            return sum(vals) / len(vals) if vals else None
        return None

    def compute_all(self, estimator, frame, xyz, backward=True):
        """Compute all active losses for this frame."""
        for loss in self.pytorch_losses:
            if loss.should_compute(frame, estimator.training):
                loss.compute(
                    estimator,
                    frame,
                    xyz,
                    backward=backward,
                )

    def get_summary(self):
        """
        Per-loss summary for monitoring:
        {name: {weight, values (per-frame unweighted), total, weighted}}.
        """
        summary = {}

        for loss in self.pytorch_losses:
            loss.drain_pending()
            total = sum(loss.values) / len(loss.values) if loss.values else 0.0
            summary[loss.name] = {
                "weight": loss.weight,
                "values": loss.values.copy(),
                "total": total,
                "weighted": total * loss.weight,
            }

        return summary
