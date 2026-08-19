import lpips
import os
import taichi as ti
import torch
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm

from appearance_refinement import (
    refine_post,
    refine_round,
    restore_gs_state,
    snapshot_gs_state,
)
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene_init import assign_gs_to_pcd, prepare_pcd, reference_camera
from simulator import Estimator, Simulator
from simulator.materials import SIG_MAX, SIG_MIN
from lib.data_utils import (
    load_gt_params,
    load_gt_pcds,
    read_estimation_result,
    write_dict_to_json,
)
from lib.debug_utils import save_debug_snapshot
from lib.eval import (
    eval_mat_est_acc,
    evaluate,
    evaluate_depth_scaled,
    export_result,
)
from lib.general_utils import safe_state
from lib.io_utils import save_experiment_configs
from lib.loss_utils import psnr, ssim
from lib.training_utils import (
    step_all_optimizers,
    zero_all_grads,
)
from lib.video_utils import create_video
from lib.volume_utils import compute_optimal_dx

LPIPS_FN = lpips.LPIPS(net="alex").cuda()

# optimization constants (kept here, not in appearance_refinement.py, so every tunable is in one file)
APPEARANCE_RESET_EVERY = 10  # iterations between opacity/scaling resets (per-round)
POST_REFINE_W_IMG = 1.0  # image-loss weight during the post-training polish
MAX_N_SUBSTEPS = 800  # substeps double on each dt-halving retry, up to this ceiling


def iter_train(
    estimator: Estimator,
    phys_args,
    max_f=None,
    gt_params=None,
    obj_name=None,
    pipe_args=None,
    background=None,
    experiment_dir=None,
    gt_cam_distance=None,
):
    losses = []

    if max_f is not None:
        estimator.max_f = max_f

    n_frames = min(phys_args.n_frames, len(estimator.views))

    max_n_particles = phys_args.n_max_particles
    if max_n_particles < 0:
        max_n_particles = len(estimator.scene.gaussians.get_xyz())

    n_iters = phys_args.n_iters
    n_init_frames = phys_args.n_init_frames
    full_rollout_at = phys_args.full_rollout_at

    frame_dt = estimator.sim.frame_dt
    n_substeps_initial = round(frame_dt / estimator.sim.dt_ori[None])

    max_retries = 1
    n_substeps_cap = n_substeps_initial
    while n_substeps_cap * 2 <= MAX_N_SUBSTEPS:
        n_substeps_cap *= 2
        max_retries += 1

    center_inside_threshold = phys_args.center_inside_threshold

    # How often the floor clamp actually bit; stays 0 on scenes that never penetrate.
    n_scale_clamped = 0

    for i in tqdm(range(n_iters)):
        # Rolled-out frames ramp from n_init_frames to n_frames by iteration
        # full_rollout_at.
        progress = min(1.0, (i + 1) / full_rollout_at)
        max_f = round(n_init_frames + (n_frames - n_init_frames) * progress)

        scheduled_max_f = max_f
        n_substeps = n_substeps_initial
        fwd_bwd_succeeded = False

        # Pre-refine: one rollout serves both the appearance refinement and the physics
        # losses. Per attempt: rollout -> center gate -> refine -> [re-rollout if frame-0
        # positions moved] -> loss pass -> adjoint backward.
        n_per_round_appearance_steps = phys_args.n_per_round_appearance_steps
        refine_active = n_per_round_appearance_steps > 0
        gs_snapshot = (
            snapshot_gs_state(estimator.scene.gaussians) if refine_active else None
        )

        for attempt in range(max_retries):
            # restore refinement state before re-refining on a dt-halving retry
            if attempt > 0 and gs_snapshot is not None:
                restore_gs_state(estimator.scene.gaussians, gs_snapshot)

            estimator.sim.set_dt(frame_dt / n_substeps)
            # 1. Rollout only; losses are computed separately in step 4.
            actual_f, forward_succeeded = forward(
                estimator,
                max_f=scheduled_max_f,
                auto_dt_halving=False,
                backward=False,
                skip_loss=True,
            )

            if not forward_succeeded:
                n_substeps *= 2
                print(
                    f"[{obj_name}] Forward failed (attempt {attempt + 1}/{max_retries}), "
                    f"new dt={frame_dt / n_substeps:.2e} (substeps={n_substeps})"
                )
                continue

            # 2. Appearance-invariant geometry gate from the cached positions.
            estimator.compute_center_inside(actual_f)

            # 3. Pre-refine appearance on the cached positions.
            if refine_active:
                if (i + 1) % APPEARANCE_RESET_EVERY == 0:
                    estimator.scene.gaussians.reset_opacity()
                    estimator.scene.gaussians.reset_scaling()
                refine_round(
                    estimator,
                    actual_f,
                    n_per_round_appearance_steps,
                    center_inside_threshold,
                )

            # 4. Separated loss pass on the cached positions, then the adjoint backward.
            estimator.compute_losses_on_cached(actual_f)
            pos_grad_dist = []  # fresh per attempt → auto-reset on dt-halving retries
            backward_succeeded = backward(
                estimator, max_f=actual_f, pos_grad_dist=pos_grad_dist
            )
            if backward_succeeded:
                max_f = actual_f
                fwd_bwd_succeeded = True
                break

            n_substeps *= 2
            print(
                f"[{obj_name}] Backward failed (attempt {attempt + 1}/{max_retries}), "
                f"new dt={frame_dt / n_substeps:.2e} (substeps={n_substeps})"
            )

        if not fwd_bwd_succeeded:
            raise RuntimeError(
                f"[{obj_name}] forward+backward did not succeed after "
                f"{max_retries} retries, terminating."
            )

        loss_summary = estimator.get_loss_summary()

        total_loss = sum(entry["weighted"] for entry in loss_summary.values())
        losses.append(total_loss)

        # entry["weighted"] is already a per-frame mean (see get_summary).
        message = {name: entry["weighted"] for name, entry in loss_summary.items()}

        if gt_params is not None:
            message.update(
                eval_mat_est_acc(
                    estimator, gt_params, prefix="", use_default_value=False
                )
            )

        if gt_cam_distance is not None and gt_cam_distance > 0:
            with torch.no_grad():
                cur_center = estimator.scene.gaussians.get_xyz().mean(dim=0)
                cam_center = estimator.scene.gaussians.c2w[:3, 3]
                cur_dist = torch.norm(cur_center - cam_center).item()
            message["scale_ratio"] = cur_dist / gt_cam_distance
        else:
            message["scale"] = estimator.scene.gaussians.get_scene_scale().item()

        mean_iou = estimator.loss_controller.get_mean_alpha_iou()
        if mean_iou is not None:
            message["IoU"] = mean_iou

        mean_com_dist = estimator.loss_controller.get_mean_com_distance()
        if mean_com_dist is not None:
            message["CoM"] = mean_com_dist

        print(
            f"[{obj_name}][iter {i + 1}/{n_iters}] "
            + ", ".join([f"{k}: {v:.3f}" for k, v in message.items()])
        )

        # Per-particle 'through' gradient (accumulated backward through the MPM dynamics).
        # Near-zero just after a substep-chunk boundary: pop_from_memory zeroes x.grad
        # and carries accumulation via the chunk-carry slot.
        if pos_grad_dist and phys_args.verbose:
            multi_chunk = (
                estimator.sim.n_substeps[None] * actual_f
                > estimator.sim.cuda_chunk_size
            )
            if multi_chunk:
                print(
                    f"[{obj_name}][iter {i + 1}/{n_iters}] pos_grad note: rollout spans "
                    "multiple substep chunks; 'through' for frames just after a chunk "
                    "boundary may read ~0 (artifact of grad-checkpoint carry slot)."
                )
            # first frame a particle entered collision range (sticky; None = no collision)
            collision_frame = estimator._first_collision_frame
            if collision_frame is None:
                print(
                    f"[{obj_name}][iter {i + 1}/{n_iters}] collision: "
                    "none detected this rollout"
                )
            else:
                print(
                    f"[{obj_name}][iter {i + 1}/{n_iters}] collision: "
                    f"first contact at frame f{collision_frame:02d}"
                )

            # per-loss breakdown of the injected rendering gradient (Flow incl. cross-frame terms)
            pos_grad_debug = estimator.pos_grad_debug
            for e in sorted(pos_grad_dist, key=lambda d: d["frame"]):
                f = e["frame"]

                loss_parts = []
                for lname in ("Img", "Sil", "Flow"):
                    seq = pos_grad_debug.get(lname)
                    if seq is not None and f < len(seq) and seq[f] is not None:
                        p50, p95, p100 = grad_pcts(seq[f].norm(dim=-1))
                        loss_parts.append(f"{lname} |{p50:.1e} {p95:.1e} {p100:.1e}|")
                loss_str = (" " + " ".join(loss_parts)) if loss_parts else ""
                collision_str = "  <-- first collision" if f == collision_frame else ""

                print(
                    f"[{obj_name}][iter {i + 1}/{n_iters}] f{f:02d} pos_grad "
                    f"through |{e['through_p50']:.1e} {e['through_p95']:.1e} "
                    f"{e['through_max']:.1e}|"
                    f"{loss_str}{collision_str}"
                )

        debugsave_every = phys_args.debugsave_every
        if (
            debugsave_every > 0
            and experiment_dir is not None
            and (i + 1) % debugsave_every == 0
        ):
            save_debug_snapshot(estimator, i + 1, experiment_dir, pipe_args, background)

        step_all_optimizers(estimator, max_f)
        zero_all_grads(
            estimator
        )  # the material optimizer is skipped before contact, so it never self-zeroes

        # Keep the frame-0 state out of the floor. Scale-only: positions are never
        # edited. Unconditional, because _xyz moves every iteration even when the
        # scale optimizer does not step, so a penetration can appear with the scale
        # unchanged.
        # Inert whenever nothing is below the plane, so free-fall runs are unaffected.
        factor = estimator.scene.gaussians.clamp_scene_scale_to_ground(
            estimator.ground_plane
        )
        if factor != 1.0:
            n_scale_clamped += 1

        # MCMC: relocate + add new
        if (
            not phys_args.freeze_geometry
            and phys_args.mcmc_refine_every > 0
            and (i + 1) % phys_args.mcmc_refine_every == 0
            and (i < n_iters - 1)
        ):
            induced_volumes = estimator.scene.gaussians.get_induced_volumes().detach()

            estimator.scene.gaussians.relocate(
                min_opacity=phys_args.mcmc_min_opacity,
                induced_volumes=induced_volumes,
            )

            # Calculate max_add_ratio based on remaining steps
            current_n = len(estimator.scene.gaussians.get_xyz())

            # Particle growth shares the rollout ramp's end point.
            remaining_iters = full_rollout_at - i
            remaining_steps = max(
                1, int(remaining_iters // phys_args.mcmc_refine_every)
            )

            if current_n < max_n_particles:
                max_add_ratio = (max_n_particles / current_n) ** (
                    1 / remaining_steps
                ) - 1
                max_add_ratio = min(max_add_ratio, 1.0)
            else:
                max_add_ratio = 0.0

            if i <= full_rollout_at:
                estimator.scene.gaussians.add_new(
                    max_n_particles,
                    phys_args.mcmc_min_opacity,
                    max_add_ratio=max_add_ratio,
                    induced_volumes=induced_volumes,
                )
                # Fresh stats for next MCMC interval.

            estimator.scene.gaussians.reset_stats()
            torch.cuda.empty_cache()

    print(
        f"[Floor clamp] scene_scale projected on {n_scale_clamped}/{n_iters} iterations"
    )

    # Post-training color refinement: appearance-only polish after main loop
    n_post_appearance_steps = phys_args.n_post_appearance_steps
    if n_post_appearance_steps > 0 and estimator.cached_positions:
        forward(estimator, max_f=n_frames, backward=False, skip_loss=True)

        estimator.scene.gaussians.reset_color()
        estimator.scene.gaussians.reset_opacity()
        estimator.scene.gaussians.reset_scaling()

        refine_post(estimator, n_post_appearance_steps, POST_REFINE_W_IMG)

    return losses


def forward(
    estimator: Estimator,
    max_f=None,
    return_positions=False,
    backward=True,
    skip_loss=False,
    auto_dt_halving=True,
):
    frame_dt = estimator.sim.frame_dt
    n_substeps = round(frame_dt / estimator.sim.dt_ori[None])
    actual_f = 0

    if max_f is None:
        max_f = estimator.max_f

    # Verbose-only: have the simulator tally sigma clamps during each substep.
    verbose = estimator.phys_args.verbose
    estimator.sim.count_sig_clamps_enabled = verbose

    while True:
        actual_f = 0
        is_nan = False
        losses = []

        if auto_dt_halving:
            assert n_substeps <= MAX_N_SUBSTEPS

        estimator.initialize()

        if auto_dt_halving:
            estimator.sim.set_dt(frame_dt / n_substeps)

        positions = []

        for idx in range(max_f):
            x = estimator.forward(idx, backward=backward, skip_loss=skip_loss)

            is_nan = torch.any(torch.isnan(x))
            if is_nan:
                break

            positions.append(x)
            actual_f = idx + 1

        if verbose:
            sim = estimator.sim
            lo = sim.n_sig_clamped_lo[None]
            hi = sim.n_sig_clamped_hi[None]
            denom = estimator.get_n_particles() * sim._clamp_substep_count * sim.dim
            if denom > 0:
                print(
                    f"[sigma clamp] lower(<{SIG_MIN}): {lo} "
                    f"({100.0 * lo / denom:.4f}%), "
                    f"upper(>{SIG_MAX}): {hi} "
                    f"({100.0 * hi / denom:.4f}%) "
                    f"of {denom} sigma components over {sim._clamp_substep_count} substeps"
                )

        if not auto_dt_halving:
            break

        if not estimator.succeed():
            n_substeps *= 2
            print(
                f"CFL violated (v_max={estimator.sim.v_max_observed[None]:.4f}), "
                f"new dt={frame_dt / n_substeps:.6e}, step cnt {n_substeps}"
            )
        elif is_nan:
            n_substeps *= 2
            print(
                f"NaN occurred during forward pass. "
                f"new dt={frame_dt / n_substeps:.6e}, step cnt {n_substeps}"
            )
        else:
            break

    forward_succeeded = estimator.succeed() and not is_nan
    if return_positions:
        return actual_f, positions, forward_succeeded
    return actual_f, forward_succeeded


def grad_pcts(mag):
    """
    Return (p50, p95, p100) of a [N] per-particle gradient-magnitude tensor.

    torch.quantile's input is capped at ~16.7M elements; N particles is well under
    this, so no chunking is needed.
    """
    return (
        torch.quantile(mag, 0.50).item(),
        torch.quantile(mag, 0.95).item(),
        mag.max().item(),
    )


def backward(
    estimator: Estimator,
    max_f=None,
    pos_grad_dist=None,
):
    """
    Run the per-frame backward pass. If `pos_grad_dist` is a list, it is
    filled per frame with the 'through' gradient distribution (the term that
    compounds/explodes), read before the rendering source is injected.
    """
    if max_f is None:
        max_f = estimator.max_f

    estimator.clear_grads()

    sim = estimator.sim
    n_sub = sim.n_substeps[None]
    chunk = sim.cuda_chunk_size
    grad_buf = None

    for i in reversed(range(max_f)):
        if pos_grad_dist is not None:
            # read the through-sim gradient BEFORE backward(i) injects the rendering source
            n_particles = sim.n_particles[None]
            if grad_buf is None or grad_buf.shape[0] != n_particles:
                grad_buf = torch.empty(
                    (n_particles, 3), dtype=torch.float32, device=estimator.device
                )
            slot = (i * n_sub) % chunk
            sim.read_x_grad_slot(slot, grad_buf)
            through_mag = grad_buf.norm(dim=-1)  # [N]

            through_p50, through_p95, through_max = grad_pcts(through_mag)
            pos_grad_dist.append(
                {
                    "frame": i,
                    "through_p50": through_p50,
                    "through_p95": through_p95,
                    "through_max": through_max,
                }
            )

        if not estimator.backward(i):
            return False
    return True


@torch.no_grad()
def inference(
    base_simulator,
    gaussians,
    x_list,
    pipe_args,
    n_train_frames,
    views,
    target_positions=None,
    target_params=None,
    background=None,
    render_sim_path=None,
    render_gt_path=None,
    eval_pc_seed=None,
    eval_pc_until=None,
    train_views=None,
    train_cam_id=0,
):
    """
    Run inference and evaluation. `base_simulator` may be an Estimator or a
    Simulator (for eval_mat_est_acc); `x_list` holds particle positions per frame.
    """
    performances = {}

    # Output paths
    if render_sim_path is not None:
        os.makedirs(render_sim_path, exist_ok=True)
    if render_gt_path is not None:
        os.makedirs(render_gt_path, exist_ok=True)

    # Defaults
    if background is None:
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    if train_views is None:
        train_views = []
    views = sorted(views + train_views, key=lambda x: x.fid)
    frames = torch.unique(torch.cat([v.fid for v in views]))

    # 2D metrics
    train_psnr_list = []
    train_ssim_list = []
    train_lpips_list = []
    test_psnr_list = []
    test_ssim_list = []
    test_lpips_list = []
    test_mono_psnr_list = []
    test_mono_ssim_list = []
    test_mono_lpips_list = []
    train_iou_list = []
    test_iou_list = []
    test_mono_iou_list = []

    for f in range(len(frames)):
        # Filter views for this frame
        curr_views = [v for v in views if v.fid == frames[f]]

        xyz = x_list[f]
        d_xyz = xyz - gaussians.get_xyz()

        for view in curr_views:
            results = render(view, gaussians, pipe_args, background, d_xyz)
            image = results["render"]
            # recover the premultiplied foreground before compositing over the per-view
            # background, so the wbg render is correct for any default background color
            image_with_bg = image + (1 - results["alpha"]) * (
                view.real_background.cuda() - background.view(3, 1, 1)
            )
            gt_image = view.original_image.cuda()

            if render_sim_path is not None:
                torchvision.utils.save_image(
                    image, os.path.join(render_sim_path, f"{view.uid}_{f:05d}_wobg.png")
                )
                torchvision.utils.save_image(
                    image_with_bg,
                    os.path.join(render_sim_path, f"{view.uid}_{f:05d}_wbg.png"),
                )
            if render_gt_path is not None:
                torchvision.utils.save_image(
                    gt_image,
                    os.path.join(render_gt_path, f"{view.uid}_{f:05d}_wobg.png"),
                )
                torchvision.utils.save_image(
                    view.gt_image.cuda(),
                    os.path.join(render_gt_path, f"{view.uid}_{f:05d}_wbg.png"),
                )

            if view.gt_alpha_mask is not None:
                with torch.no_grad():
                    alpha = results["alpha"]
                    gt_alpha = view.gt_alpha_mask.cuda()
                    iou_val = (
                        torch.minimum(alpha, gt_alpha).sum()
                        / torch.maximum(alpha, gt_alpha).sum().clamp(min=1.0)
                    ).item()
            else:
                iou_val = None

            if view in train_views:
                continue
            elif f < n_train_frames:
                train_psnr_list.append(psnr(image, gt_image))
                train_ssim_list.append(ssim(image, gt_image))
                train_lpips_list.append(
                    LPIPS_FN(image * 2 - 1, gt_image * 2 - 1).mean().item()
                )
                if iou_val is not None:
                    train_iou_list.append(iou_val)
            else:
                test_psnr_list.append(psnr(image, gt_image))
                test_ssim_list.append(ssim(image, gt_image))
                test_lpips_list.append(
                    LPIPS_FN(image * 2 - 1, gt_image * 2 - 1).mean().item()
                )
                if iou_val is not None:
                    test_iou_list.append(iou_val)
                if view.uid == train_cam_id:
                    test_mono_psnr_list.append(psnr(image, gt_image))
                    test_mono_ssim_list.append(ssim(image, gt_image))
                    test_mono_lpips_list.append(
                        LPIPS_FN(image * 2 - 1, gt_image * 2 - 1).mean().item()
                    )
                    if iou_val is not None:
                        test_mono_iou_list.append(iou_val)

    psnr_list = train_psnr_list + test_psnr_list
    ssim_list = train_ssim_list + test_ssim_list
    lpips_list = train_lpips_list + test_lpips_list
    if len(train_psnr_list) > 0:
        train_psnr = torch.mean(torch.stack(train_psnr_list)).item()
        train_ssim = torch.mean(torch.stack(train_ssim_list)).item()
        train_lpips = sum(train_lpips_list) / len(train_lpips_list)
    else:
        train_psnr = 0.0
        train_ssim = 0.0
        train_lpips = 0.0

    if len(test_psnr_list) > 0:
        test_psnr = torch.mean(torch.stack(test_psnr_list)).item()
        test_ssim = torch.mean(torch.stack(test_ssim_list)).item()
        test_lpips = sum(test_lpips_list) / len(test_lpips_list)
    else:
        test_psnr = 0.0
        test_ssim = 0.0
        test_lpips = 0.0

    train_iou = sum(train_iou_list) / len(train_iou_list) if train_iou_list else 0.0
    test_iou = sum(test_iou_list) / len(test_iou_list) if test_iou_list else 0.0

    # 3D metrics
    if target_positions is not None:
        train_cd, test_cd, cd_list = evaluate(
            x_list,
            target_positions,
            n_train_frames,
            "CD",
            seed=eval_pc_seed,
            until=eval_pc_until,
        )
        train_emd, test_emd, emd_list = evaluate(
            x_list,
            target_positions,
            n_train_frames,
            "EMD",
            seed=eval_pc_seed,
            until=eval_pc_until,
        )
    else:
        train_cd = 0.0
        test_cd = 0.0
        cd_list = []
        train_emd = 0.0
        test_emd = 0.0
        emd_list = []

    # Depth-scale-aligned CD/EMD at frame 0
    if target_positions is not None:
        all_views = train_views if train_views else views
        ref_cam = next((v for v in all_views if v.uid == train_cam_id), all_views[0])
        ds_cd, ds_emd = evaluate_depth_scaled(x_list[0], target_positions[0], ref_cam)
    else:
        ds_cd, ds_emd = 0.0, 0.0

    # State and Material
    if target_params is not None:
        performances.update(eval_mat_est_acc(base_simulator, target_params))

    camera_ids = list(set(view.uid for view in views))
    for path in [render_sim_path, render_gt_path]:
        if path is None:
            continue

        for cam_id in camera_ids:
            for ext in ["gif", "mp4"]:
                create_video(
                    input_pattern=f"{cam_id}_*_wobg.png",
                    output_path=f"_cam{cam_id}_wobg.{ext}",
                    framerate=30,
                    duration=1,
                    cwd=path,
                )

                create_video(
                    input_pattern=f"{cam_id}_*_wbg.png",
                    output_path=f"_cam{cam_id}_wbg.{ext}",
                    framerate=30,
                    duration=1,
                    cwd=path,
                )

    performances["train_psnr"] = train_psnr
    performances["train_ssim"] = train_ssim
    performances["train_lpips"] = train_lpips
    performances["test_psnr"] = test_psnr
    performances["test_ssim"] = test_ssim
    performances["test_lpips"] = test_lpips
    if test_mono_psnr_list:
        performances["test_psnr_mono"] = torch.mean(
            torch.stack(test_mono_psnr_list)
        ).item()
        performances["test_ssim_mono"] = torch.mean(
            torch.stack(test_mono_ssim_list)
        ).item()
        performances["test_lpips_mono"] = sum(test_mono_lpips_list) / len(
            test_mono_lpips_list
        )
    else:
        performances["test_psnr_mono"] = 0.0
        performances["test_ssim_mono"] = 0.0
        performances["test_lpips_mono"] = 0.0
    performances["train_iou"] = train_iou
    performances["test_iou"] = test_iou
    performances["test_iou_mono"] = (
        sum(test_mono_iou_list) / len(test_mono_iou_list) if test_mono_iou_list else 0.0
    )
    performances["train_cd"] = train_cd
    performances["test_cd"] = test_cd
    performances["train_emd"] = train_emd
    performances["test_emd"] = test_emd
    performances["depth_scaled_cd"] = ds_cd
    performances["depth_scaled_emd"] = ds_emd

    for i, psnr_value in enumerate(psnr_list):
        performances[f"psnr{i:02d}"] = psnr_value.mean().item()
    for i, ssim_value in enumerate(ssim_list):
        performances[f"ssim{i:02d}"] = ssim_value.item()
    for i, lpips_value in enumerate(lpips_list):
        performances[f"lpips{i:02d}"] = lpips_value
    iou_list = train_iou_list + test_iou_list
    for i, iou_value in enumerate(iou_list):
        performances[f"iou{i:02d}"] = iou_value
    for i, cd_value in enumerate(cd_list):
        performances[f"cd{i:02d}"] = cd_value
    for i, emd_value in enumerate(emd_list):
        performances[f"emd{i:02d}"] = emd_value

    return performances


"""    Main Entry Point Helpers    """


def build_argument_parser():
    """Build the argument parser with all required arguments."""
    parser = ArgumentParser(description="Physical parameter estimation")

    parser.add_argument(
        "--detach_induced_volumes",
        action="store_true",
        help="Detach induced volumes so volume gradients don't flow back to positions.",
    )

    parser.add_argument("--postfix", default="", type=str)

    parser.add_argument("--cam_idx", type=int, required=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame per-particle pos_grad diagnostics each iteration.",
    )

    parser.add_argument(
        "--freeze_geometry",
        action="store_true",
        help="Freeze particle positions during inverse optimization: the position "
        "optimizer is not created and MCMC relocation/growth is disabled. Appearance, "
        "material, and state parameters are still optimized.",
    )

    return parser


def parse_and_normalize_args():
    """
    Parse and normalize CLI arguments.
    Returns (gs_args, phys_args, model_args, pipe_args, opt_args, config_id).
    """
    parser = build_argument_parser()
    model_parser = ModelParams(parser)
    pipe_parser = PipelineParams(parser)
    opt_parser = OptimizationParams(parser)

    gs_args, phys_args = get_combined_args(parser)

    model_args = model_parser.extract(gs_args)
    pipe_args = pipe_parser.extract(gs_args)
    opt_args = opt_parser.extract(gs_args)

    config_id = phys_args.id

    # gs_args and phys_args are already merged with precedence CLI > config > default
    # (see get_combined_args); phys_args is a superset carrying every gs key.

    # Parse camera indices
    model_args.model_path = os.path.abspath(model_args.model_path)
    model_args.cam_idx = gs_args.cam_idx

    return gs_args, phys_args, model_args, pipe_args, opt_args, config_id


def run_training(
    gs_args,
    phys_args,
    model_args,
    pipe_args,
    opt_args,
    background,
    gt_positions,
    gt_params,
    config_id,
    image_scale,
    experiment_dir=None,
):
    """
    Run the training pipeline.
    Returns (scene, estimator, train_views, test_views).
    """
    print("Preparing Gaussians for training.")

    # Load the per-camera aligned Gaussians (loaded verbatim, then sampled)
    vol, init_opacities, cam_info = prepare_pcd(
        model_args,
        pipe_args,
        phys_args,
        image_scale=image_scale,
    )
    torch.cuda.empty_cache()

    # unified dx for iP2G volume estimation and MPM; phys_args.voxel_size is canonical
    phys_args.voxel_size = compute_optimal_dx(vol)
    print(f"[unified dx] voxel_size = {phys_args.voxel_size:.5f}")

    scene = assign_gs_to_pcd(
        vol,
        init_opacities,
        model_args,
        cam_info,
    )
    scene.gaussians.active_sh_degree = scene.gaussians.max_sh_degree
    scene.gaussians._dx = phys_args.voxel_size

    train_views = scene.getTrainCameras(scale=model_args.res_scale)
    test_views = scene.getTestCameras(scale=model_args.res_scale)
    print("N_CAMS: ", len(test_views))

    appearance_init_dir = os.path.join(model_args.model_path, "appearance_init")

    # c2w must come from the camera whose space prepare_pcd left the Gaussians in.
    # The particles carry random colors (appearance was refined pre-sampling on the
    # aligned Gaussians); their colors are learned in the main optimization.
    c2w_camera = reference_camera(scene)
    c2w = c2w_camera.world_view_transform.inverse().T  # W2C -> C2W, column to row major
    scene.gaussians.set_c2w(c2w)

    # GT camera distance for scale_ratio logging (computed once)
    gt_cam_distance = None
    if gt_positions:
        with torch.no_grad():
            com_gt = gt_positions[0].mean(dim=0).to(c2w.device)
            gt_cam_distance = torch.norm(com_gt - c2w[:3, 3]).item()

    # Compute and cache initial-particle depth-scaled CD/EMD against GT at frame 0 (skip if done)
    appearance_init_metrics_path = os.path.join(appearance_init_dir, "metrics.json")
    if gt_positions and not os.path.exists(appearance_init_metrics_path):
        with torch.no_grad():
            ci_pred = scene.gaussians.get_xyz().detach()
            ci_cd, ci_emd = evaluate_depth_scaled(ci_pred, gt_positions[0], c2w_camera)
        os.makedirs(appearance_init_dir, exist_ok=True)
        write_dict_to_json({"cd": ci_cd, "emd": ci_emd}, appearance_init_metrics_path)
        print(f"Appearance-init depth-scaled metrics: CD={ci_cd:.4f}, EMD={ci_emd:.4f}")

    scene.gaussians.training_setup(
        opt_args, fix_pcd=True, freeze_geometry=phys_args.freeze_geometry
    )
    if phys_args.freeze_geometry:
        print("[FREEZE_GEOMETRY] Particle positions frozen; MCMC disabled.")

    # Determine number of frames
    if hasattr(phys_args, "n_frames"):
        n_frames = phys_args.n_frames
    else:
        n_frames = len(torch.unique(torch.cat([v.fid for v in train_views])))
        phys_args.n_frames = n_frames
    assert n_frames > 0

    estimator = Estimator(
        phys_args,
        "float32",
        positions=scene.gaussians.get_xyz(),
        dynamic_scene=scene,
        image_scale=image_scale,
        pipeline=pipe_args,
        image_op=opt_args,
        background=background,
    )

    # Extract obj_name for output paths
    obj_name = model_args.model_path.split("/")[-1]

    losses = iter_train(
        estimator,
        phys_args,
        max_f=n_frames,
        gt_params=gt_params,
        obj_name=obj_name,
        pipe_args=pipe_args,
        background=background,
        experiment_dir=experiment_dir,
        gt_cam_distance=gt_cam_distance,
    )

    # Log volume statistics
    vols = estimator.scene.gaussians.get_volumes()
    print("\n[TRAIN] Final induced volumes:")
    print(f"  Count: {len(vols)} particles")
    print(f"  Range: [{vols.min().item():.6e}, {vols.max().item():.6e}]")
    print(f"  Mean:  {vols.mean().item():.6e}")
    print(f"  Std:   {vols.std().item():.6e}")

    export_result(
        model_args,
        phys_args,
        estimator,
        losses,
        config_id,
        prefix="iter",
        postfix=gs_args.postfix,
        output_dir=experiment_dir,
    )

    return scene, estimator, train_views, test_views


def load_trained_run(
    gs_args,
    phys_args,
    model_args,
    pipe_args,
    image_scale,
    experiment_dir,
):
    """
    Rebuild the scene and load already-trained Gaussians from saved artifacts
    (gs.ply + predictions.json) so evaluation reruns without re-optimizing.
    gs.ply stores world-space positions, so the freshly loaded Gaussians keep
    c2w=None and scene_scale=1 and get_xyz returns those positions unchanged.
    """
    print("[STAGE] Trained artifacts found — skipping optimization; evaluating gs.ply.")

    # Cameras only; particle sampling is cached, so this is cheap.
    vol, init_opacities, cam_info = prepare_pcd(
        model_args,
        pipe_args,
        phys_args,
        image_scale=image_scale,
    )
    torch.cuda.empty_cache()

    scene = assign_gs_to_pcd(vol, init_opacities, model_args, cam_info)

    # trained voxel_size is canonical; read it back rather than recomputing from vol
    pred = read_estimation_result(
        model_args,
        phys_args,
        pred_file=os.path.join(experiment_dir, "predictions.json"),
    )
    scene.gaussians.load_ply(os.path.join(experiment_dir, "gs.ply"))
    scene.gaussians.active_sh_degree = scene.gaussians.max_sh_degree
    scene.gaussians._dx = pred["voxel_size"]

    train_views = scene.getTrainCameras(scale=model_args.res_scale)
    test_views = scene.getTestCameras(scale=model_args.res_scale)
    print("N_CAMS: ", len(test_views))
    return scene, train_views, test_views


def run_evaluation(
    scene,
    gs_args,
    phys_args,
    model_args,
    pipe_args,
    background,
    gt_positions,
    gt_params,
    train_views,
    test_views,
    experiment_dir=None,
):
    """Run evaluation, write performance.json, and return the metrics dict."""
    # Load learned physics parameters and merge with config
    if experiment_dir is not None:
        pred_file = os.path.join(experiment_dir, "predictions.json")
    else:
        pred_file = os.path.join(
            model_args.model_path, f"iter-pred{gs_args.postfix}.json"
        )
    estimated_params = read_estimation_result(
        model_args, phys_args, pred_file=pred_file
    )

    # merge estimated params into phys_args. predictions.json stores the initial
    # velocity as 'vel' but the simulator reads 'init_vel', so it needs remapping --
    # without it the eval rollout silently runs with the config's velocity.
    for key, value in estimated_params.items():
        if key == "vel":
            key = "init_vel"
        setattr(phys_args, key, value)
    estimation_params = phys_args

    # Volumes are induced from positions
    volumes = scene.gaussians.get_volumes().detach()
    print(
        f"[EVAL] Induced volumes: {len(volumes)} particles, "
        f"range=[{volumes.min().item():.6e}, {volumes.max().item():.6e}], "
        f"mean={volumes.mean().item():.6e}"
    )

    simulator = Simulator(estimation_params, scene.gaussians.get_xyz(), volumes)
    simulator.scene = scene

    model_path = os.path.abspath(model_args.model_path)
    obj_name = model_args.model_path.split("/")[-1]
    gt_path = os.path.join(model_path, f"{obj_name}_img_gt")
    if experiment_dir is not None:
        img_path = os.path.join(experiment_dir, "images")
    else:
        img_path = os.path.join(
            model_path, f"{obj_name}_img_render_iter{gs_args.postfix}"
        )

    # Run forward simulation
    all_views = test_views + (train_views or [])
    total_frames = len(torch.unique(torch.cat([v.fid for v in all_views])))
    n_gt_positions = 0 if gt_positions is None else len(gt_positions)
    x_list = forward(
        simulator,
        max_f=max(total_frames, n_gt_positions),
        return_positions=True,
    )[1]

    # Run evaluation
    train_cam_id = gs_args.cam_idx
    performances = inference(
        simulator,
        scene.gaussians,
        x_list,
        pipe_args,
        phys_args.n_frames,
        test_views,
        gt_positions,
        gt_params,
        background=background,
        render_sim_path=img_path,
        render_gt_path=gt_path,
        train_views=train_views,
        train_cam_id=train_cam_id,
    )

    # Save results
    if experiment_dir is not None:
        perf_path = os.path.join(experiment_dir, "performance.json")
    else:
        perf_path = os.path.join(model_path, f"iter_perf{gs_args.postfix}.json")

    write_dict_to_json(performances, perf_path)

    return performances


def print_metrics_summary(performances):
    """Print the metrics summary to console."""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    # 2D Image Metrics
    if "train_psnr" in performances:
        print("\n2D Image Metrics:")
        print(f"  Train PSNR:  {performances['train_psnr']:.4f}")
        print(f"  Test PSNR:   {performances['test_psnr']:.4f}")
        print(f"  Train SSIM:  {performances['train_ssim']:.4f}")
        print(f"  Test SSIM:   {performances['test_ssim']:.4f}")
        print(f"  Train LPIPS: {performances['train_lpips']:.4f}")
        print(f"  Test LPIPS:  {performances['test_lpips']:.4f}")
        if (
            performances.get("train_iou", 0.0) != 0.0
            or performances.get("test_iou", 0.0) != 0.0
        ):
            print(f"  Train IoU:   {performances['train_iou']:.4f}")
            print(f"  Test IoU:    {performances['test_iou']:.4f}")

    if "test_psnr_mono" in performances:
        print("\n2D Image Metrics (Monocular Future Prediction):")
        print(f"  Test PSNR (mono):  {performances['test_psnr_mono']:.4f}")
        print(f"  Test SSIM (mono):  {performances['test_ssim_mono']:.4f}")
        print(f"  Test LPIPS (mono): {performances['test_lpips_mono']:.4f}")
        if performances.get("test_iou_mono", 0.0) != 0.0:
            print(f"  Test IoU (mono):   {performances['test_iou_mono']:.4f}")

    # 3D Geometry Metrics
    if "train_cd" in performances:
        print("\n3D Geometry Metrics:")
        print(f"  Train CD:  {performances['train_cd']:.6e}")
        print(f"  Test CD:   {performances['test_cd']:.6e}")
        print(f"  Train EMD: {performances['train_emd']:.6e}")
        print(f"  Test EMD:  {performances['test_emd']:.6e}")

    # material-parameter accuracy; eval_mat_est_acc keys are prefixed 'MAE '
    mat_keys = [k for k in performances.keys() if k.startswith("MAE ")]
    if mat_keys:
        print("\nMaterial Parameter Estimation Accuracy:")
        for key in sorted(mat_keys):
            print(f"  {key}: {performances[key]:.6e}")

    # Per-frame metrics summary
    per_frame_psnr = [
        k for k in performances.keys() if k.startswith("psnr") and k[4:].isdigit()
    ]
    if per_frame_psnr:
        n_frames = len(per_frame_psnr)
        print(f"\nPer-frame metrics: {n_frames} frames computed")
        print("  (Individual frame metrics saved to JSON file)")

    print("=" * 80 + "\n")


"""    Main Entry Point    """
if __name__ == "__main__":
    gs_args, phys_args, model_args, pipe_args, opt_args, config_id = (
        parse_and_normalize_args()
    )

    # Initialize environment
    safe_state(gs_args.quiet, None)
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.6)

    # Setup background
    background = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(background, dtype=torch.float32, device="cuda")
    image_scale = 1.0

    gt_positions = load_gt_pcds(gs_args.source_path)
    gt_params = load_gt_params(gs_args.source_path)

    # Run training
    # Prepare experiment directory
    if not gs_args.postfix:
        experiment_dir_name = "run"
    elif gs_args.postfix.startswith("_"):
        experiment_dir_name = f"run{gs_args.postfix}"
    else:
        experiment_dir_name = f"run_{gs_args.postfix}"

    experiment_dir = os.path.join(model_args.model_path, experiment_dir_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "images"), exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    save_experiment_configs(
        experiment_dir, gs_args, phys_args, model_args, pipe_args, opt_args
    )

    # Training stage — skip if its artifacts (gs.ply + predictions.json) already
    # exist, so only evaluation reruns (delete performance.json to re-evaluate,
    # or delete gs.ply to retrain from scratch).
    gs_ply_path = os.path.join(experiment_dir, "gs.ply")
    pred_json_path = os.path.join(experiment_dir, "predictions.json")
    train_done = os.path.exists(gs_ply_path) and os.path.exists(pred_json_path)

    if train_done:
        scene, train_views, test_views = load_trained_run(
            gs_args,
            phys_args,
            model_args,
            pipe_args,
            image_scale,
            experiment_dir,
        )
    else:
        scene, estimator, train_views, test_views = run_training(
            gs_args,
            phys_args,
            model_args,
            pipe_args,
            opt_args,
            background,
            gt_positions,
            gt_params,
            config_id,
            image_scale,
            experiment_dir=experiment_dir,
        )

    performances = run_evaluation(
        scene,
        gs_args,
        phys_args,
        model_args,
        pipe_args,
        background,
        gt_positions,
        gt_params,
        train_views,
        test_views,
        experiment_dir=experiment_dir,
    )

    print_metrics_summary(performances)
