#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Optional

# Repo root, so defaults for files that ship in the repo resolve from any CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: Optional[ArgumentParser], name: str, fill_none=False):
        # If parser is None,
        # just use default parameters without adding to parser
        if parser is not None:
            group = parser.add_argument_group(name)
            for key, value in vars(self).items():
                shorthand = False
                if key.startswith("_"):
                    shorthand = True
                    key = key[1:]
                t = type(value)
                value = value if not fill_none else None
                if shorthand:
                    if t == bool:
                        group.add_argument(
                            "--" + key,
                            ("-" + key[0:1]),
                            default=value,
                            action="store_true",
                        )
                    else:
                        group.add_argument(
                            "--" + key, ("-" + key[0:1]), default=value, type=t
                        )
                else:
                    if t == bool:
                        group.add_argument(
                            "--" + key, default=value, action="store_true"
                        )
                    else:
                        group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._config_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        # No leading underscore: that would register a short flag from the first
        # letter, and -i/-f are taken or ambiguous.
        self.init_ply = ""  # frame-0 Gaussian PLY; see README
        # SEA-RAFT weights: a Hugging Face repo id (downloaded and cached on
        # first use) or a local .pth. The config ships with the SEA-RAFT
        # checkout; see the README's "Optical flow" section.
        self.flow_model_path = "MemorySlices/Tartan-C-T-TSKH-kitti432x960-M"
        self.flow_config_path = os.path.join(
            REPO_ROOT, "SEA-RAFT", "config", "eval", "spring-M.json"
        )
        self.data_device = "cuda"
        self.num_frame = -1
        self.eval = False
        self.res_scale = 1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if g.init_ply:  # abspath("") is the CWD, not "unset"
            g.init_ply = os.path.abspath(g.init_ply)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.position_lr_init = 0.0002
        self.feature_lr = 0.01  # 0.0025
        self.opacity_lr = 0.01
        self.scaling_lr = 0.0002
        self.rotation_lr = 0.0002
        self.scale_lr = 0.02  # Global scene scale learning rate
        self.lambda_dssim = 0.2
        self.adam_beta_1 = 0.8
        self.adam_beta_2 = 0.98

        # Inverse-optimization tunables (real defaults live here; the CLI registers
        # them with default=None via fill_none so an explicit flag overrides config).
        self.w_img = 0.0
        self.w_alp = 1.0
        self.w_flow = 1.0
        self.w_silhouette = 1.0
        self.sinkhorn_blur = 1.0  # silhouette Sinkhorn blur (pixel units); eps ~ blur^2
        self.flow_code = "Ff"  # optical-flow components: F/f forward, B/b backward
        self.w_distribution = 1.0  # particle distribution regularizer weight
        self.distribution_k = 3
        self.distribution_alpha_lower = 0.3  # repulsion below alpha_lower * dx
        self.distribution_alpha_upper = 0.8  # attraction above alpha_upper * dx
        self.mpm_iter_cnt = 50  # MPM substeps per frame (dt = frame_dt / this)
        # Optimization schedule, all in iterations. The rollout grows from
        # n_init_frames to n_frames by iteration full_rollout_at; MCMC particle
        # growth stops at the same iteration.
        self.n_iters = 250
        self.n_init_frames = 4
        self.full_rollout_at = 150
        self.center_inside_threshold = 0.95  # appearance-refine frame gate
        self.n_init_particles = 100_000  # particles sampled from reconstruction at init
        self.n_per_round_appearance_steps = 100
        self.n_post_appearance_steps = 5000
        self.mcmc_refine_every = 10
        self.mcmc_min_opacity = 0.005
        self.n_max_particles = -1  # MCMC growth cap (-1 = keep initial count)
        self.pos_grad_clamp = 1.0  # max per-particle pos-grad norm in backward
        self.debugsave_every = 0  # iterations between debug snapshots (0 = off)

        # Per-material-param fallback init LRs, used when a config's params[<name>]
        # entry omits "init_lr". Precedence: default here < config < explicit CLI.
        self.youngs_modulus_lr = 0.1
        self.poisson_ratio_lr = 0.1
        self.bulk_modulus_lr = 0.1
        self.shear_modulus_lr = 0.1
        self.yield_stress_lr = 0.1
        self.plastic_viscosity_lr = 0.05
        self.friction_angle_lr = 1.0
        self.vel_lr = 0.02  # initial-velocity learning rate

        super().__init__(parser, "Optimization Parameters", fill_none=True)


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    model_path = None
    gs_dict = None
    phys_dict = {}
    try:
        with open(args_cmdline.config_path, "r") as f:
            config_data = json.load(f)
            gs_dict = config_data.get("gs", None)
            phys_dict = config_data.get("physics", None) or {}
            # voxel_size is computed from particles via compute_optimal_dx
            # (see inv_problem.py); ignore any value the config tries to inject.
            if phys_dict.pop("voxel_size", None) is not None:
                print(
                    "[arguments] Ignoring 'voxel_size' from config; "
                    "dx is computed from particles via compute_optimal_dx."
                )
            model_path = gs_dict.get("model_path", None)

    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read config {args_cmdline.config_path}: {e}")

    if model_path is None:
        model_path = args_cmdline.model_path
    cfgfilepath = os.path.join(model_path, "cfg_args")
    print("Looking for config file in", cfgfilepath)
    try:
        with open(cfgfilepath) as cfg_file:
            print(f"Config file found: {cfgfilepath}")
            cfgfile_string = cfg_file.read()
    except FileNotFoundError:
        print(f"No cfg_args at {cfgfilepath}; using an empty Namespace.")
    args_cfgfile = eval(cfgfile_string)

    # Real defaults for the Optimization group. Its CLI args register with
    # default=None (fill_none), so args_cmdline never carries these defaults;
    # a non-None parsed value therefore means the user passed the flag explicitly.
    opt_defaults = vars(OptimizationParams(None))
    cli = {k: v for k, v in vars(args_cmdline).items() if v is not None}
    cli_opt = {k: v for k, v in cli.items() if k in opt_defaults}
    cli_other = {k: v for k, v in cli.items() if k not in opt_defaults}

    def apply_config(merged, section):
        for k, v in (section or {}).items():
            if v is not None:
                merged[k] = v

    # Precedence, low -> high: cfg_args < optimization defaults < model/pipeline/
    # per-run CLI < config file < explicit optimization-arg CLI.
    gs_merged = vars(args_cfgfile).copy()
    gs_merged.update(opt_defaults)
    gs_merged.update(cli_other)
    apply_config(gs_merged, gs_dict)
    gs_merged.update(cli_opt)

    # phys_args carries every gs key (superset for downstream phys_args.* access);
    # the physics section then overrides defaults, explicit CLI overrides both.
    phys_merged = dict(gs_merged)
    apply_config(phys_merged, phys_dict)
    phys_merged.update(cli_opt)

    return Namespace(**gs_merged), Namespace(**phys_merged)
