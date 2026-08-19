import json
import numpy as np
import os
import torch


def save_experiment_configs(
    experiment_dir, gs_args, phys_args, model_args, pipe_args, opt_args
):
    """Save all experiment configurations to <experiment_dir>/args.json."""

    def _convert_to_serializable(obj):
        """Recursively convert obj to JSON-serializable format."""
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [_convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _convert_to_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return _convert_to_serializable(vars(obj))
        else:
            return str(obj)

    all_args = {
        "gs_args": _convert_to_serializable(gs_args),
        "phys_args": _convert_to_serializable(phys_args),
        "model_args": _convert_to_serializable(model_args),
        "pipe_args": _convert_to_serializable(pipe_args),
        "opt_args": _convert_to_serializable(opt_args),
    }

    output_path = os.path.join(experiment_dir, "args.json")
    with open(output_path, "w") as f:
        json.dump(all_args, f, indent=4)
    print(f"[Config] Saved experiment arguments to {output_path}")
