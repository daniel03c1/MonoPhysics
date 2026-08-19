"""
Optical flow estimation on top of an unmodified SEA-RAFT checkout.

SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow (ECCV 2024),
Princeton Vision Lab -- https://github.com/princeton-vl/SEA-RAFT

The checkout is used verbatim (see README): clone it into the repo root as
``SEA-RAFT/``; nothing inside it is patched. Its modules import their siblings
absolutely (``from utils.utils import ...``, ``from extractor import ...``), so
``SEA-RAFT/core`` has to go on ``sys.path`` -- which is also why this package is
called ``lib`` and not ``utils``: at runtime the top-level name ``utils``
belongs to SEA-RAFT.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEA_RAFT_ROOT = os.path.join(REPO_ROOT, "SEA-RAFT")
SEA_RAFT_CORE = os.path.join(SEA_RAFT_ROOT, "core")

if not os.path.isdir(SEA_RAFT_CORE):
    raise ImportError(
        f"SEA-RAFT checkout not found at {SEA_RAFT_ROOT}. Clone it in the repo "
        "root:\n\n    git clone https://github.com/princeton-vl/SEA-RAFT.git\n"
    )

if SEA_RAFT_CORE not in sys.path:
    sys.path.append(SEA_RAFT_CORE)

from raft import RAFT  # noqa: E402  (SEA-RAFT/core)
from utils.flow_viz import flow_to_image  # noqa: E402  (SEA-RAFT/core, not lib/)

PAD_DIVISOR = 8  # SEA-RAFT downsamples by 8, so both sides must be a multiple


def is_hf_repo_id(model_path):
    """True for "<org>/<name>" Hub ids, as opposed to a local checkpoint path."""
    parts = model_path.split("/")
    return (
        len(parts) == 2
        and all(parts)
        and not model_path.endswith(".pth")
        and not os.path.isabs(model_path)
    )


def json_to_args(json_path):
    """
    Read a SEA-RAFT eval config into the Namespace its model expects.

    Mirrors the checkout's ``config/parser.py``, which cannot be imported as
    ``config.parser`` because this repo has its own ``config/`` directory of
    JSON files that shadows it.
    """
    args = argparse.Namespace()
    args.__dict__.update(json.load(open(json_path)))
    return args


class FlowEstimator:
    """
    SEA-RAFT optical flow between two frames.

    Images are float tensors in [0, 1], either [3, H, W] or [1, 3, H, W]; the
    model wants [0, 255] and sides divisible by 8, both handled here and undone
    on the way out, so flows come back at the input resolution.
    """

    def __init__(self, config_path, model_path, device="cuda", iters=32):
        """
        Build the model from a SEA-RAFT eval config and load its weights.

        `model_path` is either a local checkpoint (.pth) or a Hugging Face repo
        id such as "MemorySlices/Tartan-C-T-TSKH-kitti432x960-M", which is
        downloaded and cached on first use. `iters` is the number of refinement
        iterations the model runs per pair.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.iters = iters
        self.config = json_to_args(config_path)

        if os.path.exists(model_path):
            print(f"[SEA-RAFT] loading weights from {model_path}")
            self.model = RAFT(self.config)
            checkpoint = torch.load(model_path, map_location=self.device)

            # Published checkpoints come in a few shapes: a bare state dict, or
            # one wrapped under "model"/"state_dict", optionally with the
            # "module." prefix a DataParallel save leaves behind.
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
        elif is_hf_repo_id(model_path):
            # SEA-RAFT publishes its weights on the Hub as safetensors, so they
            # go through the model's own PyTorchModelHubMixin rather than
            # torch.load. Cached under ~/.cache/huggingface after the first run.
            print(f"[SEA-RAFT] loading weights from Hugging Face: {model_path}")
            self.model = RAFT.from_pretrained(model_path, args=self.config)
        else:
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        self.model.to(self.device)
        self.model.eval()
        print(f"[SEA-RAFT] ready on {self.device}")

    def _preprocess_image(self, image):
        """
        Convert a [3, H, W] or [1, 3, H, W] image in [0, 1] into a padded
        [1, 3, H, W] batch in [0, 255], the range SEA-RAFT was trained on.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.float() * 255.0

        height, width = image.shape[-2:]
        pad_h = (PAD_DIVISOR - height % PAD_DIVISOR) % PAD_DIVISOR
        pad_w = (PAD_DIVISOR - width % PAD_DIVISOR) % PAD_DIVISOR
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        return image

    def _unpad(self, tensor, height, width):
        """Crop a [B, C, H, W] model output back to the input resolution."""
        return tensor[:, :, :height, :width]

    @torch.no_grad()
    def compute_flow(self, image1, image2, return_numpy=True, return_uncertainty=False):
        """
        Flow from image1 to image2, as (H, W, 2) numpy or a (2, H, W) tensor.

        With `return_uncertainty`, also returns the model's final per-pixel
        uncertainty, shaped to match.
        """
        height, width = image1.shape[-2:]
        img1 = self._preprocess_image(image1).to(self.device)
        img2 = self._preprocess_image(image2).to(self.device)

        result = self.model(img1, img2, iters=self.iters, test_mode=True)
        flow = self._unpad(result["final"], height, width)

        if return_numpy:
            flow = flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        else:
            flow = flow[0]  # (2, H, W)

        if return_uncertainty:
            # result["info"] holds one entry per refinement step; the last is
            # the most refined.
            uncertainty = self._unpad(result["info"][-1], height, width)
            uncertainty = (
                uncertainty[0].cpu().numpy() if return_numpy else uncertainty[0]
            )
            return flow, uncertainty

        return flow
