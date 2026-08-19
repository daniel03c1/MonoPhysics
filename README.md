# MonoPhysics: Estimating Geometry, Appearance, and Physical Parameters from Monocular Videos

Daniel Rho, Jun Myeong Choi, Matthew Thornton, Biswadip Dey, Roni Sengupta

[Project page](https://daniel03c1.github.io/MonoPhysics) · [arXiv](https://arxiv.org/abs/2605.30320)

Existing inverse physics methods recover physical parameters from multi-view video, where geometric constraints across views resolve scale and 3D structure.
Monocular video offers no such constraints, which leaves severe scale ambiguity, inaccurate geometry, and weak coupling between appearance optimization and physical simulation.
MonoPhysics bridges vision and physics in three ways: global scale alignment, physics-aware geometry refinement, and a differentiable position map.

## Contents

- [Setup](#setup): conda environment and dependencies.
- [Running](#running): training a scene and reading the outputs.
- [Data](#data): the three supported dataset layouts.
- [Initial Gaussians](#initial-gaussians): the first-frame reconstruction every run starts from, and how to produce it.
- [Citation](#citation) · [Acknowledgements](#acknowledgements) · [License](#license)

## Setup

```bash
# Python 3.10. Taichi 1.4.0 ships no wheels for 3.11.
conda create -n monophysics python=3.10
conda activate monophysics

conda install nvidia::cuda-toolkit==12.8.0

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 --no-build-isolation

pip install tqdm matplotlib trimesh opencv-python plyfile einops scipy open3d
pip install lpips geomloss pykeops
pip install taichi==1.4.0
pip install huggingface_hub safetensors

pip install submodules/knn --no-build-isolation
pip install submodules/diff_pixel_coords --no-build-isolation

# Image to 3D, used only to build the initial Gaussians.
# Set up its env and checkpoints as that repo describes.
git clone https://github.com/facebookresearch/sam-3d-objects

# Optical flow, used as it is. Weights download themselves on first run.
git clone https://github.com/princeton-vl/SEA-RAFT.git
```

## Running

```bash
python inv_problem.py \
  -c <config>.json \
  -s <dataset>/<scene> \
  -m <output_dir> \
  --init_ply <scene>/<gaussians_cam>.ply \
  --cam_idx <camera>
```

For example:

```bash
python inv_problem.py \
  -c config/vid2sim.json \
  -s data/Vid2Sim/bell \
  -m output/bell \
  --init_ply init_ply/bell/cam02/gaussians_cam.ply \
  --cam_idx 2
```

- `-c / --config_path`: JSON config. See `config/` for per-dataset examples.
- `-s / --source_path`: dataset root for one scene (images, cameras, masks).
- `-m / --model_path`: output directory.
- `--init_ply`: a 3D Gaussian Splatting reconstruction of the scene's first frame, in the reference camera's frame.
- `--cam_idx`: the camera to optimize against. This is a monocular framework, so exactly one camera trains and every other view is held out for evaluation.
- `--postfix`: optional label for the run directory, so several runs can share one `-m` without overwriting each other. `--postfix lr01` writes to `run_lr01/`, and omitting it writes to `run/`.

Results land in `<model_path>/run[_<postfix>]/`: `predictions.json` (estimated material parameters), `performance.json` (metrics), `gs.ply`, and rendered frames under `images/`.
A run whose `gs.ply` and `predictions.json` already exist skips training and only re-evaluates.
Delete `gs.ply` to retrain.


## Data


- **[Vid2Sim](https://github.com/CzzzzH/Vid2Sim)**: download from its authors.
- **[Spring-Gaus](https://github.com/Colmar-zlicheng/Spring-Gaus)** (real capture): download from its authors.
- **[Our Dataset](https://drive.google.com/drive/folders/1CWgiC5zZxk5lKad5dW7rlVZyyosm9rOg?usp=drive_link)**: elastic and plastic objects, released with the paper. The code and configs call it `jukebox`, its working name.

Our dataset is laid out per scene as:

```
<root>/<scene>/
  camera.json               # n_cameras, width, height, convention, cameras[]
  images/<cam:03d>_<frame:03d>.png
  masks/<cam:03d>_<frame:03d>.png
  points3d.ply
```

Readers for all three layouts live in `scene/dataset_readers.py` and `make_init_ply.py`.
For anything else, reconstruct the first-frame PLY with `make_init_ply.py`'s `--image`/`--camera` path and add a reader there.

## Initial Gaussians

The pipeline does not reconstruct the object itself.
It starts from a **3D Gaussian Splatting reconstruction of the first frame**, supplied as a PLY via `--init_ply`, which everything downstream inherits as-is.
`inv_problem.py` accepts any such reconstruction, whatever produced it, as long as it covers only the first frame and is expressed in the reference camera's OpenCV frame.
`make_init_ply.py` is a helper that obtains one with SAM 3D and aligns it to the camera.

The PLY must be:

- the **first frame only**, a single static reconstruction rather than a sequence
- expressed in the **reference camera's OpenCV frame**, not world coordinates

The *reference camera* is the one given by `--cam_idx`, so `--cam_idx 2` needs the PLY in camera 2's frame.
Startup prints which uid it used, as `[Gaussians] Loading <path> (reference camera uid N)`.
Check that it matches the camera your PLY was built for.

The PLY is then cached at `<model_path>/aligned_gaussians/frame_000.ply`.
Delete that file to swap in a new reconstruction.

### Producing the PLY

`make_init_ply.py` runs [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) on one masked image, then aligns the result to that camera.

SAM 3D needs its own environment and checkpoints, so follow its repository for both.
`make_init_ply.py` runs inside *that* environment, reaches SAM 3D through its public `Inference` API, and never modifies the checkout:

```bash
conda run -n <sam3d-env> python make_init_ply.py --sam3d-repo /path/to/sam-3d-objects \
  --dataset vid2sim --root /path/to/Vid2Sim --object bell --cam 2 \
  --out init_ply/bell/cam02
```

For data outside the built-in adapters, pass an image and a camera directly:

```bash
python make_init_ply.py --sam3d-repo /path/to/sam-3d-objects \
  --image frame000.png --camera cam.json --out init_ply/my_object/cam00
```

`--image` is RGBA with alpha as the object mask, or RGB plus `--mask`.
`cam.json` holds the intrinsics as either `K` (3×3) or `camera_angle_x` (horizontal FOV in radians, NeRF style):

```json
{"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}
```

Everything happens in the reference camera's frame, so no extrinsics are needed.
Built-in adapters are `vid2sim`, `jukebox`, and `springgaus_real`.

The run writes `gaussians_cam.ply` into `--out`, the file to pass to `--init_ply`, alongside `compare.png` and the unaligned `gaussians_cam_raw.ply` / `compare_raw.png`.
**Check `compare.png` first**: the red silhouette should cover the green mask.
Nothing downstream re-aligns the PLY, so a misalignment here propagates.


## Citation

```bibtex
@article{rho2026monophysics,
  title={MonoPhysics: Estimating Geometry, Appearance, and Physical Parameters from Monocular Videos},
  author={Rho, Daniel and Choi, Jun Myeong and Thornton, Matthew and Dey, Biswadip and Sengupta, Roni},
  journal={arXiv preprint arXiv:2605.30320},
  year={2026}
}
```

## Acknowledgements

We sincerely thank the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [PAC-NeRF](https://github.com/xuan-li/PAC-NeRF), [Gaussian-Informed Continuum](https://github.com/Jukgei/gic) and [Spring-Gaus](https://github.com/Colmar-zlicheng/Spring-Gaus), whose code and datasets were used in this work.
This repository also builds on our earlier project, [ProJo4D](https://daniel03c1.github.io/ProJo4D/).

## License
This code is released under the [Gaussian-Splatting License](LICENSE.md), for non-commercial research and evaluation use only, since it builds on 3D Gaussian Splatting.
