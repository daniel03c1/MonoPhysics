import math
import scipy
import torch
import torch.nn.functional as F
from math import exp

from knn import knn_idx

"""    Basic Primitives    """


def l1_loss(network_output, gt):
    return (network_output - gt).abs().mean()


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


"""    SSIM    """


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


SSIM_WINDOW_CACHE = {}


def ssim(img1, img2, window_size=11):
    channel = img1.size(-3)
    key = (window_size, channel, img1.dtype, img1.device)
    window = SSIM_WINDOW_CACHE.get(key)
    if window is None:
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        SSIM_WINDOW_CACHE[key] = window

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    # [3, H, W]
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


"""    Composite Image / Render Losses    """


def image_loss_value(image, gt_image, lambda_dssim):
    """
    RGB image loss on [3, H, W] images: lambda_dssim weights the SSIM term,
    (1 - lambda_dssim) the L1 term.
    """
    pixel_loss = l1_loss(image, gt_image)
    return (1.0 - lambda_dssim) * pixel_loss + lambda_dssim * (
        1.0 - ssim(image, gt_image)
    )


def alpha_loss_value(alpha, gt_alpha):
    """L1 alpha mask loss on [1, H, W] masks."""
    return l1_loss(alpha, gt_alpha)


"""    Optical Flow    """


def optical_flow_nll(pred_flow, gt_flow, uncertainty, var_min=0, var_max=10):
    assert uncertainty.shape[0] == 4

    weight = uncertainty[:2]  # [2, H, W]
    log_b = torch.stack(
        [
            uncertainty[2].clamp(min=0, max=var_max),
            uncertainty[3].clamp(min=var_min, max=0),
        ],
        dim=0,
    )  # [2, H, W]

    return -torch.logsumexp(
        (weight - math.log(2) - log_b).unsqueeze(0)
        - (gt_flow - pred_flow).abs().unsqueeze(1) * torch.exp(-log_b).unsqueeze(0),
        dim=1,
    )


def optical_flow_loss_value(pred_flow, gt_flow, uncertainty, gt_alpha_mask):
    """
    Flow loss for one frame pair: alpha-mask-weighted mean negative
    log-likelihood.
    """
    return (
        optical_flow_nll(pred_flow, gt_flow, uncertainty)
        * (gt_alpha_mask / gt_alpha_mask.sum())
    ).sum()


"""    3D / Geometry    """


def earth_movers_distance(x, y):
    x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
    y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
    cost_matrix = dis.detach().cpu().numpy()
    ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)

    return torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))


def knn_distribution_loss(xyz, k, r_lower, r_upper):
    """
    Two-sided KNN distance hinge: penalizes neighbor distances outside
    [r_lower, r_upper]. Gradients are mutual (center and neighbors both move).
    Returns (scalar loss, detached [N] k-th NN distance for diagnostics).
    """
    if not (r_lower < r_upper):
        raise ValueError(f"Need r_lower < r_upper, got {r_lower} >= {r_upper}")
    indices = knn_idx(xyz, k=k)  # [N, k]
    neighbors = xyz[indices]
    if neighbors.ndim == 2:
        neighbors = neighbors.unsqueeze(1)  # [N, k, 3]
    dist = (xyz[:, None, :] - neighbors).norm(dim=-1)  # [N, k]
    loss = (r_lower - dist).clamp(min=0).sum() + (dist - r_upper).clamp(min=0).sum()
    with torch.no_grad():
        dist_k = dist[:, k - 1]
    return loss, dist_k
