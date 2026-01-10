import math

import numpy as np
import torch
from skimage.transform import resize


def round_32(x):
    return 32 * math.ceil(x / 32)


def invert(x):
    return 1.0 / (x + 1.0)


def uninvert(x, eps=0.001, clip=True):
    if clip:
        x = x.clip(eps, 1.0)
    return (1.0 / x) - 1.0


def get_brightness(rgb, mode='numpy', keep_dim=True):
    if mode == 'torch' or torch.is_tensor(rgb):
        brightness = (0.3 * rgb[0, :, :]) + (0.59 * rgb[1, :, :]) + (0.11 * rgb[2, :, :])
        if keep_dim:
            brightness = brightness.unsqueeze(0)
        return brightness
    brightness = (0.3 * rgb[:, :, 0]) + (0.59 * rgb[:, :, 1]) + (0.11 * rgb[:, :, 2])
    if keep_dim:
        brightness = brightness[:, :, np.newaxis]
    return brightness


def to2np(img):
    return img.detach().cpu().permute(1, 2, 0).numpy()


def batch_rgb2iuv(rgb, eps=0.001):
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]
    l = (r * 0.299) + (g * 0.587) + (b * 0.114)
    i = invert(l)
    u = invert(r / (g + eps))
    v = invert(b / (g + eps))
    return torch.stack((i, u, v), axis=1)


def batch_iuv2rgb(iuv, eps=0.001):
    l = uninvert(iuv[:, 0, :, :], eps=eps)
    u = uninvert(iuv[:, 1, :, :], eps=eps)
    v = uninvert(iuv[:, 2, :, :], eps=eps)
    g = l / ((u * 0.299) + (v * 0.114) + 0.587)
    r = g * u
    b = g * v
    return torch.stack((r, g, b), axis=1)


def optimal_resize(img, conf=0.01):
    if conf is None or conf <= 0:
        h, w = img.shape[:2]
        return resize(img, (round_32(h), round_32(w)), anti_aliasing=True)
    h, w, _ = img.shape
    max_dim = max(h, w)
    target = min(max_dim * (1.0 + conf), 1500)
    scale = target / max_dim
    return resize(img, (round_32(h * scale), round_32(w * scale)), anti_aliasing=True)
