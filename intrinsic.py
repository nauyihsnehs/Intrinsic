import math

import numpy as np
import torch
from skimage.transform import resize

from altered_midas import MidasNet, MidasNet_small

V1_DICT = {
    'paper_weights': 'https://github.com/compphoto/Intrinsic/releases/download/v1.0/final_weights.pt',
    'rendered_only': 'https://github.com/compphoto/Intrinsic/releases/download/v1.0/rendered_only_weights.pt',
}


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


def load_decompile(path):
    compiled_dict = torch.load(path)
    remove_prefix = '_orig_mod.'
    return {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in compiled_dict.items()}


def load_models(path, device='cuda', compiled=False, alb_residual=False):
    models = {}
    load_func = load_decompile if compiled else torch.load

    if isinstance(path, str):
        if path in ['paper_weights', 'rendered_only']:
            combined_dict = torch.hub.load_state_dict_from_url(V1_DICT[path], map_location=device, progress=True)
            ord_state_dict = combined_dict['ord_state_dict']
            iid_state_dict = combined_dict['iid_state_dict']
            col_state_dict = None
            alb_state_dict = None
        elif path == 'v2':
            base_url = 'https://github.com/compphoto/Intrinsic/releases/download/v2.0/'
            ord_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_0.pt', map_location=device, progress=True)
            iid_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_1.pt', map_location=device, progress=True)
            col_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_2.pt', map_location=device, progress=True)
            alb_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_3.pt', map_location=device, progress=True)
        elif path == 'v2.1':
            base_url = 'https://github.com/compphoto/Intrinsic/releases/download/v2.1/'
            ord_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_0_v21.pt', map_location=device, progress=True)
            iid_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_1_v21.pt', map_location=device, progress=True)
            col_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_2_v21.pt', map_location=device, progress=True)
            alb_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_3_v21.pt', map_location=device, progress=True)
            alb_residual = True
        else:
            raise ValueError("Unknown weights option.")
    elif isinstance(path, list):
        ord_state_dict = load_func(path[0])
        iid_state_dict = load_func(path[1])
        col_state_dict = load_func(path[2])
        alb_state_dict = load_func(path[3])
    else:
        raise ValueError("path must be a string or list of weight paths.")

    ord_model = MidasNet()
    ord_model.load_state_dict(ord_state_dict)
    ord_model.eval()
    models['ord_model'] = ord_model.to(device)

    iid_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    iid_model.load_state_dict(iid_state_dict)
    iid_model.eval()
    models['iid_model'] = iid_model.to(device)

    if col_state_dict is not None:
        col_model = MidasNet(activation='sigmoid', input_channels=7, output_channels=2)
        col_model.load_state_dict(col_state_dict)
        col_model.eval()
        models['col_model'] = col_model.to(device)

    if alb_state_dict is not None:
        alb_model = MidasNet(activation='sigmoid', input_channels=9, output_channels=3, last_residual=alb_residual)
        alb_model.load_state_dict(alb_state_dict)
        alb_model.eval()
        models['alb_model'] = alb_model.to(device)

    return models


def base_resize(img, base_size=384):
    h, w, _ = img.shape
    max_dim = max(h, w)
    scale = base_size / max_dim
    new_h, new_w = round_32(h * scale), round_32(w * scale)
    return resize(img, (new_h, new_w, 3), anti_aliasing=True)


def equalize_predictions(img, base, full, p=0.5):
    h, w, _ = img.shape
    full_shd = (1.0 / full.clip(1e-5)) - 1.0
    base_shd = (1.0 / base.clip(1e-5)) - 1.0
    full_alb = get_brightness(img) / full_shd.clip(1e-5)
    base_alb = get_brightness(img) / base_shd.clip(1e-5)
    rand_msk = (np.random.randn(h, w) > p).astype(np.uint8)
    flat_full_alb = full_alb[rand_msk == 1]
    flat_base_alb = base_alb[rand_msk == 1]
    scale, _, _, _ = np.linalg.lstsq(flat_full_alb.reshape(-1, 1), flat_base_alb, rcond=None)
    new_full_alb = scale * full_alb
    new_full_shd = get_brightness(img) / new_full_alb.clip(1e-5)
    new_full = 1.0 / (1.0 + new_full_shd)
    return base, new_full


def run_gray_pipeline(
        models,
        img_arr,
        resize_conf=None,
        base_size=384,
        linear=False,
        device='cuda',
        lstsq_p=0.0,
        inputs='all',
):
    orig_h, orig_w, _ = img_arr.shape

    if resize_conf is None:
        img_arr = resize(img_arr, (round_32(orig_h), round_32(orig_w)), anti_aliasing=True)
    elif isinstance(resize_conf, int):
        scale = resize_conf / max(orig_h, orig_w)
        img_arr = resize(
            img_arr,
            (round_32(orig_h * scale), round_32(orig_w * scale)),
            anti_aliasing=True,
        )
    elif isinstance(resize_conf, float):
        img_arr = optimal_resize(img_arr, conf=resize_conf)

    fh, fw, _ = img_arr.shape
    lin_img = img_arr ** 2.2 if not linear else img_arr

    with torch.no_grad():
        base_input = base_resize(lin_img, base_size)
        full_input = lin_img

        base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(device).float()
        full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(device).float()

        base_out = models['ord_model'](base_input.unsqueeze(0)).squeeze(0)
        full_out = models['ord_model'](full_input.unsqueeze(0)).squeeze(0)

        base_out = base_out.permute(1, 2, 0).cpu().numpy()
        full_out = full_out.permute(1, 2, 0).cpu().numpy()
        base_out = resize(base_out, (fh, fw))

        if inputs == 'all':
            ord_base, ord_full = equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)
        else:
            ord_base, ord_full = base_out, full_out

        inp = torch.from_numpy(lin_img).permute(2, 0, 1).to(device)
        bse = torch.from_numpy(ord_base).permute(2, 0, 1).to(device)
        fll = torch.from_numpy(ord_full).permute(2, 0, 1).to(device)

        if inputs == 'full':
            combined = torch.cat((inp, fll), 0).unsqueeze(0)
        elif inputs == 'base':
            combined = torch.cat((inp, bse), 0).unsqueeze(0)
        elif inputs == 'rgb':
            combined = inp.unsqueeze(0)
        else:
            combined = torch.cat((inp, bse, fll), 0).unsqueeze(0)

        inv_shd = models['iid_model'](combined).squeeze(1)
        shd = uninvert(inv_shd)
        alb = inp / shd

    inv_shd = inv_shd.squeeze(0).detach().cpu().numpy()
    alb = alb.permute(1, 2, 0).detach().cpu().numpy()

    return {
        'lin_img': lin_img,
        'gry_shd': inv_shd,
        'gry_alb': alb,
    }


def run_pipeline(models, img_arr, resize_conf=None, base_size=384, linear=False, device='cuda'):
    results = run_gray_pipeline(
        models,
        img_arr,
        resize_conf=resize_conf,
        linear=linear,
        device=device,
        base_size=base_size,
    )

    img = results['lin_img']
    gry_shd = results['gry_shd'][:, :, None]
    gry_alb = results['gry_alb']

    net_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    net_shd = torch.from_numpy(gry_shd).permute(2, 0, 1).unsqueeze(0).to(device)
    net_alb = torch.from_numpy(gry_alb).permute(2, 0, 1).unsqueeze(0).to(device)

    in_img_luv = batch_rgb2iuv(net_img)
    in_alb_luv = batch_rgb2iuv(net_alb)

    orig_sz = img.shape[:2]
    scale = base_size / max(orig_sz)
    base_sz = (round_32(orig_sz[0] * scale), round_32(orig_sz[1] * scale))

    in_img_luv = torch.nn.functional.interpolate(in_img_luv, size=base_sz, mode='bilinear', align_corners=True, antialias=True)
    in_alb_luv = torch.nn.functional.interpolate(in_alb_luv, size=base_sz, mode='bilinear', align_corners=True, antialias=True)
    in_gry_shd = torch.nn.functional.interpolate(net_shd, size=base_sz, mode='bilinear', align_corners=True, antialias=True)

    inp = torch.cat([in_img_luv, in_gry_shd, in_alb_luv], 1)

    with torch.no_grad():
        uv_shd = models['col_model'](inp)

    uv_shd = torch.nn.functional.interpolate(uv_shd, size=orig_sz, mode='bilinear', align_corners=True)

    iuv_shd = torch.cat((net_shd, uv_shd), 1)
    rough_shd = batch_iuv2rgb(iuv_shd)
    rough_alb = net_img / rough_shd

    rough_alb *= 0.75 / torch.quantile(rough_alb, 0.99)
    rough_alb = rough_alb.clip(0.001)
    rough_shd = net_img / rough_alb

    inp = torch.cat([net_img, invert(rough_shd), rough_alb], 1)
    with torch.no_grad():
        pred_alb = models['alb_model'](inp)

    hr_alb = pred_alb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return {'hr_alb': hr_alb}
