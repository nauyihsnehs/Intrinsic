import argparse
import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image

from intrinsic import load_models, run_pipeline


def load_image(path_or_url):
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        with urlopen(path_or_url) as response:
            data = response.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_outputs(hr_alb, output_path, output_image=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, hr_alb)
    if output_image:
        img = (np.clip(hr_alb, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)
        Image.fromarray(img).save(output_image)


def main():
    parser = argparse.ArgumentParser(description="Predict hr_alb using the intrinsic pipeline.")
    parser.add_argument("image", help="Path or URL to an RGB image.")
    parser.add_argument("--weights", default="v2.1", choices=["v2", "v2.1"], help="Weights to load.")
    parser.add_argument("--device", default=None, help="Torch device (default: auto).")
    parser.add_argument("--output", default="hr_alb.npy", help="Output .npy file for hr_alb.")
    parser.add_argument("--output-image", default=None, help="Optional output PNG for hr_alb visualization.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(args.image)

    models = load_models(args.weights, device=device)
    result = run_pipeline(models, img, device=device)

    save_outputs(result["hr_alb"], args.output, args.output_image)


if __name__ == "__main__":
    main()
