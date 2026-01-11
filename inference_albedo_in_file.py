import torch
from inference_albedo_single_file import run_pipeline


def infer_hr_alb_batch(models, batch_tensor, base_size=384):
    device = batch_tensor.device
    outputs = []
    with torch.no_grad():
        for item in batch_tensor:
            img = item.detach().permute(1, 2, 0).cpu().numpy()
            pred = run_pipeline(models, img, base_size=base_size, device=device)["hr_alb"]
            outputs.append(torch.from_numpy(pred).permute(2, 0, 1).to(device))
    return torch.stack(outputs, dim=0)
