from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class PreprocessConfig:
    image_size: int = 256
    patch_power: int = 3
    region_quantile: float = 1.0 / 3.0


def texture_score(patch):
    r1, r2 = patch[:, 0:-1, :], patch[:, 1:, :]
    r3, r4 = patch[:, :, 0:-1], patch[:, :, 1:]
    r5, r6 = patch[:, 0:-1, 0:-1], patch[:, 1:, 1:]
    r7, r8 = patch[:, 0:-1, 1:], patch[:, 1:, 0:-1]
    return (
        torch.sum(torch.abs(r1 - r2))
        + torch.sum(torch.abs(r3 - r4))
        + torch.sum(torch.abs(r5 - r6))
        + torch.sum(torch.abs(r7 - r8))
    ).item()


def _random_crop(tensor, patch_size):
    _, height, width = tensor.shape
    if height < patch_size or width < patch_size:
        tensor = TF.resize(tensor, [max(height, patch_size), max(width, patch_size)])
        _, height, width = tensor.shape
    top = torch.randint(0, height - patch_size + 1, (1,)).item()
    left = torch.randint(0, width - patch_size + 1, (1,)).item()
    return tensor[:, top:top + patch_size, left:left + patch_size]


def _reconstruct(patches, image_size, patch_size, num_block):
    out = torch.zeros(3, image_size, image_size, dtype=patches[0].dtype)
    for index, patch in enumerate(patches):
        row, col = divmod(index, num_block)
        out[
            :,
            row * patch_size:(row + 1) * patch_size,
            col * patch_size:(col + 1) * patch_size,
        ] = patch
    return out


def preprocess_image(image, config=None):
    """Create the two-view AiDet_FFT input tensor from a PIL image.

    Returns a tensor with shape ``(2, 3, image_size, image_size)``.
    """
    config = config or PreprocessConfig()
    if not 0.0 < config.region_quantile <= 0.5:
        raise ValueError("region_quantile must be in (0, 0.5]")

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    min_side = min(image.size)
    num_block = 2 ** config.patch_power
    patch_size = config.image_size // num_block
    if min_side < patch_size:
        image = TF.resize(image, [patch_size, patch_size])

    tensor = TF.to_tensor(image)
    patch_count = num_block * num_block
    candidate_count = max(2 * patch_count, int(round(patch_count / config.region_quantile)))
    candidates = [_random_crop(tensor, patch_size) for _ in range(candidate_count)]
    candidates = sorted(candidates, key=texture_score)

    poor = _reconstruct(candidates[:patch_count], config.image_size, patch_size, num_block)
    rich = _reconstruct(candidates[-patch_count:], config.image_size, patch_size, num_block)
    stacked = torch.stack([poor, rich], dim=0)
    return (stacked - IMAGENET_MEAN) / IMAGENET_STD
