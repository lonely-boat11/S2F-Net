import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .preprocess import PreprocessConfig, preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(path):
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(
        str(item)
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_paths(root):
    """Load ``0_real`` and ``1_fake`` images from direct or class-nested layout."""
    root = Path(root)
    real_dir = root / "0_real"
    fake_dir = root / "1_fake"
    real_paths = _list_images(real_dir)
    fake_paths = _list_images(fake_dir)

    if not real_paths and not fake_paths:
        for class_dir in sorted(root.iterdir() if root.is_dir() else [], key=lambda p: p.name):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            real_paths.extend(_list_images(class_dir / "0_real"))
            fake_paths.extend(_list_images(class_dir / "1_fake"))

    if not real_paths:
        raise FileNotFoundError(f"No real images found under {os.fspath(root)}")
    if not fake_paths:
        raise FileNotFoundError(f"No fake images found under {os.fspath(root)}")
    return real_paths, fake_paths


class AiDetDataset(Dataset):
    def __init__(self, root, image_size=256, patch_power=3, region_quantile=1.0 / 3.0):
        real_paths, fake_paths = load_image_paths(root)
        self.paths = real_paths + fake_paths
        self.labels = [0.0] * len(real_paths) + [1.0] * len(fake_paths)
        self.config = PreprocessConfig(
            image_size=image_size,
            patch_power=patch_power,
            region_quantile=region_quantile,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = preprocess_image(image, self.config)
        return tensor, self.labels[index]
