import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from aigcdetect.model import AiDetFFT
from aigcdetect.preprocess import PreprocessConfig, preprocess_image
from aigcdetect.utils import load_checkpoint, set_random_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict whether images are AI-generated.")
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--weights", default="weights/AiDet_FFT_8.pth")
    parser.add_argument("--output", default=None, help="Optional CSV path for batch prediction.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-power", type=int, default=3)
    parser.add_argument("--region-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-each", action="store_true", help="Print every image prediction to the terminal.")
    return parser.parse_args()


def collect_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(path)


def progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


@torch.no_grad()
def predict_one(model, image_path, config, device, seed):
    set_random_seed(seed)
    with Image.open(image_path) as image:
        tensor = preprocess_image(image, config).unsqueeze(0).to(device)
    probability = model(tensor).sigmoid().item()
    return probability


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = PreprocessConfig(
        image_size=args.image_size,
        patch_power=args.patch_power,
        region_quantile=args.region_quantile,
    )

    model = AiDetFFT().to(device)
    load_checkpoint(model, args.weights, map_location=device)
    model.eval()

    rows = []
    for image_path in progress(collect_images(args.input), desc="predict"):
        probability = predict_one(model, image_path, config, device, args.seed)
        label = "fake" if probability >= args.threshold else "real"
        rows.append({
            "image": str(image_path),
            "fake_probability": probability,
            "prediction": label,
        })
        if args.print_each:
            message = f"{image_path}\tfake_probability={probability:.6f}\tprediction={label}"
            if tqdm is not None:
                tqdm.write(message)
            else:
                print(message)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "fake_probability", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} predictions to {output}")
    else:
        print(f"Finished {len(rows)} predictions.")


if __name__ == "__main__":
    main()
