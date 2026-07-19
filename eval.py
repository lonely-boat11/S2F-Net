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
    parser = argparse.ArgumentParser(description="Evaluate AiDet_FFT on a labeled 0_real/1_fake dataset.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing 0_real/ and 1_fake/ folders.")
    parser.add_argument("--weights", default="weights/AiDet_FFT_8.pth")
    parser.add_argument("--name", default="AiDet_FFT_clean", help="Model name printed in the result table.")
    parser.add_argument("--output", default="outputs/eval_predictions.csv", help="Per-image CSV path.")
    parser.add_argument("--summary", default="outputs/eval_summary.csv", help="Summary CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-power", type=int, default=3)
    parser.add_argument("--region-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


def collect_labeled_images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    samples = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image_path.parent.name == "0_real":
            samples.append((image_path, 0.0))
        elif image_path.parent.name == "1_fake":
            samples.append((image_path, 1.0))

    if not samples:
        raise FileNotFoundError(f"No images found under 0_real/ or 1_fake/ in {root}")
    return samples


def discover_testsets(root):
    """Return named evaluation sets under ``root``.

    If ``root`` itself contains labeled images, evaluate it as one set.
    Otherwise each direct child directory that contains any nested 0_real/1_fake
    images is treated as one test set, matching common paper benchmark layouts:
    ``testdata/progan/.../0_real`` and ``testdata/stylegan/.../1_fake``.
    """
    root = Path(root)
    direct_real = root / "0_real"
    direct_fake = root / "1_fake"
    if direct_real.is_dir() or direct_fake.is_dir():
        return [(root.name, root)]

    testsets = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        has_labeled_images = any(
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.parent.name in {"0_real", "1_fake"}
            for path in child.rglob("*")
        )
        if has_labeled_images:
            testsets.append((child.name, child))

    if not testsets:
        raise FileNotFoundError(f"No test sets with 0_real/ or 1_fake/ found under {root}")
    return testsets


def binary_accuracy(labels, scores, threshold=0.5):
    correct = sum(int((score >= threshold) == bool(label)) for label, score in zip(labels, scores))
    return correct / max(1, len(labels))


def average_precision(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label in pairs if label == 1.0)
    if positives == 0:
        return 0.0
    hit = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1.0:
            hit += 1
            precision_sum += hit / rank
    return precision_sum / positives


def compute_metrics(labels, scores, threshold):
    real_labels = [label for label in labels if label == 0.0]
    real_scores = [score for label, score in zip(labels, scores) if label == 0.0]
    fake_labels = [label for label in labels if label == 1.0]
    fake_scores = [score for label, score in zip(labels, scores) if label == 1.0]
    return {
        "accuracy": binary_accuracy(labels, scores, threshold),
        "avg precision": average_precision(labels, scores),
        "r_acc": binary_accuracy(real_labels, real_scores, threshold) if real_labels else 0.0,
        "f_acc": binary_accuracy(fake_labels, fake_scores, threshold) if fake_labels else 0.0,
        "num_images": len(labels),
        "num_real": len(real_labels),
        "num_fake": len(fake_labels),
    }


@torch.no_grad()
def predict_one(model, image_path, config, device, seed):
    set_random_seed(seed)
    with Image.open(image_path) as image:
        tensor = preprocess_image(image, config).unsqueeze(0).to(device)
    return model(tensor).sigmoid().item()


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

    prediction_rows = []
    summary_rows = []
    for testset_name, testset_root in discover_testsets(args.data_root):
        samples = collect_labeled_images(testset_root)
        labels = []
        scores = []
        for image_path, label in progress(samples, desc=testset_name):
            probability = predict_one(model, image_path, config, device, args.seed)
            prediction = 1.0 if probability >= args.threshold else 0.0
            prediction_rows.append({
                "testset": testset_name,
                "image": str(image_path),
                "label": "fake" if label == 1.0 else "real",
                "fake_probability": probability,
                "prediction": "fake" if prediction == 1.0 else "real",
                "correct": int(prediction == label),
            })
            labels.append(label)
            scores.append(probability)
        metrics = compute_metrics(labels, scores, args.threshold)
        summary_rows.append({"testset": testset_name, **metrics})

    metric_names = ["accuracy", "avg precision", "r_acc", "f_acc"]
    avg_row = {
        "testset": "avg",
        **{
            name: sum(row[name] for row in summary_rows) / len(summary_rows)
            for name in metric_names
        },
        "num_images": sum(row["num_images"] for row in summary_rows),
        "num_real": sum(row["num_real"] for row in summary_rows),
        "num_fake": sum(row["num_fake"] for row in summary_rows),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["testset", "image", "label", "fake_probability", "prediction", "correct"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([f"{args.name} model testing on..."])
        writer.writerow(["testset", "accuracy", "avg precision", "r_acc", "f_acc"])
        for row in summary_rows + [avg_row]:
            writer.writerow([row["testset"], row["accuracy"], row["avg precision"], row["r_acc"], row["f_acc"]])

    print(f"{args.name} model testing on...")
    print("testset,accuracy,avg precision,r_acc,f_acc")
    for row in summary_rows + [avg_row]:
        print(f"{row['testset']},{row['accuracy']},{row['avg precision']},{row['r_acc']},{row['f_acc']}")
    print(f"Saved per-image results to {output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
