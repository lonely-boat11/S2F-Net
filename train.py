import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from aigcdetect.data import AiDetDataset
from aigcdetect.model import AiDetFFT
from aigcdetect.utils import load_checkpoint, save_checkpoint, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train AiDet_FFT for AIGC image detection.")
    parser.add_argument("--train-root", required=True, help="Folder containing 0_real/ and 1_fake/.")
    parser.add_argument("--val-root", default=None, help="Optional validation folder.")
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint to resume/fine-tune from.")
    parser.add_argument("--output-dir", default="checkpoints/AiDet_FFT", help="Where checkpoints are saved.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-power", type=int, default=3)
    parser.add_argument("--region-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


def binary_accuracy(y_true, y_score, threshold=0.5):
    correct = sum(int((score >= threshold) == bool(label)) for label, score in zip(y_true, y_score))
    return correct / max(1, len(y_true))


def average_precision(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    for images, labels in progress(loader, desc="val", leave=False):
        images = images.to(device)
        scores = model(images).sigmoid().flatten().cpu()
        y_score.extend(scores.tolist())
        y_true.extend(labels.tolist())
    return {
        "acc": binary_accuracy(y_true, y_score),
        "ap": average_precision(y_true, y_score),
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)

    train_set = AiDetDataset(
        args.train_root,
        image_size=args.image_size,
        patch_power=args.patch_power,
        region_quantile=args.region_quantile,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if args.val_root:
        val_set = AiDetDataset(
            args.val_root,
            image_size=args.image_size,
            patch_power=args.patch_power,
            region_quantile=args.region_quantile,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = AiDetFFT().to(device)
    if args.pretrained:
        load_checkpoint(model, args.pretrained, map_location=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_progress = progress(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.float().to(device)
            logits = model(images).flatten()
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if tqdm is not None:
                train_progress.set_postfix(loss=f"{loss.item():.4f}")

        save_checkpoint(output_dir / f"model_epoch_{epoch}.pth", model, optimizer, epoch)
        message = f"epoch {epoch}: train_loss={running_loss / max(1, len(train_loader)):.4f}"

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            message += f", val_acc={metrics['acc']:.4f}, val_ap={metrics['ap']:.4f}"
            if metrics["ap"] > best_ap:
                best_ap = metrics["ap"]
                save_checkpoint(output_dir / "model_best.pth", model, optimizer, epoch, best_metric=best_ap)
        else:
            save_checkpoint(output_dir / "model_latest.pth", model, optimizer, epoch)
        print(message)


if __name__ == "__main__":
    main()
