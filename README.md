# S2F-Net

This is the clean release code for the paper model **S2F-Net: A Robust Spatial-Spectrum Fusion Framework for Cross-Model
AIGC Detection**, an AI-generated image detector.
Only the authors' model, training pipeline, preprocessing, and prediction code are included.

## Files

```text
  aigcdetect/
    model.py              # AiDet_FFT network
    preprocess.py         # Smash-and-reconstruct preprocessing
    data.py               # 0_real / 1_fake dataset loader
    utils.py              # checkpoint and seed helpers
    srm_filter_kernel.py  # SRM high-pass filters
  weights/
    AiDet_FFT_8.pth       # released checkpoint
  train.py
  predict.py
  requirements.txt
```

## Installation

Install PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Use the CPU build of PyTorch if CUDA is unavailable.

## Dataset Layout

Training and validation folders should contain real and generated images:

```text
dataset/
  0_real/
    image1.png
  1_fake/
    image2.png
```

Class-nested layout is also supported:

```text
dataset/
  class_name/
    0_real/
    1_fake/
```

## Prediction

Single image:

```bash
python predict.py --input path/to/image.png --weights weights/AiDet_FFT_8.pth
```

Folder prediction with CSV output:

```bash
python predict.py --input path/to/images --output outputs/predictions.csv
```

The output probability is the probability that the image is AI-generated.

## Evaluation

For a labeled test set containing `0_real` and `1_fake` folders, run:

```bash
python eval.py \
  --data-root path/to/testdata \
  --weights weights/AiDet_FFT_8.pth \
  --name AiDet_FFT_clean \
  --output outputs/eval_predictions.csv \
  --summary outputs/eval_summary.csv
```

The evaluator recursively finds images whose parent folder is `0_real` or
`1_fake`, so nested layouts such as `testdata/progan/airplane/0_real/*.png`
are supported. It reports a paper-style table with per-testset accuracy,
average precision, real-image accuracy, fake-image accuracy, and an average row.

## Training

```bash
python train.py \
  --train-root path/to/train \
  --val-root path/to/val \
  --output-dir checkpoints/AiDet_FFT \
  --epochs 20 \
  --batch-size 16
```

Fine-tune from the released checkpoint:

```bash
python train.py \
  --train-root path/to/train \
  --val-root path/to/val \
  --pretrained weights/AiDet_FFT_8.pth
```

## Checkpoint

The default checkpoint path is:

```text
weights/AiDet_FFT_8.pth
```

The loader accepts checkpoints saved either as a raw state dict or as a dictionary with a `model` key.
