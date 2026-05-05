from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import random_split

from train_resnet152_unet import (
    TRAIN_ROOT,
    VALIDATION_ROOT,
    BraTSSliceDataset,
    build_model,
    dice_loss,
    normalize_volume,
)


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "resnet152_unet_brats2020.pt"


def ascii_bar(value: float, scale: float = 40.0) -> str:
    width = max(1, int(round(value * scale)))
    return "#" * width


def count_patient_dirs(root: Path) -> int:
    return sum(1 for path in root.iterdir() if path.is_dir())


def patient_dirs_with_seg(root: Path) -> list[Path]:
    dirs = []
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        seg_path = patient_dir / f"{patient_dir.name}_seg.nii"
        if seg_path.exists():
            dirs.append(patient_dir)
    return dirs


def first_patient_summary(root: Path) -> dict:
    patient_dir = patient_dirs_with_seg(root)[0]
    flair = nib.load(str(patient_dir / f"{patient_dir.name}_flair.nii")).get_fdata()
    seg = nib.load(str(patient_dir / f"{patient_dir.name}_seg.nii")).get_fdata().astype(np.int16)
    nonzero = flair[flair != 0]
    flair_norm = normalize_volume(flair.copy())
    flair_norm_nonzero = flair_norm[flair_norm != 0]
    tumor_slices = int(np.sum(np.any(seg > 0, axis=(0, 1))))
    return {
        "patient_id": patient_dir.name,
        "raw_shape": tuple(int(v) for v in flair.shape),
        "raw_nonzero_mean": float(nonzero.mean()),
        "raw_nonzero_std": float(nonzero.std()),
        "norm_nonzero_mean": float(flair_norm_nonzero.mean()),
        "norm_nonzero_std": float(flair_norm_nonzero.std()),
        "tumor_slices": tumor_slices,
    }


def segmentation_distribution(root: Path) -> dict:
    counts = Counter()
    tumor_slice_total = 0
    total_slices = 0
    complete_patients = 0
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        seg_path = patient_dir / f"{patient_dir.name}_seg.nii"
        if not seg_path.exists():
            continue
        complete_patients += 1
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
        total_slices += int(seg.shape[-1])
        tumor_slice_total += int(np.sum(np.any(seg > 0, axis=(0, 1))))
        unique, unique_counts = np.unique(seg, return_counts=True)
        for label, label_count in zip(unique.tolist(), unique_counts.tolist()):
            counts[int(label)] += int(label_count)
    total_voxels = sum(counts.values())
    remapped = {
        0: counts.get(0, 0),
        1: counts.get(1, 0),
        2: counts.get(2, 0),
        3: counts.get(4, 0),
    }
    return {
        "raw_counts": dict(counts),
        "remapped_counts": remapped,
        "total_voxels": total_voxels,
        "tumor_slice_total": tumor_slice_total,
        "total_slices": total_slices,
        "complete_patients": complete_patients,
    }


def dataset_sample_summary() -> dict:
    dataset = BraTSSliceDataset(root=TRAIN_ROOT, image_size=256, only_tumor_slices=True)
    image, target = dataset[0]
    class_pixels = target.argmax(dim=0)
    unique, counts = torch.unique(class_pixels, return_counts=True)
    return {
        "indexed_slices": len(dataset),
        "image_shape": tuple(image.shape),
        "mask_shape": tuple(target.shape),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "sample_class_counts": {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())},
    }


def hard_dice_scores(pred_class: torch.Tensor, target_class: torch.Tensor, num_classes: int = 4) -> list[float]:
    scores = []
    for class_idx in range(num_classes):
        pred_mask = pred_class == class_idx
        target_mask = target_class == class_idx
        intersection = (pred_mask & target_mask).sum().item()
        denom = pred_mask.sum().item() + target_mask.sum().item()
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append((2.0 * intersection) / denom)
    return scores


def checkpoint_results(eval_batches: int = 8, batch_size: int = 2) -> dict:
    if not CHECKPOINT_PATH.exists():
        return {
            "checkpoint_found": False,
            "eval_slices": 0,
            "mean_dice_loss": None,
            "per_class_hard_dice": None,
            "pred_distribution": {},
            "target_distribution": {},
        }

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    dataset = BraTSSliceDataset(
        root=Path(checkpoint["train_root"]),
        image_size=int(checkpoint["image_size"]),
        only_tumor_slices=True,
    )
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    model = build_model(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    losses = []
    dice_sum = np.zeros(4, dtype=np.float64)
    slices_seen = 0
    pred_distribution = Counter()
    target_distribution = Counter()

    for batch_start in range(0, min(len(val_set), eval_batches * batch_size), batch_size):
        batch = [val_set[idx] for idx in range(batch_start, min(len(val_set), batch_start + batch_size))]
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        with torch.no_grad():
            logits = model(images)
            losses.append(float(dice_loss(logits, targets).item()))
            pred_class = logits.argmax(dim=1)
            target_class = targets.argmax(dim=1)
        for sample_idx in range(pred_class.shape[0]):
            scores = hard_dice_scores(pred_class[sample_idx], target_class[sample_idx])
            dice_sum += np.array(scores)
            slices_seen += 1
        pred_unique, pred_counts = torch.unique(pred_class, return_counts=True)
        for key, value in zip(pred_unique.tolist(), pred_counts.tolist()):
            pred_distribution[int(key)] += int(value)
        target_unique, target_counts = torch.unique(target_class, return_counts=True)
        for key, value in zip(target_unique.tolist(), target_counts.tolist()):
            target_distribution[int(key)] += int(value)

    return {
        "checkpoint_found": True,
        "eval_slices": slices_seen,
        "mean_dice_loss": float(np.mean(losses)),
        "per_class_hard_dice": (dice_sum / max(1, slices_seen)).tolist(),
        "pred_distribution": dict(pred_distribution),
        "target_distribution": dict(target_distribution),
    }


def main() -> None:
    train_patients = count_patient_dirs(TRAIN_ROOT)
    val_patients = count_patient_dirs(VALIDATION_ROOT)
    first_patient = first_patient_summary(TRAIN_ROOT)
    distribution = segmentation_distribution(TRAIN_ROOT)
    sample = dataset_sample_summary()
    results = checkpoint_results()

    raw_counts = distribution["raw_counts"]
    remapped_counts = distribution["remapped_counts"]
    total_voxels = distribution["total_voxels"]

    print("BraTS 2020 project analysis")
    print(f"train_patients={train_patients}")
    print(f"validation_patients={val_patients}")
    print(f"train_patients_with_seg={distribution['complete_patients']}")
    print(f"train_patients_missing_seg={train_patients - distribution['complete_patients']}")
    print(f"first_patient={first_patient['patient_id']}")
    print(f"raw_volume_shape={first_patient['raw_shape']}")
    print(
        "flair_nonzero_stats_before="
        f"(mean={first_patient['raw_nonzero_mean']:.3f}, std={first_patient['raw_nonzero_std']:.3f})"
    )
    print(
        "flair_nonzero_stats_after="
        f"(mean={first_patient['norm_nonzero_mean']:.3f}, std={first_patient['norm_nonzero_std']:.3f})"
    )
    print(f"tumor_slices_in_first_patient={first_patient['tumor_slices']}")
    print(f"indexed_tumor_slices={sample['indexed_slices']}")
    print(f"tensor_image_shape={sample['image_shape']}")
    print(f"tensor_mask_shape={sample['mask_shape']}")
    print(
        "sample_image_stats="
        f"(min={sample['image_min']:.3f}, max={sample['image_max']:.3f}, "
        f"mean={sample['image_mean']:.3f}, std={sample['image_std']:.3f})"
    )
    print(f"raw_segmentation_counts={raw_counts}")
    print(f"remapped_segmentation_counts={remapped_counts}")
    print(
        "tumor_slice_ratio="
        f"{distribution['tumor_slice_total']}/{distribution['total_slices']} "
        f"({distribution['tumor_slice_total'] / distribution['total_slices']:.3%})"
    )
    print("class_balance_visual")
    for class_idx in sorted(remapped_counts):
        pct = remapped_counts[class_idx] / total_voxels
        print(f"class_{class_idx}: {pct:.3%} {ascii_bar(pct)}")
    if not results["checkpoint_found"]:
        print("checkpoint_status=missing")
        print("checkpoint_note=train the model first if you want checkpoint evaluation metrics")
        return
    print(f"eval_slices={results['eval_slices']}")
    print(f"mean_dice_loss={results['mean_dice_loss']:.4f}")
    print(
        "per_class_hard_dice="
        f"background:{results['per_class_hard_dice'][0]:.4f}, "
        f"label1:{results['per_class_hard_dice'][1]:.4f}, "
        f"label2:{results['per_class_hard_dice'][2]:.4f}, "
        f"label4_as_3:{results['per_class_hard_dice'][3]:.4f}"
    )
    print(f"predicted_pixel_distribution={results['pred_distribution']}")
    print(f"target_pixel_distribution={results['target_distribution']}")


if __name__ == "__main__":
    main()
