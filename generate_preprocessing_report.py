from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = BASE_DIR / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
REPORT_DIR = BASE_DIR / "preprocessing_report"
MODALITIES = ("t1", "t1ce", "t2", "flair")
TARGET_SIZE = 256
DEFAULT_ANALYSIS_PATIENTS = 48
MASK_COLORS = {
    0: (0, 0, 0, 0),
    1: (255, 99, 71, 140),
    2: (65, 105, 225, 140),
    3: (255, 215, 0, 140),
}


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    nonzero = volume != 0
    if not np.any(nonzero):
        return volume
    mean = volume[nonzero].mean()
    std = volume[nonzero].std()
    if std == 0:
        std = 1.0
    volume[nonzero] = (volume[nonzero] - mean) / std
    return volume


def resize_slice(slice_2d: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(slice_2d.astype(np.float32), mode="F")
    return np.array(image.resize((size, size), resample=resample), dtype=np.float32)


def patient_dirs_with_seg(root: Path) -> list[Path]:
    result = []
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        seg_path = patient_dir / f"{patient_dir.name}_seg.nii"
        if seg_path.exists():
            result.append(patient_dir)
    return result


def load_volume(patient_dir: Path, suffix: str) -> np.ndarray:
    path = patient_dir / f"{patient_dir.name}_{suffix}.nii"
    return nib.load(str(path)).get_fdata()


def remap_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.int16).copy()
    mask[mask == 4] = 3
    return mask


def choose_representative_slice(mask_3d: np.ndarray) -> int:
    tumor_pixels = [int(np.count_nonzero(mask_3d[:, :, idx] > 0)) for idx in range(mask_3d.shape[-1])]
    return int(np.argmax(tumor_pixels))


def scaled_uint8(image_2d: np.ndarray) -> np.ndarray:
    image = image_2d.astype(np.float32)
    low = float(np.percentile(image, 1))
    high = float(np.percentile(image, 99))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(image, low, high)
    scaled = (clipped - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def mask_overlay(base_image: np.ndarray, mask_2d: np.ndarray) -> Image.Image:
    base = Image.fromarray(scaled_uint8(base_image), mode="L").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_rgba = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 4), dtype=np.uint8)
    for class_idx, color in MASK_COLORS.items():
        mask_rgba[mask_2d == class_idx] = color
    overlay = Image.fromarray(mask_rgba, mode="RGBA")
    return Image.alpha_composite(base, overlay)


def draw_histogram(values: np.ndarray, output_path: Path, title: str, color: tuple[int, int, int]) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 60, 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 20), title, fill="black")
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="black", width=2)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)

    counts, bins = np.histogram(values.astype(np.float32), bins=40)
    max_count = max(int(counts.max()), 1)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_width = plot_width / len(counts)

    for idx, count in enumerate(counts):
        x0 = margin_left + idx * bar_width
        x1 = margin_left + (idx + 1) * bar_width - 1
        y1 = height - margin_bottom
        bar_height = (count / max_count) * plot_height
        y0 = y1 - bar_height
        draw.rectangle((x0, y0, x1, y1), fill=color)

    draw.text((margin_left, height - margin_bottom + 15), f"min={bins[0]:.2f}", fill="black")
    draw.text((width - 180, height - margin_bottom + 15), f"max={bins[-1]:.2f}", fill="black")
    draw.text((margin_left + 5, margin_top - 25), f"max count={max_count}", fill="black")
    canvas.save(output_path)


def draw_bar_chart(items: list[tuple[str, float]], output_path: Path, title: str, color: tuple[int, int, int]) -> None:
    width, height = 1000, 560
    margin_left, margin_right, margin_top, margin_bottom = 120, 40, 70, 100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 25), title, fill="black")
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="black", width=2)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)

    max_value = max((value for _, value in items), default=1.0)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    gap = 18
    bar_width = max(20, int((plot_width - gap * (len(items) - 1)) / max(len(items), 1)))

    for idx, (label, value) in enumerate(items):
        x0 = margin_left + idx * (bar_width + gap)
        x1 = x0 + bar_width
        y1 = height - margin_bottom
        y0 = y1 - (value / max_value) * plot_height
        draw.rectangle((x0, y0, x1, y1), fill=color)
        draw.text((x0, y1 + 12), label, fill="black")
        draw.text((x0, max(35, y0 - 18)), f"{value:.2%}", fill="black")

    canvas.save(output_path)


def build_panel(images: list[tuple[str, Image.Image]], output_path: Path) -> None:
    cell_width, cell_height = 320, 320
    label_height = 34
    cols = 3
    rows = (len(images) + cols - 1) // cols
    panel = Image.new("RGB", (cols * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(panel)

    for idx, (label, image) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * (cell_height + label_height)
        fitted = image.convert("RGB").resize((cell_width, cell_height))
        panel.paste(fitted, (x, y))
        draw.text((x + 8, y + cell_height + 8), label, fill="black")

    panel.save(output_path)


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 36
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image.convert("RGB"), (0, title_height))
    draw.text((10, 10), title, fill="black")
    return canvas


def save_step_visuals(
    raw_slice: np.ndarray,
    normalized_slice: np.ndarray,
    resized_slice: np.ndarray,
    raw_mask_slice: np.ndarray,
    remapped_mask_slice: np.ndarray,
    resized_mask_slice: np.ndarray,
    output_dir: Path,
) -> None:
    raw_img = Image.fromarray(scaled_uint8(raw_slice), mode="L").convert("RGB")
    norm_img = Image.fromarray(scaled_uint8(normalized_slice), mode="L").convert("RGB")
    resized_img = Image.fromarray(scaled_uint8(resized_slice), mode="L").convert("RGB")

    steps = [
        ("step_1_raw_flair.png", add_title(raw_img, "Step 1: Raw flair slice")),
        (
            "step_2_raw_with_mask.png",
            add_title(mask_overlay(raw_slice, raw_mask_slice).convert("RGB"), "Step 2: Raw slice with original mask"),
        ),
        (
            "step_3_normalized_flair.png",
            add_title(norm_img, "Step 3: After intensity normalization"),
        ),
        (
            "step_4_normalized_with_remapped_mask.png",
            add_title(
                mask_overlay(normalized_slice, remapped_mask_slice).convert("RGB"),
                "Step 4: Normalized slice with remapped mask labels",
            ),
        ),
        (
            "step_5_resized_flair.png",
            add_title(resized_img, "Step 5: Resized slice to 256 x 256"),
        ),
        (
            "step_6_resized_with_mask.png",
            add_title(
                mask_overlay(resized_slice, resized_mask_slice).convert("RGB"),
                "Step 6: Final processed image used by the model",
            ),
        ),
    ]

    for filename, image in steps:
        image.save(output_dir / filename)


def segmentation_distribution(patient_dirs: list[Path]) -> dict:
    counts = Counter()
    total_slices = 0
    tumor_slices = 0
    for patient_dir in patient_dirs:
        seg = load_volume(patient_dir, "seg").astype(np.int16)
        total_slices += int(seg.shape[-1])
        tumor_slices += int(np.sum(np.any(seg > 0, axis=(0, 1))))
        unique, unique_counts = np.unique(seg, return_counts=True)
        for label, label_count in zip(unique.tolist(), unique_counts.tolist()):
            counts[int(label)] += int(label_count)
    remapped = {
        0: counts.get(0, 0),
        1: counts.get(1, 0),
        2: counts.get(2, 0),
        3: counts.get(4, 0),
    }
    return {
        "raw_counts": dict(counts),
        "remapped_counts": remapped,
        "total_slices": total_slices,
        "tumor_slices": tumor_slices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a visual BraTS preprocessing report.")
    parser.add_argument(
        "--max-analysis-patients",
        type=int,
        default=DEFAULT_ANALYSIS_PATIENTS,
        help="Number of labeled patients to scan for dataset-wide slice and class distributions.",
    )
    return parser.parse_args()


def tensor_sample_summary(patient_dir: Path, slice_idx: int) -> dict:
    channels = []
    channel_stats = {}
    for modality in MODALITIES:
        raw = load_volume(patient_dir, modality)
        normalized = normalize_volume(raw.copy())
        resized = resize_slice(normalized[:, :, slice_idx], TARGET_SIZE, nearest=False)
        channels.append(resized)
        nonzero = raw[raw != 0]
        normalized_nonzero = normalized[normalized != 0]
        channel_stats[modality] = {
            "raw_mean": float(nonzero.mean()),
            "raw_std": float(nonzero.std()),
            "normalized_mean": float(normalized_nonzero.mean()),
            "normalized_std": float(normalized_nonzero.std()),
            "slice_min": float(resized.min()),
            "slice_max": float(resized.max()),
            "slice_mean": float(resized.mean()),
            "slice_std": float(resized.std()),
        }

    mask = remap_mask(load_volume(patient_dir, "seg"))
    mask_slice = resize_slice(mask[:, :, slice_idx].astype(np.float32), TARGET_SIZE, nearest=True).astype(np.int64)
    class_counts = {int(idx): int(count) for idx, count in zip(*np.unique(mask_slice, return_counts=True))}
    image = np.stack(channels, axis=0).astype(np.float32)
    target = np.eye(4, dtype=np.float32)[mask_slice].transpose(2, 0, 1)
    return {
        "image_shape": tuple(int(v) for v in image.shape),
        "mask_shape": tuple(int(v) for v in target.shape),
        "class_counts": class_counts,
        "channel_stats": channel_stats,
        "flair_patch": image[3, 100:105, 100:105].round(3).tolist(),
        "mask_patch": mask_slice[100:105, 100:105].tolist(),
        "resized_mask": mask_slice,
        "resized_flair": image[3],
    }


def write_markdown(report_path: Path, summary: dict) -> None:
    lines = [
        "# Preprocessing Report",
        "",
        f"Generated from `{summary['patient_id']}` using slice `{summary['slice_idx']}`.",
        f"Dataset-level distributions below were computed from `{summary['analyzed_patients']}` labeled patients for speed.",
        "",
        "## What the preprocessing pipeline does",
        "",
        "1. Load the four MRI volumes (`t1`, `t1ce`, `t2`, `flair`) and the segmentation mask.",
        "2. Clean the image intensities by normalizing only non-zero voxels in each 3D volume.",
        "3. Keep only axial slices that contain tumor pixels when building the training dataset.",
        "4. Remap mask label `4` to class index `3` so the model trains on four classes: `0, 1, 2, 3`.",
        "5. Resize image slices and masks to `256 x 256` before converting them into model tensors.",
        "",
        "## Dataset shape summary",
        "",
        f"- Training patient folders: `{summary['train_patients']}`",
        f"- Training patients with segmentation masks: `{summary['patients_with_seg']}`",
        f"- Patients scanned for distribution summary: `{summary['analyzed_patients']}`",
        f"- Raw 3D volume shape per modality: `{summary['raw_shape']}`",
        f"- Total axial slices across labeled training cases: `{summary['total_slices']}`",
        f"- Tumor-only slices kept for training: `{summary['tumor_slices']}`",
        f"- Tumor slice ratio: `{summary['tumor_ratio']:.3%}`",
        f"- Processed image tensor shape: `{summary['image_shape']}`",
        f"- Processed one-hot mask shape: `{summary['mask_shape']}`",
        "",
        "## Distribution summary",
        "",
        f"- Raw BraTS mask counts: `{summary['raw_counts']}`",
        f"- Remapped training class counts: `{summary['remapped_counts']}`",
        f"- Sample processed mask class counts: `{summary['sample_class_counts']}`",
        "",
        "## Sample output after cleaning and preprocessing",
        "",
        "### Flair patch from the processed tensor",
        "These values come from the resized and normalized `flair` slice, so they are ready for model input.",
        "",
        "```text",
        str(summary["flair_patch"]),
        "```",
        "",
        "### Matching mask patch after label remap and resize",
        "These are class IDs after preprocessing. `0` is background and `1/2/3` are tumor classes.",
        "",
        "```text",
        str(summary["mask_patch"]),
        "```",
        "",
        "## Per-modality cleaning effect",
        "",
    ]

    for modality in MODALITIES:
        stats = summary["channel_stats"][modality]
        lines.extend(
            [
                f"### {modality}",
                f"- Before normalization: mean `{stats['raw_mean']:.3f}`, std `{stats['raw_std']:.3f}` over non-zero voxels",
                f"- After normalization: mean `{stats['normalized_mean']:.3f}`, std `{stats['normalized_std']:.3f}` over non-zero voxels",
                f"- Processed sample slice stats: min `{stats['slice_min']:.3f}`, max `{stats['slice_max']:.3f}`, mean `{stats['slice_mean']:.3f}`, std `{stats['slice_std']:.3f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Saved visuals",
            "",
            "- `preprocessing_report/sample_visuals.png`: raw vs normalized vs resized sample views with segmentation overlays",
            "- `preprocessing_report/step_1_raw_flair.png` to `step_6_resized_with_mask.png`: step-by-step image transformation for one sample slice",
            "- `preprocessing_report/flair_before_hist.png`: raw non-zero flair intensity distribution",
            "- `preprocessing_report/flair_after_hist.png`: normalized non-zero flair intensity distribution",
            "- `preprocessing_report/class_balance.png`: remapped training class balance",
            "",
            "## Step-by-step image process",
            "",
            "1. `step_1_raw_flair.png`: the original MRI slice before any cleaning.",
            "2. `step_2_raw_with_mask.png`: the same raw slice with the original segmentation labels overlaid.",
            "3. `step_3_normalized_flair.png`: the slice after non-zero voxel normalization.",
            "4. `step_4_normalized_with_remapped_mask.png`: normalized image with BraTS label `4` remapped to model class `3`.",
            "5. `step_5_resized_flair.png`: the normalized slice resized to `256 x 256`.",
            "6. `step_6_resized_with_mask.png`: the final preprocessed image and mask alignment used for training.",
            "",
            "## How to interpret the visuals",
            "",
            "- The histogram shift shows the intensity cleaning step. After normalization, the non-zero voxel distribution is centered near 0 with standard deviation near 1.",
            "- The sample visuals show what one representative slice looks like before and after preprocessing, including the segmentation overlay.",
            "- The class balance chart shows the label imbalance you should expect during training; background dominates and tumor classes are much smaller.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    patient_dirs = patient_dirs_with_seg(TRAIN_ROOT)
    analyzed_patient_dirs = patient_dirs[: args.max_analysis_patients]
    first_patient = patient_dirs[0]

    flair = load_volume(first_patient, "flair")
    flair_normalized = normalize_volume(flair.copy())
    seg_raw = load_volume(first_patient, "seg").astype(np.int16)
    seg_remapped = remap_mask(seg_raw)
    slice_idx = choose_representative_slice(seg_raw)

    raw_slice = flair[:, :, slice_idx]
    normalized_slice = flair_normalized[:, :, slice_idx]
    resized_slice = resize_slice(normalized_slice, TARGET_SIZE, nearest=False)
    raw_mask_slice = seg_raw[:, :, slice_idx]
    remapped_resized_mask = resize_slice(seg_remapped[:, :, slice_idx].astype(np.float32), TARGET_SIZE, nearest=True).astype(np.int16)

    build_panel(
        [
            ("Raw flair slice", Image.fromarray(scaled_uint8(raw_slice), mode="L")),
            ("Normalized flair slice", Image.fromarray(scaled_uint8(normalized_slice), mode="L")),
            ("Resized flair slice (256x256)", Image.fromarray(scaled_uint8(resized_slice), mode="L")),
            ("Raw flair + raw mask", mask_overlay(raw_slice, raw_mask_slice)),
            ("Normalized flair + remapped mask", mask_overlay(normalized_slice, seg_remapped[:, :, slice_idx])),
            ("Resized flair + resized mask", mask_overlay(resized_slice, remapped_resized_mask)),
        ],
        REPORT_DIR / "sample_visuals.png",
    )
    save_step_visuals(
        raw_slice,
        normalized_slice,
        resized_slice,
        raw_mask_slice,
        seg_remapped[:, :, slice_idx],
        remapped_resized_mask,
        REPORT_DIR,
    )

    draw_histogram(
        flair[flair != 0],
        REPORT_DIR / "flair_before_hist.png",
        "Flair Intensity Distribution Before Normalization",
        (70, 130, 180),
    )
    draw_histogram(
        flair_normalized[flair_normalized != 0],
        REPORT_DIR / "flair_after_hist.png",
        "Flair Intensity Distribution After Normalization",
        (46, 139, 87),
    )

    distribution = segmentation_distribution(analyzed_patient_dirs)
    total_voxels = sum(distribution["remapped_counts"].values())
    draw_bar_chart(
        [(f"class {idx}", count / total_voxels) for idx, count in distribution["remapped_counts"].items()],
        REPORT_DIR / "class_balance.png",
        "Remapped Class Balance Across Training Masks",
        (205, 92, 92),
    )

    sample = tensor_sample_summary(first_patient, slice_idx)
    summary = {
        "patient_id": first_patient.name,
        "slice_idx": slice_idx,
        "train_patients": sum(1 for path in TRAIN_ROOT.iterdir() if path.is_dir()),
        "patients_with_seg": len(patient_dirs),
        "analyzed_patients": len(analyzed_patient_dirs),
        "raw_shape": tuple(int(v) for v in flair.shape),
        "total_slices": distribution["total_slices"],
        "tumor_slices": distribution["tumor_slices"],
        "tumor_ratio": distribution["tumor_slices"] / distribution["total_slices"],
        "raw_counts": distribution["raw_counts"],
        "remapped_counts": distribution["remapped_counts"],
        "image_shape": sample["image_shape"],
        "mask_shape": sample["mask_shape"],
        "sample_class_counts": sample["class_counts"],
        "channel_stats": sample["channel_stats"],
        "flair_patch": sample["flair_patch"],
        "mask_patch": sample["mask_patch"],
    }

    write_markdown(REPORT_DIR / "README.md", summary)

    print(f"saved report to {REPORT_DIR / 'README.md'}")
    print(f"saved visuals to {REPORT_DIR}")


if __name__ == "__main__":
    main()
