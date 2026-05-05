import argparse
from pathlib import Path
from typing import List, Tuple

try:
    import nibabel as nib
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Install: torch torchvision nibabel "
        "segmentation-models-pytorch"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = BASE_DIR / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
VALIDATION_ROOT = BASE_DIR / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData"
MODALITIES = ("t1", "t1ce", "t2", "flair")


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


def resize_slice(slice_2d: np.ndarray, size: int, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float()
    kwargs = {"align_corners": False} if mode != "nearest" else {}
    resized = F.interpolate(tensor, size=(size, size), mode=mode, **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()


class BraTSSliceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int = 256,
        only_tumor_slices: bool = True,
        max_patients: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.only_tumor_slices = only_tumor_slices
        self.patient_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if max_patients is not None:
            self.patient_dirs = self.patient_dirs[:max_patients]
        self._volume_cache: dict[tuple[str, str], np.ndarray] = {}
        self._seg_cache: dict[str, np.ndarray] = {}
        self.samples = self._index_samples()

    def _load_volume(self, patient_dir: Path, modality: str) -> np.ndarray:
        key = (patient_dir.name, modality)
        if key not in self._volume_cache:
            path = patient_dir / f"{patient_dir.name}_{modality}.nii"
            self._volume_cache[key] = normalize_volume(nib.load(str(path)).get_fdata())
        return self._volume_cache[key]

    def _load_segmentation(self, patient_dir: Path) -> np.ndarray:
        key = patient_dir.name
        if key not in self._seg_cache:
            seg_path = patient_dir / f"{patient_dir.name}_seg.nii"
            self._seg_cache[key] = nib.load(str(seg_path)).get_fdata().astype(np.int16)
        return self._seg_cache[key]

    def _index_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for patient_dir in self.patient_dirs:
            seg_path = patient_dir / f"{patient_dir.name}_seg.nii"
            if not seg_path.exists():
                continue
            seg = self._load_segmentation(patient_dir)
            for slice_idx in range(seg.shape[-1]):
                if self.only_tumor_slices and not np.any(seg[:, :, slice_idx] > 0):
                    continue
                samples.append((patient_dir, slice_idx))
        if not samples:
            raise ValueError(f"No training slices found under {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_dir, slice_idx = self.samples[index]
        channels: List[np.ndarray] = []
        for modality in MODALITIES:
            volume = self._load_volume(patient_dir, modality)
            slice_2d = resize_slice(volume[:, :, slice_idx], self.image_size, mode="bilinear")
            channels.append(slice_2d)

        mask = self._load_segmentation(patient_dir).astype(np.int64)[:, :, slice_idx]
        mask[mask == 4] = 3
        mask = resize_slice(mask.astype(np.float32), self.image_size, mode="nearest").astype(np.int64)

        image = torch.from_numpy(np.stack(channels, axis=0)).float()
        target = F.one_hot(torch.from_numpy(mask), num_classes=4).permute(2, 0, 1).float()
        return image, target


def build_model(encoder_name: str, encoder_weights: str | None = "imagenet") -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_depth=5,
        decoder_channels=(1024, 512, 256, 128, 64),
        in_channels=4,
        classes=4,
        activation=None,
    )


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)
    score = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - score.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = len(loader)
    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(
            f"train epoch {epoch}/{total_epochs} "
            f"batch {batch_idx}/{total_batches} "
            f"loss={loss.item():.4f}",
            flush=True,
        )
    return total_loss / len(loader)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = len(loader)
    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = dice_loss(logits, masks).item()
        total_loss += loss
        print(
            f"valid batch {batch_idx}/{total_batches} loss={loss:.4f}",
            flush=True,
        )
    return total_loss / len(loader)


def make_loaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    val_split: float,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet152 U-Net on BraTS 2020.")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("resnet152_unet_brats2020.pt"))
    parser.add_argument("--encoder", type=str, default="resnet152")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"device={device} encoder={args.encoder}", flush=True)
    print(f"train_root={args.train_root}", flush=True)

    dataset = BraTSSliceDataset(
        root=args.train_root,
        image_size=args.image_size,
        only_tumor_slices=True,
        max_patients=args.max_patients,
    )
    print(f"indexed_slices={len(dataset)}", flush=True)
    train_loader, val_loader = make_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        pin_memory=pin_memory,
    )

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_model(args.encoder, encoder_weights=encoder_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, device)
        print(
            f"epoch={epoch} train_dice_loss={train_loss:.4f} "
            f"val_dice_loss={val_loss:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_root": str(args.train_root),
            "modalities": MODALITIES,
            "image_size": args.image_size,
            "encoder": args.encoder,
            "encoder_weights": args.encoder_weights,
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")
    print(f"validation dataset is available at: {VALIDATION_ROOT}")


if __name__ == "__main__":
    main()
