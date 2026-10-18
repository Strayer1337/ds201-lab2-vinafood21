from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(img_size, augment=True):
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def _make_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def build_train_val_loaders(
    data_root,
    img_size,
    batch_size,
    val_ratio=0.1,
    num_workers=2,
    seed=42,
    augment=True,
):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {train_dir}")
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn 0.")
    if num_workers < 0:
        raise ValueError("num_workers không được âm.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio phải nằm trong khoảng (0, 1).")

    train_tf, eval_tf = build_transforms(img_size, augment=augment)

    if val_dir.is_dir():
        # Dùng split có sẵn; validation luôn dùng transform không augmentation.
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

        if train_ds.classes != val_ds.classes:
            raise ValueError("Thứ tự/tên lớp trong train và val không khớp.")
        classes = train_ds.classes
    else:
        # Tạo hai dataset cùng đọc train/ để áp dụng transform riêng.
        full_train = datasets.ImageFolder(train_dir, transform=train_tf)
        full_eval = datasets.ImageFolder(train_dir, transform=eval_tf)

        if len(full_train) < 2:
            raise ValueError(
                "Cần ít nhất 2 ảnh trong train để tách train và validation."
            )

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=generator).tolist()

        # Bảo đảm cả hai subset đều có ảnh, kể cả dataset nhỏ.
        n_val = int(len(indices) * val_ratio)
        n_val = max(1, min(n_val, len(indices) - 1))

        train_ds = Subset(full_train, indices[n_val:])
        val_ds = Subset(full_eval, indices[:n_val])
        classes = full_train.classes

    train_dl = _make_loader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers, seed=seed
    )
    val_dl = _make_loader(
        val_ds, batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )
    return train_dl, val_dl, classes


def build_test_loader(data_root, img_size, batch_size, num_workers=2):
    test_dir = Path(data_root) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

    _, eval_tf = build_transforms(img_size, augment=False)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)
    test_loader = _make_loader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers, seed=42
    )
    return test_loader, test_dataset.classes

