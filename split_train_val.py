import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


SEED = 42
VAL_RATIO = 0.1


def split_train_val(data_root, val_ratio=VAL_RATIO):
    root = Path(data_root).expanduser().resolve()
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")

    # Tránh vô tình chia lại train khi val đã được tạo.
    if val_dir.exists():
        raise FileExistsError(
            f"Thư mục val đã tồn tại: {val_dir}\n"
            "Không chia lại để giữ nguyên phép chia dùng chung cho các model."
        )

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio phải nằm trong khoảng (0, 1).")

    dataset = ImageFolder(str(train_dir))

    if len(dataset) < 2:
        raise ValueError("Cần ít nhất 2 ảnh trong train để tách validation.")

    n_val = max(1, int(len(dataset) * val_ratio))
    n_val = min(n_val, len(dataset) - 1)
    n_train = len(dataset) - n_val

    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Chuyển ảnh validation sang val/, giữ nguyên thư mục lớp.
    val_dir.mkdir(parents=True)

    for index in val_subset.indices:
        source = Path(dataset.samples[index][0])
        relative_path = source.relative_to(train_dir)
        destination = val_dir / relative_path

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    print(f"Số lớp: {len(dataset.classes)}")
    print(f"Ảnh còn trong train: {len(train_subset)}")
    print(f"Ảnh chuyển sang val: {len(val_subset)}")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        help="Thư mục chứa train/ và test/, ví dụ .\\VinaFood21",
    )
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()

    split_train_val(args.data_root, args.val_ratio)
