import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from model import build_model, n_params


DEFAULT_IMG_SIZE = {
    "lenet": 64,
    "googlenet": 64,
    "resnet18": 96,
    "resnet50": 224,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện các mô hình cho bài Lab 2"
    )
    parser.add_argument(
        "--model",
        choices=("lenet", "googlenet", "resnet18", "resnet50"),
        default="lenet",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Thư mục chứa train/, val/ và test/.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Mặc định: LeNet/GoogLeNet 64, ResNet18 96, ResNet-50 224.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Mặc định 1e-3; ResNet-50 fine-tune 1e-4.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Tắt augmentation; dùng để chạy phiên bản ResNet18 không augment.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Chỉ huấn luyện classifier của ResNet-50 pretrained.",
    )
    parser.add_argument("--out_dir", type=str, default="outputs")
    return parser.parse_args()


def _run_name(args, augment):
    name = args.model
    if args.model == "resnet50":
        name += "_frozen" if args.freeze_backbone else "_finetune"
    name += "_aug" if augment else "_noaug"
    return name

def _print_model_details(model, model_name):
    print(f"Tổng số tham số: {n_params(model):,}")
    print(f"Số tham số được huấn luyện: {n_params(model, trainable_only=True):,}")
    if model_name == "googlenet":
        for index, block_name in enumerate(
            ("inception1", "inception2", "inception3"), start=1
        ):
            block = getattr(model, block_name)
            print(f"Inception block {index}: {block.out_channels} output channels")


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()
    if args.epochs < 1:
        raise ValueError("--epochs phải lớn hơn 0.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio phải nằm trong khoảng (0, 1).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = args.img_size or DEFAULT_IMG_SIZE[args.model]
    use_augmentation = not args.no_augment

    train_loader, val_loader, classes = build_train_val_loaders(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=use_augmentation,
    )

    model = build_model(
        model_name=args.model,
        num_classes=len(classes),
        input_size=img_size,
        in_ch=3,
        freeze_backbone=args.freeze_backbone,
        pretrained=True,
    ).to(device)

    print(
        f"Model: {args.model} | device: {device} | input: 3x{img_size}x{img_size}"
    )
    print(
        f"Lớp: {len(classes)} | train: {len(train_loader.dataset)}"
        f" | validation: {len(val_loader.dataset)}"
    )
    _print_model_details(model, args.model)

    if args.lr is not None:
        learning_rate = args.lr
    elif args.model == "resnet50" and not args.freeze_backbone:
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    print(f"Optimizer: Adam | learning rate: {learning_rate:g}")

    run_dir = Path(args.out_dir) / _run_name(args, use_augmentation)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            labels=list(range(len(classes))),
        )
        epoch_seconds = time.perf_counter() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision_macro": val_precision,
            "val_recall_macro": val_recall,
            "val_f1_macro": val_f1,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            checkpoint = {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "classes": classes,
                "img_size": img_size,
                "num_classes": len(classes),
                "freeze_backbone": args.freeze_backbone,
                "epoch": epoch,
                "learning_rate": learning_rate,
            }
            if args.model == "resnet50":
                checkpoint["backbone_config"] = model.backbone_config
            torch.save(checkpoint, run_dir / "best.pt")

        mark = " *" if improved else ""
        print(
            f"Epoch {epoch:>2}/{args.epochs} | "
            f"train loss/acc {train_loss:.4f}/{train_acc:.4f} | "
            f"val loss/acc {val_loss:.4f}/{val_acc:.4f} | "
            f"P/R/F1-macro {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f} | "
            f"{epoch_seconds:.1f}s{mark}"
        )

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    save_loss_curve(history, run_dir / "loss_curve.png")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Kết quả được lưu tại: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

