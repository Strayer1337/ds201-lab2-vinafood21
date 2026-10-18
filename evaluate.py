import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from data import build_test_loader
from engine import evaluate
from model import build_model


def get_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá checkpoint trên tập test"
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/lenet_aug/best.pt",
        help="Đường dẫn best.pt do train.py tạo.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def save_confusion_matrix(matrix, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=90)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Nhãn dự đoán")
    axis.set_ylabel("Nhãn thật")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location=device)
    classes = checkpoint["classes"]
    img_size = checkpoint["img_size"]
    model_name = checkpoint.get("model_name", "lenet")
    num_classes = checkpoint.get("num_classes", len(classes))

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        input_size=img_size,
        in_ch=3,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        pretrained=(model_name != "resnet50"),
        backbone_config=checkpoint.get("backbone_config"),
    ).to(device)

    state_key = "model_state" if "model_state" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])

    test_loader, test_classes = build_test_loader(
        args.data_root,
        img_size,
        args.batch_size,
        args.num_workers,
    )
    if test_classes != classes:
        raise ValueError("Thứ tự lớp trong test không khớp với checkpoint.")

    loss, accuracy, precision, recall, macro_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
        labels=list(range(len(classes))),
    )

    print(
        f"Model: {model_name} | checkpoint epoch: "
        f"{checkpoint.get('epoch', 'không rõ')} | test loss: {loss:.4f}"
    )
    print(
        f"Accuracy: {accuracy:.4f} | Precision-macro: {precision:.4f} | "
        f"Recall-macro: {recall:.4f} | F1-macro: {macro_f1:.4f}"
    )
    print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    output_path = Path(args.ckpt).parent / "confusion_matrix.png"
    save_confusion_matrix(
        confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
        classes,
        output_path,
    )
    print(f"Confusion matrix được lưu tại: {output_path}")


if __name__ == "__main__":
    main()



