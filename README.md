# DS201 Lab 2: Image Classification on VinaFood21

Các mô hình PyTorch phân loại ảnh VinaFood21 gồm:

- LeNet.
- GoogLeNet rút gọn với 3 Inception block.
- ResNet18 huấn luyện từ đầu, có thể chạy có hoặc không augmentation.
- ResNet-50 pretrained từ Hugging Face, hỗ trợ feature extraction hoặc fine-tuning.

Chương trình dùng Adam và báo cáo accuracy, precision-macro, recall-macro, F1-macro. Khi đánh giá, chương trình in thêm kết quả theo từng lớp và lưu confusion matrix.

## Cấu trúc dữ liệu

Đặt dataset ngoài repository, ở thư mục `VinaFood21` cùng cấp với thư mục project. Khi chạy lệnh từ thư mục project, đường dẫn là `..\VinaFood21`:

```text
VinaFood21/
├── train/
│   ├── ten_lop_1/
│   └── ...
├── val/                 # được tạo từ train/
│   ├── ten_lop_1/
│   └── ...
└── test/
    ├── ten_lop_1/
    └── ...
```

Mỗi split cần có thư mục con cho từng lớp. Tách `val/` từ `train/` một lần trước khi huấn luyện:

```powershell
python split_train_val.py --data_root "..\VinaFood21"
```

Script dùng `random_split` với `torch.Generator().manual_seed(42)`, chuyển mặc định 10% ảnh từ `train/` sang `val/` và giữ nguyên thư mục lớp. `test/` không bị thay đổi. Chạy script một lần; script sẽ dừng nếu `val/` đã tồn tại để tránh chia lại dữ liệu. Dùng cùng `VinaFood21/` này cho cả bốn model để chúng dùng chung phép chia train/validation.

Nếu chưa tạo `val/`, `train.py` vẫn có thể tách validation tạm thời từ `train/` theo `--val_ratio` (mặc định 10%) và `--seed` (mặc định 42). Validation luôn dùng transform không augmentation. Tập test không được dùng trong quá trình huấn luyện.

Ảnh được resize theo kích thước của từng model và chuẩn hóa theo mean/std ImageNet. Dataset nằm ngoài repository để không đẩy các ảnh dung lượng lớn lên GitHub.

## Cài đặt

Mở PowerShell tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường ảo, có thể gọi trực tiếp Python trong `.venv\Scripts\python.exe`.

## Huấn luyện

Chạy các lệnh từ thư mục dự án. Mặc định ảnh được tăng cường khi huấn luyện.

```powershell
# LeNet
python train.py --model lenet --data_root "..\VinaFood21"

# GoogLeNet rút gọn
python train.py --model googlenet --data_root "..\VinaFood21"

# ResNet18: so sánh augmentation và không augmentation
python train.py --model resnet18 --data_root "..\VinaFood21"
python train.py --model resnet18 --data_root "..\VinaFood21" --no_augment

# ResNet-50: feature extraction (đóng băng backbone) và fine-tuning
python train.py --model resnet50 --data_root "..\VinaFood21" --freeze_backbone
python train.py --model resnet50 --data_root "..\VinaFood21"
```

Kích thước ảnh mặc định: LeNet/GoogLeNet 64, ResNet18 96, ResNet-50 224. Có thể đổi bằng `--img_size`. ResNet-50 cần tải trọng số pretrained `microsoft/resnet-50` từ Hugging Face trong lần chạy đầu.

Checkpoint tốt nhất theo validation F1-macro, lịch sử CSV và loss curve được lưu trong `outputs/<model>_<setting>/`. Ví dụ LeNet AvgPool: `outputs/lenet_aug/`.

## Đánh giá trên test

Ví dụ đánh giá checkpoint LeNet AvgPool:

```powershell
python evaluate.py --data_root "..\VinaFood21" --ckpt ".\outputs\lenet_aug\best.pt"
```

Đổi `--ckpt` sang checkpoint của model cần đánh giá. Báo cáo test gồm accuracy, precision-macro, recall-macro, F1-macro, precision/recall/F1 từng lớp và confusion matrix.


