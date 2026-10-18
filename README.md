# DS201 Lab 2: Image Classification on VinaFood21

Các mô hình PyTorch phân loại ảnh VinaFood21 gồm:

- LeNet.
- GoogLeNet rút gọn với 3 Inception block.
- ResNet18 huấn luyện từ đầu, có thể chạy có hoặc không augmentation.
- ResNet-50 pretrained từ Hugging Face, hỗ trợ feature extraction hoặc fine-tuning.

Chương trình dùng Adam và báo cáo accuracy, precision-macro, recall-macro, F1-macro. Khi đánh giá, chương trình in thêm kết quả theo từng lớp và lưu confusion matrix.

## Cấu trúc dữ liệu

Đặt dữ liệu cạnh các file Python theo cấu trúc:

```text
VinaFood21/
├── train/
│   ├── ten_lop_1/
│   └── ...
├── val/                 # tùy chọn
│   ├── ten_lop_1/
│   └── ...
└── test/
    ├── ten_lop_1/
    └── ...
```

Mỗi split cần có thư mục con cho từng lớp. Nếu thiếu `val/`, chương trình sẽ tách validation ngẫu nhiên từ `train/` theo `--val_ratio` (mặc định 10%, seed mặc định 42). Validation dùng transform không augmentation. Tập test không bị dùng để tách dữ liệu.

Ảnh được resize theo kích thước của từng model và chuẩn hóa theo mean/std ImageNet. Thư mục dataset bị loại khỏi Git vì dữ liệu ảnh có dung lượng lớn; hãy tải/đặt dữ liệu riêng tại đường dẫn trên.

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
python train.py --model lenet --data_root ".\VinaFood21"


# GoogLeNet rút gọn
python train.py --model googlenet --data_root ".\VinaFood21"

# ResNet18: so sánh augmentation và không augmentation
python train.py --model resnet18 --data_root ".\VinaFood21"
python train.py --model resnet18 --data_root ".\VinaFood21" --no_augment

# ResNet-50: feature extraction (đóng băng backbone) và fine-tuning
python train.py --model resnet50 --data_root ".\VinaFood21" --freeze_backbone
python train.py --model resnet50 --data_root ".\VinaFood21"
```

Kích thước ảnh mặc định: LeNet/GoogLeNet 64, ResNet18 96, ResNet-50 224. Có thể đổi bằng `--img_size`. ResNet-50 cần tải trọng số pretrained `microsoft/resnet-50` từ Hugging Face trong lần chạy đầu.

Checkpoint tốt nhất theo validation F1-macro, lịch sử CSV và loss curve được lưu trong `outputs/<model>_<setting>/`. Ví dụ LeNet AvgPool: `outputs/lenet_aug/`.

## Đánh giá trên test

Ví dụ đánh giá checkpoint LeNet AvgPool:

```powershell
python evaluate.py --data_root ".\VinaFood21" --ckpt ".\outputs\lenet_aug\best.pt"
```

Đổi `--ckpt` sang checkpoint của model cần đánh giá. Báo cáo test gồm accuracy, precision-macro, recall-macro, F1-macro, precision/recall/F1 từng lớp và confusion matrix.


