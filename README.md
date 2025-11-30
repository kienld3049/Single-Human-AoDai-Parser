# Segment Workflow

Workflow tích hợp DWPose, Grounded-SAM và các công cụ xử lý ảnh để phân đoạn và phân tích hình ảnh người. 

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Tải Checkpoints](#tải-checkpoints)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Troubleshooting](#troubleshooting)

## 🎯 Tổng Quan

Project này cung cấp workflow hoàn chỉnh để:
- **DWPose**: Phát hiện tư thế người (pose estimation)
- **Grounded-SAM**: Phân đoạn đối tượng dựa trên text prompt
- **NMS**: Non-Maximum Suppression để loại bỏ bounding boxes trùng lặp
- **mIoU**: Đánh giá độ chính xác phân đoạn

## 💻 Yêu Cầu Hệ Thống

- **Python**: >= 3.8
- **PyTorch**: >= 1.7
- **TorchVision**: >= 0.8
- **CUDA**: 11.3 hoặc cao hơn (khuyến nghị cho GPU)
- **RAM**: Tối thiểu 8GB
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị)

## 🔧 Cài Đặt

### Bước 1: Tạo Conda Environment

```bash
# Tạo environment từ file cấu hình
conda env create -f environment.yaml

# Kích hoạt environment
conda activate control-v11
```

### Bước 2: Cài Đặt DWPose Dependencies

```bash
# Cài đặt ONNX Runtime (CPU)
pip install onnxruntime

# Hoặc cài đặt với GPU support (khuyến nghị)
pip install onnxruntime-gpu
```

**Lưu ý**: Nếu gặp khó khăn với onnxruntime, tham khảo [opencv_onnx branch](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx). 

### Bước 3: Cài Đặt Grounded-SAM

#### Thiết Lập Biến Môi Trường (cho GPU)

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.3/  # Điều chỉnh theo đường dẫn CUDA của bạn
```

#### Cài Đặt Segment Anything

```bash
python -m pip install -e segment_anything
```

#### Cài Đặt Grounding DINO

```bash
pip install --no-build-isolation -e GroundingDINO
```

#### Cài Đặt Dependencies Bổ Sung

```bash
# Diffusers
pip install --upgrade diffusers[torch]

# OpenCV và các thư viện xử lý ảnh
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

## 📦 Tải Checkpoints

### DWPose Models

```bash
# Tạo thư mục checkpoints
mkdir -p annotator/ckpts

# Tải DWPose model
# dw-ll_ucoco_384.onnx: 
# - Baidu: https://pan. baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7
# - Google Drive: https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing

# Tải Detection model
# yolox_l.onnx:
# - Baidu: https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn
# - Google Drive: https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing
```

### Grounded-SAM Models

```bash
# Tải SAM checkpoint (1.2GB)
wget https://dl. fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Tải GroundingDINO checkpoint (694MB)
wget https://github. com/IDEA-Research/GroundingDINO/releases/download/v0.1. 0-alpha2/groundingdino_swinb_cogcoor.pth
```

## 📁 Cấu Trúc Thư Mục

```
Segment-workflow-main/
├── annotator/                    # Các annotator modules
│   ├── ckpts/                   # Checkpoints cho DWPose
│   │   ├── dw-ll_ucoco_384.onnx
│   │   └── yolox_l.onnx
│   ├── dwpose/                  # DWPose implementation
│   └── ... 
├── GroundingDINO/               # GroundingDINO module
├── segment_anything/            # Segment Anything Model
├── get_dwpose_results.py        # Script chạy DWPose
├── get_grounded_sam_output.py   # Script chạy Grounded-SAM
├── get_miou.py                  # Script tính mIoU
├── non_max_suppression.py       # Script NMS
├── environment.yaml             # Conda environment config
└── README.md
```

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chạy DWPose (Pose Estimation)

```bash
python get_dwpose_results.py \
    --input_dir ./input_images \
    --output_dir ./output_dwpose
```

**Parameters**:
- `--input_dir`: Thư mục chứa ảnh đầu vào
- `--output_dir`: Thư mục lưu kết quả pose estimation

### 2. Chạy Grounded-SAM (Segmentation)

```bash
python get_grounded_sam_output. py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
    --grounded_checkpoint groundingdino_swinb_cogcoor.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_image_dir ./input_images \
    --output_dir ./output_segments \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --dataset "ATR"
```

**Parameters**:
- `--config`: File cấu hình GroundingDINO
- `--grounded_checkpoint`: GroundingDINO model checkpoint
- `--sam_checkpoint`: SAM model checkpoint
- `--input_image_dir`: Thư mục ảnh đầu vào
- `--output_dir`: Thư mục lưu kết quả
- `--box_threshold`: Ngưỡng confidence cho bounding box (0.0-1.0)
- `--text_threshold`: Ngưỡng confidence cho text matching (0.0-1.0)
- `--dataset`: Dataset format ("ATR", "COCO", etc.)

### 3. Chạy Non-Maximum Suppression

```bash
python non_max_suppression.py \
    --input_dir ./detections \
    --output_dir ./nms_results \
    --iou_threshold 0.5
```

**Parameters**:
- `--input_dir`: Thư mục chứa detection results
- `--output_dir`: Thư mục lưu kết quả sau NMS
- `--iou_threshold`: IoU threshold cho NMS (default: 0.5)

### 4. Tính mIoU (Evaluation)

```bash
python get_miou.py \
    --pred_dir ./predictions \
    --gt_dir ./ground_truth \
    --num_classes 18
```

**Parameters**:
- `--pred_dir`: Thư mục chứa predicted masks
- `--gt_dir`: Thư mục chứa ground truth masks
- `--num_classes`: Số lượng classes (ATR: 18, LIP: 20)

## 🎨 Ví Dụ Workflow Hoàn Chỉnh

```bash
# 1. Kích hoạt environment
conda activate control-v11

# 2.  Tạo thư mục output
mkdir -p outputs/{dwpose,segments,nms,evaluation}

# 3. Chạy DWPose
python get_dwpose_results.py \
    --input_dir ./data/images \
    --output_dir ./outputs/dwpose

# 4. Chạy Grounded-SAM
python get_grounded_sam_output.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
    --grounded_checkpoint groundingdino_swinb_cogcoor.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_image_dir ./data/images \
    --output_dir ./outputs/segments \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --dataset "ATR"

# 5. Áp dụng NMS (nếu cần)
python non_max_suppression.py \
    --input_dir ./outputs/segments \
    --output_dir ./outputs/nms \
    --iou_threshold 0.5

# 6. Đánh giá với mIoU (nếu có ground truth)
python get_miou.py \
    --pred_dir ./outputs/segments \
    --gt_dir ./data/ground_truth \
    --num_classes 18
```

## 🔍 Troubleshooting

### Lỗi CUDA

**Vấn đề**: `RuntimeError: CUDA out of memory`

**Giải pháp**:
```bash
# Giảm batch size hoặc image resolution
# Hoặc dùng CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Lỗi Import

**Vấn đề**: `ModuleNotFoundError: No module named 'groundingdino'`

**Giải pháp**:
```bash
# Cài đặt lại GroundingDINO
pip install --no-build-isolation -e GroundingDINO
```

### Lỗi ONNX Runtime

**Vấn đề**: ONNX Runtime không tương thích

**Giải pháp**:
```bash
# Gỡ cài đặt và cài lại
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu  # hoặc onnxruntime cho CPU
```

### Checkpoints Không Tìm Thấy

**Vấn đề**: `FileNotFoundError: [Errno 2] No such file or directory: 'xxx.pth'`

**Giải pháp**:
- Kiểm tra checkpoints đã được tải về đúng thư mục
- Đảm bảo đường dẫn trong command chính xác
- Sử dụng đường dẫn tuyệt đối nếu cần

## 📚 Tài Liệu Tham Khảo

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [DWPose](https://github. com/IDEA-Research/DWPose)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

## 📝 Notes

- Đảm bảo có đủ dung lượng ổ cứng (~5GB cho checkpoints)
- GPU memory tối thiểu 8GB cho các models lớn
- Thời gian xử lý phụ thuộc vào kích thước ảnh và hardware

## ⚖️ License

Tham khảo LICENSE files trong các thư mục con cho thông tin chi tiết.

# For DWPose (Legacy Documentation)
🌵🌵🌵 This environment helps you to apply DWPose to ControlNet and prepare for installing Grounded-SAM.

🌵 First, make sure to run ControlNet successfully.
```
# Set ControlNet environment
conda env create -f environment.yaml
conda activate control-v11
```
🌵 Second, install tools to apply DWPose to ControlNet. If it's hard to install onnxruntime, you can refer branch [opencv_onnx](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx), which runs the onnx model with opencv.
```
# Set ControlNet environment
pip install onnxruntime
# if gpu is available
pip install onnxruntime-gpu
```

# Grounded-Segment-Anything (Legacy Documentation)

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Install without Docker (Recommended)
You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```


Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

More details can be found in [install segment anything](https://github.com/facebookresearch/segment-anything#installation) and [install GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install) and [install OSX](https://github.com/IDEA-Research/OSX)

# How to get segment images

First, you need to create your image folder and download all necessary checkpoints for models:

For DWPose: Download dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)), then put them into annotator/ckpts.

For GroundingDINO: Download SwinB checkpoints for the best quality ([Github link](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth)).

For Segment Anything Model: Download the SAM-ViT-h checkpoints ([link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)). 

Or run this command to get both checkpoints for Grounded-SAM:

```

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

Then, simply run:

```

python get_grounded_sam_output.py   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py   --grounded_checkpoint groundingdino_swinB_cogcoor.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_image_dir [YOUR_IMAGE_FOLDER]   --output_dir [YOUR_OUTPUT_FOLDER]   --box_threshold 0.3   --text_threshold 0.25 --dataset "ATR"
```
