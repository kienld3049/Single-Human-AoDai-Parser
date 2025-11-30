# Segment Workflow

A workflow integrating DWPose, Grounded-SAM, and image processing tools for human image segmentation and analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Download Checkpoints](#download-checkpoints)
- [Directory Structure](#directory-structure)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Notes](#notes)
- [License](#license)
- [Dataset Access](#dataset-access)

## 🎯 Overview

This project provides a complete workflow for:
- **DWPose**: Human pose estimation
- **Grounded-SAM**: Object segmentation based on text prompts
- **NMS**: Non-Maximum Suppression to remove duplicate bounding boxes
- **mIoU**: Segmentation accuracy evaluation

## 💻 System Requirements

- **Python**: >= 3.8
- **PyTorch**: >= 1.7
- **TorchVision**: >= 0.8
- **CUDA**: 11.3 or higher (recommended for GPU)
- **RAM**: Minimum 8GB
- **GPU**: NVIDIA GPU with CUDA support (recommended)

## 🔧 Installation

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate control-v11
```

### Step 2: Install DWPose Dependencies

```bash
# Install ONNX Runtime (CPU)
pip install onnxruntime

# Or install with GPU support (recommended)
pip install onnxruntime-gpu
```

**Note**: If you encounter issues with onnxruntime, refer to the [opencv_onnx branch](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx).

### Step 3: Install Grounded-SAM

#### Set Environment Variables (for GPU)

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.3/  # Adjust to your CUDA path
```

#### Install Segment Anything

```bash
python -m pip install -e segment_anything
```

#### Install Grounding DINO

```bash
pip install --no-build-isolation -e GroundingDINO
```

#### Install Additional Dependencies

```bash
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

## 📦 Download Checkpoints

### DWPose Models

```bash
mkdir -p annotator/ckpts
# Download DWPose model:
# dw-ll_ucoco_384.onnx:
# - Baidu: https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7
# - Google Drive: https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing

# Download Detection model:
# yolox_l.onnx:
# - Baidu: https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn
# - Google Drive: https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing
```

### Grounded-SAM Models

```bash
# Download SAM checkpoint (1.2GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download GroundingDINO checkpoint (694MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## 📁 Directory Structure

```
Segment-workflow-main/
├── annotator/                    # Annotator modules
│   ├── ckpts/                   # DWPose checkpoints
│   │   ├── dw-ll_ucoco_384.onnx
│   │   └── yolox_l.onnx
│   ├── dwpose/                  # DWPose implementation
│   └── ... 
├── GroundingDINO/               # GroundingDINO module
├── segment_anything/            # Segment Anything Model
├── get_dwpose_results.py        # DWPose script
├── get_grounded_sam_output.py   # Grounded-SAM script
├── get_miou.py                  # mIoU evaluation script
├── non_max_suppression.py       # NMS script
├── environment.yaml             # Conda environment config
└── README.md
```

## 🚀 Usage Guide

### 1. Run DWPose (Pose Estimation)

```bash
python get_dwpose_results.py \
    --input_dir ./input_images \
    --output_dir ./output_dwpose
```

**Parameters**:
- `--input_dir`: Input image directory
- `--output_dir`: Output directory for pose estimation results

### 2. Run Grounded-SAM (Segmentation)

```bash
python get_grounded_sam_output.py \
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
- `--config`: GroundingDINO config file
- `--grounded_checkpoint`: GroundingDINO model checkpoint
- `--sam_checkpoint`: SAM model checkpoint
- `--input_image_dir`: Input image directory
- `--output_dir`: Output directory
- `--box_threshold`: Bounding box confidence threshold (0.0-1.0)
- `--text_threshold`: Text matching confidence threshold (0.0-1.0)
- `--dataset`: Dataset format ("ATR", "COCO", etc.)

### 3. Run Non-Maximum Suppression

```bash
python non_max_suppression.py \
    --input_dir ./detections \
    --output_dir ./nms_results \
    --iou_threshold 0.5
```

**Parameters**:
- `--input_dir`: Directory with detection results
- `--output_dir`: Output directory after NMS
- `--iou_threshold`: IoU threshold for NMS (default: 0.5)

### 4. Compute mIoU (Evaluation)

```bash
python get_miou.py \
    --pred_dir ./predictions \
    --gt_dir ./ground_truth \
    --num_classes 18
```

**Parameters**:
- `--pred_dir`: Directory with predicted masks
- `--gt_dir`: Directory with ground truth masks
- `--num_classes`: Number of classes (ATR: 18, LIP: 20)

## 🎨 Example Full Workflow

```bash
# 1. Activate environment
conda activate control-v11

# 2. Create output directories
mkdir -p outputs/{dwpose,segments,nms,evaluation}

# 3. Run DWPose
python get_dwpose_results.py \
    --input_dir ./data/images \
    --output_dir ./outputs/dwpose

# 4. Run Grounded-SAM
python get_grounded_sam_output.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
    --grounded_checkpoint groundingdino_swinb_cogcoor.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_image_dir ./data/images \
    --output_dir ./outputs/segments \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --dataset "ATR"

# 5. Apply NMS (if needed)
python non_max_suppression.py \
    --input_dir ./outputs/segments \
    --output_dir ./outputs/nms \
    --iou_threshold 0.5

# 6. Evaluate with mIoU (if ground truth available)
python get_miou.py \
    --pred_dir ./outputs/segments \
    --gt_dir ./data/ground_truth \
    --num_classes 18
```

## 🔍 Troubleshooting

### CUDA Error

**Issue**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size or image resolution
# Or use CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Import Error

**Issue**: `ModuleNotFoundError: No module named 'groundingdino'`

**Solution**:
```bash
# Reinstall GroundingDINO
pip install --no-build-isolation -e GroundingDINO
```

### ONNX Runtime Error

**Issue**: ONNX Runtime incompatibility

**Solution**:
```bash
# Uninstall and reinstall
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu  # or onnxruntime for CPU
```

### Checkpoints Not Found

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: 'xxx.pth'`

**Solution**:
- Ensure checkpoints are downloaded to the correct directory
- Verify the path in your command
- Use absolute paths if necessary

## 📚 References

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [DWPose](https://github.com/IDEA-Research/DWPose)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

## 📝 Notes

- Ensure sufficient disk space (~5GB for checkpoints)
- Minimum 8GB GPU memory for large models
- Processing time depends on image size and hardware

## ⚖️ License

See LICENSE files in subdirectories for details.

## 📥 Dataset Access

To request access to the fully labeled AoDai dataset, please contact:
- ltha@vnu.edu.vn
- 22025004@vnu.edu.vn
