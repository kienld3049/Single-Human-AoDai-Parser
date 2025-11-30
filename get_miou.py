import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image

# Danh sách các class
CLASSES = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
           'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
checked_classes = ['Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress',
           'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Scarf']
# CLASSES = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants',
#             'Torso-skin', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
NUM_CLASSES = len(CLASSES)
# checked_classes = ['Hat', 'Hair', 'Upper-clothes', 'Dress', 'Pants',
#             'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
def compute_iou(gt_mask, pred_mask, num_classes=NUM_CLASSES):
    """Tính IoU cho từng class và mIoU."""
    iou_per_class = np.zeros(num_classes)
    class_count = np.zeros(num_classes)
    valid_classes = 0

    for cls in range(num_classes):
        # if cls == 10:
        #     continue
        gt_cls = (gt_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()
        # if cls == 13:
        #     gt_cls_torso = (gt_mask == 10)
        #     pred_cls_torso = (pred_mask == 10)
        #     gt_cls_face = np.logical_or(gt_cls, gt_cls_torso)
        #     pred_cls_face = np.logical_or(pred_cls, pred_cls_torso)
        #     intersection = np.logical_and(gt_cls_face, pred_cls_face).sum()
        #     union = np.logical_or(gt_cls_face, pred_cls_face).sum()

        if union > 0 and CLASSES[cls] in checked_classes:
            iou_per_class[cls] = intersection / union
            class_count[cls] += 1
            valid_classes += 1
        elif union == 0 and CLASSES[cls] in checked_classes:
            iou_per_class[cls] = 1
            class_count[cls] += 1
            valid_classes += 1
    
    miou = np.sum(iou_per_class) / valid_classes if valid_classes > 0 else 0
    return iou_per_class, miou, class_count

def evaluate_iou(gt_dir, pred_dir):
    """Tính IoU cho từng ảnh và tính trung bình mIoU trên tập dữ liệu."""
    iou_total = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)
    miou_total = []
    num_images = 0

    gt_images = sorted(os.listdir(gt_dir))
    pred_images = sorted(os.listdir(pred_dir))

    for gt_file, pred_file in tqdm(zip(gt_images, pred_images), total=len(gt_images)):
        if gt_file.split(".")[0] in ["NBD06886", "TNTMEDIA-2523"]:
            continue
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)

        gt_mask = np.array(Image.open(gt_path).convert("L"))
        # pred_mask = color2gray(Image.open(pred_path), NUM_CLASSES)
        pred_mask = np.array(Image.open(pred_path).convert("L"))

        if gt_mask is None or pred_mask is None:
            continue

        iou_per_class, miou, class_count = compute_iou(gt_mask, pred_mask)
        # print(f"IoU cho image {gt_file}: {iou_per_class}")
        iou_total += iou_per_class
        class_total += class_count
        miou_total.append(miou)
        num_images += 1
    print(iou_total)
    print(class_total)
    avg_iou_per_class = np.divide(iou_total, class_total, out=np.zeros_like(iou_total), where=class_total > 0)
    avg_miou = np.mean(miou_total)

    return avg_iou_per_class, avg_miou

# Thư mục chứa ground_truth và predicted_output
gt_dir = "ATR_parsing"
pred_dir = "ATR_outputs_2"

# Chạy tính toán
iou_per_class, miou = evaluate_iou(gt_dir, pred_dir)

# Hiển thị kết quả
print("\nIoU cho từng class:")
for i, class_name in enumerate(CLASSES):
    if class_name in checked_classes:
        print(f"{class_name}: {iou_per_class[i]:.4f}")

print(f"\nMean IoU (mIoU): {miou:.4f}")
