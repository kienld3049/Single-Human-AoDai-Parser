import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
from non_max_suppression import nms_filter
from collections import deque
from natsort import natsorted
from scipy.spatial import KDTree
from get_dwpose_results import get_pose

LIP_LABELS = {
    "background": 0, "hat": 1, "hair": 2, "glove": 3, "sunglasses": 4,
    "upper clothes": 5, "dress": 6, "coat": 7, "socks": 8, "pants": 9,
    "jumpsuits": 10, "scarf": 11, "skirt": 12, "face": 13, "left arm": 14,
    "right arm": 15, "left leg": 16, "right leg": 17, "left shoe": 18, "right shoe": 19
}

LIP_PALETTE = [
    (0, 0, 0),        # Background
    (128, 0, 0),      # Hat
    (255, 0, 0),      # Hair
    (0, 85, 0),       # Glove
    (170, 0, 51),     # Sunglasses
    (255, 85, 0),     # UpperClothes
    (0, 0, 85),       # Dress
    (0, 119, 221),    # Coat
    (85, 85, 0),      # Socks
    (0, 85, 85),      # Pants
    (85, 51, 0),      # Jumpsuits
    (52, 86, 128),    # Scarf
    (0, 128, 0),      # Skirt
    (0, 255, 0),      # Face
    (170, 255, 0),    # LeftArm
    (255, 255, 0),    # RightArm
    (0, 170, 170),    # LeftLeg
    (0, 255, 255),    # RightLeg
    (85, 51, 170),    # LeftShoe
    (170, 170, 255)   # RightShoe
]

ATR_LABELS = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper clothes": 4, 
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left shoe": 9,
    "right shoe": 10, "face": 11, "left leg": 12, "right leg": 13, "left arm": 14,
    "right arm": 15, "bag": 16, "scarf": 17
}

ATR_PALETTE = [
    (0, 0, 0),        # Background
    (128, 0, 0),      # Hat
    (255, 0, 0),      # Hair
    (0, 85, 0),       # Glove
    (170, 0, 51),     # Sunglasses
    (255, 85, 0),     # UpperClothes
    (0, 0, 85),       # Dress
    (0, 119, 221),    # Coat
    (85, 85, 0),      # Socks
    (0, 85, 85),      # Pants
    (85, 51, 0),      # Jumpsuits
    (52, 86, 128),    # Scarf
    (0, 128, 0),      # Skirt
    (0, 255, 0),      # Face
    (170, 255, 0),    # LeftArm
    (255, 255, 0),    # RightArm
    (0, 170, 170),    # LeftLeg
    (0, 255, 255),    # RightLeg
]

AODAI_LABELS = {
    "background": 0, "hat": 1, "hair": 2, "glove": 3, "sunglasses": 4,
    "upper clothes": 5, "dress": 6, "coat": 7, "socks": 8, "pants": 9,
    "torso skin": 10, "scarf": 11, "skirt": 12, "face": 13, "left arm": 14,
    "right arm": 15, "left leg": 16, "right leg": 17, "left shoe": 18, "right shoe": 19
}

AODAI_PALETTE = [
    (0, 0, 0),        # Background
    (128, 0, 0),      # Hat
    (255, 0, 0),      # Hair
    (0, 85, 0),       # Glove
    (170, 0, 51),     # Sunglasses
    (255, 85, 0),     # UpperClothes
    (0, 0, 85),       # Dress
    (0, 119, 221),    # Coat
    (85, 85, 0),      # Socks
    (0, 85, 85),      # Pants
    (85, 51, 0),      # Torso skin
    (52, 86, 128),    # Scarf
    (0, 128, 0),      # Skirt
    (0, 255, 0),      # Face
    (170, 255, 0),    # LeftArm
    (255, 255, 0),    # RightArm
    (0, 170, 170),    # LeftLeg
    (0, 255, 255),    # RightLeg
    (85, 51, 170),    # LeftShoe
    (170, 170, 255)   # RightShoe
]

LIST_OF_UPPER_CLOTHES = ["shirt", "t - shirt", "coat", "blouse", "tanktop", "vest", "sweater"]

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_mask_data(output_dir, mask_path, list_of_mask_list, list_of_box_list, list_of_label_list, height, width, dataset_type):
  mask_img = torch.zeros((height, width))
  if dataset_type == "ATR":
    has_upper_clothes = False
    for label_list in list_of_label_list:
        if len(label_list) == 0:
            continue
        elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
            has_upper_clothes = True
    for mask_list, label_list in zip(list_of_mask_list, list_of_label_list):
        if len(label_list) == 0:
            value = 0
        elif label_list[0].split('(')[0] == "arm":
            value = ATR_LABELS["left arm"]
        elif label_list[0].split('(')[0] == "leg":
            value = ATR_LABELS["left leg"]
        elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
            value = ATR_LABELS["left shoe"]
        elif label_list[0].split('(')[0] == "neck" or label_list[0].split('(')[0] == "ear":
            value = ATR_LABELS["face"]
        elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
            value = ATR_LABELS["pants"]
        elif label_list[0].split('(')[0] == "maxi skirt":
            value = ATR_LABELS["skirt"]
        elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
            continue
        elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
            value = ATR_LABELS["upper clothes"]
        else:
            value = ATR_LABELS[label_list[0].split('(')[0]]
        for idx, mask in enumerate(mask_list):
            if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                break
            mask_img[mask.cpu().numpy()[0] == True] = value
	
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f'{mask_path}_mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    mask_img = mask_img.numpy()
    # color_mask_img = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), dtype=np.uint8)
    # for label_id, color in enumerate(AODAI_PALETTE):
    #     color_mask_img[mask_img == label_id] = color
    # cv2.imwrite(os.path.join(output_dir, f'{mask_path}_mask.jpg'), cv2.cvtColor(color_mask_img, cv2.COLOR_RGB2BGR))
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label_list, box_list in zip(list_of_label_list, list_of_box_list):
        if len(label_list) == 0:
            value = 0
        elif label_list[0].split('(')[0] == "arm":
            value = ATR_LABELS["left arm"]
        elif label_list[0].split('(')[0] == "leg":
            value = ATR_LABELS["left leg"]
        elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
            value = ATR_LABELS["left shoe"]
        elif label_list[0].split('(')[0] == "neck" or label_list[0].split('(')[0] == "ear":
            value = ATR_LABELS["face"]
        elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
            value = ATR_LABELS["pants"]
        elif label_list[0].split('(')[0] == "maxi skirt":
            value = ATR_LABELS["skirt"]
        elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
            continue
        elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
            value = ATR_LABELS["upper clothes"]
        else:
            value = ATR_LABELS[label_list[0].split('(')[0]]
        for idx, (label, box) in enumerate(zip(label_list, box_list)):
            if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                break
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.tolist(),
            })
    with open(os.path.join(output_dir, f'{mask_path}_mask.json'), 'w') as f:
        json.dump(json_data, f)
  else:
    has_upper_clothes = False
    for label_list in list_of_label_list:
        if len(label_list) == 0:
            continue
        elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
            has_upper_clothes = True
    for mask_list, label_list in zip(list_of_mask_list, list_of_label_list):
        if len(label_list) == 0:
            value = 0
        elif label_list[0].split('(')[0] == "arm":
            value = AODAI_LABELS["left arm"]
        elif label_list[0].split('(')[0] == "leg":
            value = AODAI_LABELS["left leg"]
        elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
            value = AODAI_LABELS["left shoe"]
        elif label_list[0].split('(')[0] == "neck":
            value = AODAI_LABELS["torso skin"]
        elif label_list[0].split('(')[0] == "ear":
            value = AODAI_LABELS["face"]
        elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
            value = AODAI_LABELS["pants"]
        elif label_list[0].split('(')[0] == "maxi skirt":
            value = AODAI_LABELS["skirt"]
        elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
            continue
        elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
            value = AODAI_LABELS["upper clothes"]
        else:
            value = AODAI_LABELS[label_list[0].split('(')[0]]
        for idx, mask in enumerate(mask_list):
            if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                break
            mask_img[mask.cpu().numpy()[0] == True] = value
	
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f'{mask_path}_mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    mask_img = mask_img.numpy()
    # color_mask_img = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), dtype=np.uint8)
    # for label_id, color in enumerate(AODAI_PALETTE):
    #     color_mask_img[mask_img == label_id] = color
    # cv2.imwrite(os.path.join(output_dir, f'{mask_path}_mask.jpg'), cv2.cvtColor(color_mask_img, cv2.COLOR_RGB2BGR))
    
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label_list, box_list in zip(list_of_label_list, list_of_box_list):
        if len(label_list) == 0:
            value = 0
        elif label_list[0].split('(')[0] == "arm":
            value = AODAI_LABELS["left arm"]
        elif label_list[0].split('(')[0] == "leg":
            value = AODAI_LABELS["left leg"]
        elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
            value = AODAI_LABELS["left shoe"]
        elif label_list[0].split('(')[0] == "neck":
            value = AODAI_LABELS["torso skin"]
        elif label_list[0].split('(')[0] == "ear":
            value = AODAI_LABELS["face"]
        elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
            value = AODAI_LABELS["pants"]
        elif label_list[0].split('(')[0] == "maxi skirt":
            value = AODAI_LABELS["skirt"]
        elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
            continue
        elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
            value = AODAI_LABELS["upper clothes"]
        else:
            value = AODAI_LABELS[label_list[0].split('(')[0]]
        for idx, (label, box) in enumerate(zip(label_list, box_list)):
            if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                break
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.tolist(),
            })
    with open(os.path.join(output_dir, f'{mask_path}_mask.json'), 'w') as f:
        json.dump(json_data, f)
  return mask_img

def flood_fill(mask_img, start_point, true_label, false_labels, boundary_labels):
    h, w = mask_img.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([start_point])

    def is_valid(x, y):
        return (x >= 0 and x < w) and (y >= 0 and y < h)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    print(f"False labels are {false_labels}")
    print(f"Boundary labels are {boundary_labels}")
    x, y = start_point
    visited[y, x] = True
    count = 0
    while len(queue) > 0:
        count += 1
        x, y = queue.popleft()
        
        if mask_img[y, x] in boundary_labels:
            continue

        if mask_img[y, x] in false_labels:
            mask_img[y, x] = true_label


        for dx, dy in directions:
            x1, y1 = x + dx, y + dy
            if is_valid(x1, y1) and visited[y1, x1] == False:
                queue.append((x1, y1))
                visited[y1, x1] = True
    return mask_img

def point_to_line_distance(x0, y0, x1, y1, x2, y2):
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

def point_to_point_distance(x0, y0, x1, y1):
    return np.sqrt((y1 - y0)**2 + (x1 - x0)**2)

def assign_limbs_points(mask_img, keypoints, labels, pose_type):
    h, w = mask_img.shape

    area_left_arm = np.sum(mask_img == labels["left arm"])
    area_right_arm = np.sum(mask_img == labels["right arm"])
    area_left_leg = np.sum(mask_img == labels["left leg"])
    area_right_leg = np.sum(mask_img == labels["right leg"])
    area_left_shoe = np.sum(mask_img == labels["left shoe"])
    area_right_shoe = np.sum(mask_img == labels["right shoe"])

    all_points = np.argwhere((mask_img == labels["left arm"]) | (mask_img == labels["right arm"]) |
                             (mask_img == labels["left leg"]) | (mask_img == labels["right leg"]) |
                             (mask_img == labels["left shoe"]) | (mask_img == labels["right shoe"]))

    for y, x in all_points:
        if pose_type == "mediapipe":
            dist_left_arm = point_to_line_distance(x, y, *keypoints[13], *keypoints[15])  
            dist_right_arm = point_to_line_distance(x, y, *keypoints[14], *keypoints[16])  
            dist_left_leg = point_to_line_distance(x, y, *keypoints[25], *keypoints[27])  
            dist_right_leg = point_to_line_distance(x, y, *keypoints[26], *keypoints[28])
            # dist_left_shoe = point_to_point_distance(x, y, *keypoints[27])
            # dist_right_shoe = point_to_point_distance(x, y, *keypoints[28])
        elif pose_type == "dwpose":
            dist_left_arm = point_to_line_distance(x, y, *keypoints[6], *keypoints[7])  
            dist_right_arm = point_to_line_distance(x, y, *keypoints[3], *keypoints[4])  
            dist_left_leg = point_to_line_distance(x, y, *keypoints[13], *keypoints[14])  
            dist_right_leg = point_to_line_distance(x, y, *keypoints[10], *keypoints[11])
            # dist_left_shoe = point_to_point_distance(x, y, *keypoints[14])
            # dist_right_shoe = point_to_point_distance(x, y, *keypoints[11])

        if mask_img[y, x] in [labels["left arm"], labels["right arm"]] and (area_left_arm / (area_left_arm + area_right_arm) > 0.6 or area_right_arm / (area_left_arm + area_right_arm) > 0.6):
            mask_img[y, x] = labels["left arm"] if dist_left_arm < dist_right_arm else labels["right arm"]
        if mask_img[y, x] in [labels["left leg"], labels["right leg"]] and (area_left_leg / (area_left_leg + area_right_leg) > 0.6 or area_right_leg / (area_left_leg + area_right_leg) > 0.6):
            mask_img[y, x] = labels["left leg"] if dist_left_leg < dist_right_leg else labels["right leg"]
        # if mask_img[y, x] in [labels["left shoe"], labels["right shoe"]]:
        #     mask_img[y, x] = labels["left shoe"] if dist_left_shoe < dist_right_shoe else labels["right shoe"]

    return mask_img

def point_augment(pair_of_points, max_height=1024):
    x1, y1 = pair_of_points[0]
    x2, y2 = pair_of_points[1]
    if y1 >= 2 * max_height or y2 >= 2 * max_height:
        return pair_of_points
    vector_len_x = x2 - x1
    vector_len_y = y2 - y1
    for i in range(4):
        if i == 3:
            x = x2 + (vector_len_x // 4)
            y = y2 + (vector_len_y // 4)
        else:
            x = x1 + (vector_len_x // 4) * (i + 1)
            y = y1 + (vector_len_y // 4) * (i + 1)
        pair_of_points.append((x, y))
    return pair_of_points

def check_points_in_background(points):
    points = [pt for pt in points if pt[0] > 0 and pt[0] < mask_img.shape[1] and pt[1] > 0 and pt[1] < mask_img.shape[0]]
    bg_points = [pt for pt in points if mask_img[pt[1], pt[0]] == 0]
    # fg_points = np.argwhere(mask_img > 0)
    # result = []
    # for pt in bg_points:
    #     valid_point = True
    #     min_dist = 100000000
    #     for pt2 in fg_points:
    #         y, x = pt2
    #         if point_to_point_distance(*pt, x, y) < min_dist:
    #             min_dist = point_to_point_distance(*pt, x, y)
    #         if point_to_point_distance(*pt, x, y) <= 5:
    #             valid_point = False
    #     print(min_dist)
    #     if valid_point:
    #         result.append(pt)
    # return result
    return bg_points

def select_best_box(point_list, box_list):
    best_box = None
    max_points = 0
    min_area = float('inf')

    for box in box_list:
        x_min, y_min, x_max, y_max = box
        points_inside = sum([
            (x_min <= x <= x_max and y_min <= y <= y_max) for x, y in point_list
        ])
        area = (x_max - x_min) * (y_max - y_min)

        if points_inside > max_points or (points_inside == max_points and area < min_area):
            max_points = points_inside
            min_area = area
            best_box = box

    return best_box

def mask_process(output_dir, mask_path, list_of_box_list, list_of_mask_list, list_of_label_list, predictor, device, mask_img, keypoints_path, dataset_type, pose_type, special_type):
    with open(keypoints_path, 'r') as f:
        keypoints_data = json.load(f)
    
    if pose_type == "mediapipe":
        pose_keypoints_pred = keypoints_data["people"][0]["pose_keypoints_2d"]
        keypoints = [
            (int(pose_keypoints_pred[i]), int(pose_keypoints_pred[i + 1]))
            for i in range(0, len(pose_keypoints_pred), 2)
        ]
    
        x, y = keypoints[33]

        head_points = [(x, y - 10), keypoints[0]]
        left_arm_points = [keypoints[13], keypoints[15], keypoints[17], keypoints[19], keypoints[21]]
        right_arm_points = [keypoints[14], keypoints[16], keypoints[18], keypoints[20], keypoints[22]]
        left_leg_points = [keypoints[25], keypoints[27], keypoints[29]]
        right_leg_points = [keypoints[26], keypoints[28], keypoints[30]]
    elif pose_type == "dwpose":
        pose_keypoints_pred = keypoints_data["people"][0]["pose_keypoints_2d"]
        keypoints = [
            (int(pose_keypoints_pred[i]), int(pose_keypoints_pred[i + 1]))
            for i in range(0, len(pose_keypoints_pred), 2)
        ]
    
        x, y = keypoints[1]

        head_points = [(x, y - 10), keypoints[0]]
        left_arm_points = [keypoints[6], keypoints[7]]
        right_arm_points = [keypoints[3], keypoints[4]]
        left_arm_points = point_augment(left_arm_points)
        right_arm_points = point_augment(right_arm_points)
        if dataset_type == "ATR":
            left_leg_points = [keypoints[14], keypoints[13]]
            right_leg_points = [keypoints[11], keypoints[10]]
        else:
            left_leg_points = [keypoints[14]]
            right_leg_points = [keypoints[11]]

    left_arm_box = None
    right_arm_box = None
    left_leg_box = None
    right_leg_box = None

    for boxes_list, label_list in zip(list_of_box_list, list_of_label_list):
        if len(label_list) == 0:
            continue
        if label_list[0].split("(")[0] == "arm":
            if len(left_arm_points) > 0:
                left_arm_box = select_best_box(left_arm_points, boxes_list)
            if len(right_arm_points) > 0:
                right_arm_box = select_best_box(right_arm_points, boxes_list)
        elif label_list[0].split("(")[0] == "leg":
            if len(left_leg_points) > 0:
                left_leg_box = select_best_box(left_leg_points, boxes_list)
            if len(right_leg_points) > 0:
                right_leg_box = select_best_box(right_leg_points, boxes_list)

    head_points = check_points_in_background(head_points)
    left_arm_points = check_points_in_background(left_arm_points)
    right_arm_points = check_points_in_background(right_arm_points)
    left_leg_points = check_points_in_background(left_leg_points)
    right_leg_points = check_points_in_background(right_leg_points)
    
    list_of_masks = []

    if len(head_points) > 0:
        input_points = np.array(head_points)
        input_labels = np.ones(len(head_points))

        head_masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        list_of_masks.append(head_masks)
    else:
        list_of_masks.append(None)  
    
    if len(left_arm_points) > 0:
        input_points = np.array(left_arm_points)
        input_labels = np.ones(len(left_arm_points))
        if left_arm_box is not None and dataset_type != "ATR":
            left_arm_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box = np.array(left_arm_box),
                multimask_output=False
            )
        else:
            left_arm_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
        list_of_masks.append(left_arm_masks)
    else:
        if left_arm_box is not None and dataset_type != "ATR":
            left_arm_masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box = np.array(left_arm_box),
                multimask_output=False
            )
        else:
            left_arm_masks = None
        list_of_masks.append(left_arm_masks)

    if len(right_arm_points) > 0:
        input_points = np.array(right_arm_points)
        input_labels = np.ones(len(right_arm_points))

        if right_arm_box is not None and dataset_type != "ATR":
            right_arm_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box = np.array(right_arm_box),
                multimask_output=False
            )
        else:
            right_arm_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
        list_of_masks.append(right_arm_masks)
    else:
        if right_arm_box is not None and dataset_type != "ATR":
            right_arm_masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box = np.array(right_arm_box),
                multimask_output=False
            )
        else:
            right_arm_masks = None
        list_of_masks.append(right_arm_masks)

    if len(left_leg_points) > 0:
        input_points = np.array(left_leg_points)
        input_labels = np.ones(len(left_leg_points))

        if left_leg_box is not None and dataset_type != "ATR":
            left_leg_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box = np.array(left_leg_box),
                multimask_output=False
            )
        else:
            left_leg_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
        list_of_masks.append(left_leg_masks)
    else:
        if left_leg_box is not None and dataset_type != "ATR":
            left_leg_masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box = np.array(left_leg_box),
                multimask_output=False
            )
        else:
            left_leg_masks = None
        list_of_masks.append(left_leg_masks)

    if len(right_leg_points) > 0:
        input_points = np.array(right_leg_points)
        input_labels = np.ones(len(right_leg_points))

        if right_leg_box is not None and dataset_type != "ATR":
            right_leg_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box = np.array(right_leg_box),
                multimask_output=False
            )
        else:
            right_leg_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
        list_of_masks.append(right_leg_masks)
    else:
        if right_leg_box is not None and dataset_type != "ATR":
            right_leg_masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box = np.array(right_leg_box),
                multimask_output=False
            )
        else:
            right_leg_masks = None
        list_of_masks.append(right_leg_masks)
    
    if dataset_type == "ATR":
        list_of_labels = [ATR_LABELS["face"], ATR_LABELS["left arm"], ATR_LABELS["right arm"], ATR_LABELS["left leg"], ATR_LABELS["right leg"]]
    elif special_type == True:
        list_of_labels = [AODAI_LABELS["face"], AODAI_LABELS["left arm"], AODAI_LABELS["right arm"], AODAI_LABELS["pants"], AODAI_LABELS["pants"]]
    else:
        list_of_labels = [AODAI_LABELS["face"], AODAI_LABELS["left arm"], AODAI_LABELS["right arm"], AODAI_LABELS["left leg"], AODAI_LABELS["right leg"]]
    
    for masks, label in zip(list_of_masks, list_of_labels):
        if masks is not None:
            for mask in masks:
                mask_img[mask == True] = label
    if dataset_type == "ATR":
        has_upper_clothes = False
        has_belt = False
        has_skirt = False
        for label_list in list_of_label_list:
            if len(label_list) == 0:
                continue
            elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
                has_upper_clothes = True
            elif label_list[0].split('(')[0] == "belt":
                has_belt = True
            elif label_list[0].split('(')[0] == "skirt":
                has_skirt = True
        for mask_list, label_list in zip(list_of_mask_list, list_of_label_list):
            if len(label_list) == 0:
                value = 0
            elif label_list[0].split('(')[0] == "arm":
                continue
            elif label_list[0].split('(')[0] == "leg":
                continue
            elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
                value = ATR_LABELS["left shoe"]
            elif label_list[0].split('(')[0] == "neck" or label_list[0].split('(')[0] == "ear":
                value = ATR_LABELS["face"]
            elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
                value = ATR_LABELS["pants"]
            elif label_list[0].split('(')[0] == "maxi skirt":
                value = ATR_LABELS["skirt"]
            elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
                continue
            elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
                value = ATR_LABELS["upper clothes"]
            else:
                value = ATR_LABELS[label_list[0].split('(')[0]]
            for idx, mask in enumerate(mask_list):
                if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                    break
                mask_img[mask.cpu().numpy()[0] == True] = value
        
        list_of_boundary_labels = [ATR_LABELS["hair"], ATR_LABELS["upper clothes"], ATR_LABELS["background"], ATR_LABELS["scarf"], ATR_LABELS["bag"], ATR_LABELS["dress"]]
        mask_img = flood_fill(mask_img, keypoints[0], ATR_LABELS["face"], list_of_labels[1:], list_of_boundary_labels)
        if len(left_arm_points) > 0:
            list_of_boundary_labels = [ATR_LABELS["hair"], ATR_LABELS["upper clothes"], ATR_LABELS["background"], ATR_LABELS["dress"], ATR_LABELS["skirt"], ATR_LABELS["pants"], ATR_LABELS["belt"], ATR_LABELS["bag"]]
            mask_img = flood_fill(mask_img, left_arm_points[0], ATR_LABELS["left arm"], [ATR_LABELS["left leg"], ATR_LABELS["right leg"], ATR_LABELS["face"]], list_of_boundary_labels)
        if len(right_arm_points) > 0:
            list_of_boundary_labels = [ATR_LABELS["hair"], ATR_LABELS["upper clothes"], ATR_LABELS["background"], ATR_LABELS["dress"], ATR_LABELS["skirt"], ATR_LABELS["pants"], ATR_LABELS["belt"], ATR_LABELS["bag"]]
            mask_img = flood_fill(mask_img, right_arm_points[0], ATR_LABELS["right arm"], [ATR_LABELS["left leg"], ATR_LABELS["right leg"], ATR_LABELS["face"]], list_of_boundary_labels)
        if len(left_leg_points) > 0:
            list_of_boundary_labels = [ATR_LABELS["hair"], ATR_LABELS["upper clothes"], ATR_LABELS["background"], ATR_LABELS["dress"], ATR_LABELS["skirt"], ATR_LABELS["pants"], ATR_LABELS["belt"], ATR_LABELS["bag"]]
            mask_img = flood_fill(mask_img, left_leg_points[0], ATR_LABELS["left leg"], [ATR_LABELS["left arm"], ATR_LABELS["right arm"]], list_of_boundary_labels)
        if len(right_leg_points) > 0:
            list_of_boundary_labels = [ATR_LABELS["hair"], ATR_LABELS["upper clothes"], ATR_LABELS["background"], ATR_LABELS["dress"], ATR_LABELS["skirt"], ATR_LABELS["pants"], ATR_LABELS["belt"], ATR_LABELS["bag"]]
            mask_img = flood_fill(mask_img, right_leg_points[0], ATR_LABELS["right leg"], [ATR_LABELS["left arm"], ATR_LABELS["right arm"]], list_of_boundary_labels)
        
        for mask_list, label_list in zip(list_of_mask_list, list_of_label_list):
            if len(label_list) == 0:
                value = 0
            elif label_list[0].split('(')[0] == "arm":
                value = ATR_LABELS["right arm"]
            elif label_list[0].split('(')[0] == "leg":
                value = ATR_LABELS["right leg"]
            elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
                value = ATR_LABELS["right shoe"]
            elif label_list[0].split('(')[0] == "neck" or label_list[0].split('(')[0] == "ear":
                value = ATR_LABELS["face"]
            elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
                value = ATR_LABELS["pants"]
            elif label_list[0].split('(')[0] == "maxi skirt":
                value = ATR_LABELS["skirt"]
            elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
                continue
            elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
                value = ATR_LABELS["upper clothes"]
            else:
                value = ATR_LABELS[label_list[0].split('(')[0]]
            for idx, mask in enumerate(mask_list):
                if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                    break
                if label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm" and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot":
                    mask_img[mask.cpu().numpy()[0] == True] = value
                else:
                    mask_img[mask.cpu().numpy()[0] == True] = value - idx
        list_of_boundary_labels = [ATR_LABELS["face"], ATR_LABELS["hair"], ATR_LABELS["background"], ATR_LABELS["belt"], ATR_LABELS["skirt"], ATR_LABELS["pants"]]
        if has_upper_clothes == True:
            if has_belt == True or has_skirt == True:
                if pose_type == "mediapipe":
                    mask_img = flood_fill(mask_img, keypoints[12], ATR_LABELS["upper clothes"], [ATR_LABELS["dress"]], list_of_boundary_labels)
                    mask_img = flood_fill(mask_img, keypoints[11], ATR_LABELS["upper clothes"], [ATR_LABELS["dress"]], list_of_boundary_labels)
                else:
                    mask_img = flood_fill(mask_img, keypoints[5], ATR_LABELS["upper clothes"], [ATR_LABELS["dress"]], list_of_boundary_labels)
                    mask_img = flood_fill(mask_img, keypoints[2], ATR_LABELS["upper clothes"], [ATR_LABELS["dress"]], list_of_boundary_labels)
            if np.any(mask_img == ATR_LABELS["dress"]):
                mask_img[mask_img == ATR_LABELS["dress"]] = ATR_LABELS["skirt"]
    
        mask_img = assign_limbs_points(mask_img, keypoints, ATR_LABELS, "dwpose")
        color_mask_img = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), dtype=np.uint8)
        for label_id, color in enumerate(ATR_PALETTE):
            color_mask_img[mask_img == label_id] = color
    
    else:
        has_upper_clothes = False
        has_belt = False
        has_skirt = False
        for label_list in list_of_label_list:
            if len(label_list) == 0:
                continue
            elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
                has_upper_clothes = True
            elif label_list[0].split('(')[0] == "belt":
                has_belt = True
            elif label_list[0].split('(')[0] == "skirt":
                has_skirt = True
        for mask_list, label_list in zip(list_of_mask_list, list_of_label_list):
            if len(label_list) == 0:
                value = 0
            elif label_list[0].split('(')[0] == "arm":
                continue
            elif label_list[0].split('(')[0] == "leg":
                continue
            elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
                value = AODAI_LABELS["left shoe"]
            elif label_list[0].split('(')[0] == "neck":
                value = AODAI_LABELS["torso skin"]
            elif label_list[0].split('(')[0] == "ear":
                value = AODAI_LABELS["face"]
            elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
                value = AODAI_LABELS["pants"]
            elif label_list[0].split('(')[0] == "maxi skirt":
                value = AODAI_LABELS["skirt"]
            elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
                continue
            elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
                value = AODAI_LABELS["upper clothes"]
            else:
                value = AODAI_LABELS[label_list[0].split('(')[0]]
            for idx, mask in enumerate(mask_list):
                if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                    break
                mask_img[mask.cpu().numpy()[0] == True] = value
        
        list_of_boundary_labels = [AODAI_LABELS["hair"], AODAI_LABELS["upper clothes"], AODAI_LABELS["background"], AODAI_LABELS["scarf"], AODAI_LABELS["dress"]]
        mask_img = flood_fill(mask_img, keypoints[0], AODAI_LABELS["face"], list_of_labels[1:], list_of_boundary_labels)
        if len(left_arm_points) > 0:
            list_of_boundary_labels = [AODAI_LABELS["hair"], AODAI_LABELS["upper clothes"], AODAI_LABELS["background"], AODAI_LABELS["dress"], AODAI_LABELS["skirt"], AODAI_LABELS["pants"]]
            mask_img = flood_fill(mask_img, left_arm_points[0], AODAI_LABELS["left arm"], [AODAI_LABELS["left leg"], AODAI_LABELS["right leg"], AODAI_LABELS["face"]], list_of_boundary_labels)
        if len(right_arm_points) > 0:
            list_of_boundary_labels = [AODAI_LABELS["hair"], AODAI_LABELS["upper clothes"], AODAI_LABELS["background"], AODAI_LABELS["dress"], AODAI_LABELS["skirt"], AODAI_LABELS["pants"]]
            mask_img = flood_fill(mask_img, right_arm_points[0], AODAI_LABELS["right arm"], [AODAI_LABELS["left leg"], AODAI_LABELS["right leg"], AODAI_LABELS["face"]], list_of_boundary_labels)
        if len(left_leg_points) > 0:
            list_of_boundary_labels = [AODAI_LABELS["hair"], AODAI_LABELS["upper clothes"], AODAI_LABELS["background"], AODAI_LABELS["dress"], AODAI_LABELS["skirt"], AODAI_LABELS["pants"]]
            mask_img = flood_fill(mask_img, left_leg_points[0], AODAI_LABELS["left leg"], [AODAI_LABELS["left arm"], AODAI_LABELS["right arm"]], list_of_boundary_labels)
        if len(right_leg_points) > 0:
            list_of_boundary_labels = [AODAI_LABELS["hair"], AODAI_LABELS["upper clothes"], AODAI_LABELS["background"], AODAI_LABELS["dress"], AODAI_LABELS["skirt"], AODAI_LABELS["pants"]]
            mask_img = flood_fill(mask_img, right_leg_points[0], AODAI_LABELS["right leg"], [AODAI_LABELS["left arm"], AODAI_LABELS["right arm"]], list_of_boundary_labels)
        
        for mask_list, label_list in zip(list_of_mask_list, list_of_label_list):
            if len(label_list) == 0:
                value = 0
            elif label_list[0].split('(')[0] == "arm":
                value = AODAI_LABELS["right arm"]
            elif label_list[0].split('(')[0] == "leg":
                value = AODAI_LABELS["right leg"]
            elif label_list[0].split('(')[0] == "shoe" or label_list[0].split('(')[0] == "boot":
                value = AODAI_LABELS["right shoe"]
            elif label_list[0].split('(')[0] == "neck":
                value = AODAI_LABELS["torso skin"]
            elif label_list[0].split('(')[0] == "ear":
                value = AODAI_LABELS["face"]
            elif label_list[0].split('(')[0] == "socks" or label_list[0].split('(')[0] == "shorts":
                value = AODAI_LABELS["pants"]
            elif label_list[0].split('(')[0] == "maxi skirt":
                value = AODAI_LABELS["skirt"]
            elif label_list[0].split('(')[0] == "None" or (has_upper_clothes == False and label_list[0].split('(')[0] == "skirt"):
                continue
            elif label_list[0].split('(')[0] in LIST_OF_UPPER_CLOTHES:
                value = AODAI_LABELS["upper clothes"]
            else:
                value = AODAI_LABELS[label_list[0].split('(')[0]]
            for idx, mask in enumerate(mask_list):
                if (idx > 0 and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot" and label_list[0].split('(')[0] != "socks" and label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm") or idx > 1:
                    break
                if label_list[0].split('(')[0] != "leg" and label_list[0].split('(')[0] != "arm" and label_list[0].split('(')[0] != "shoe" and label_list[0].split('(')[0] != "boot":
                    mask_img[mask.cpu().numpy()[0] == True] = value
                else:
                    mask_img[mask.cpu().numpy()[0] == True] = value - idx
        list_of_boundary_labels = [AODAI_LABELS["face"], AODAI_LABELS["hair"], AODAI_LABELS["background"], AODAI_LABELS["skirt"], AODAI_LABELS["pants"]]
        if has_upper_clothes == True:
            if has_belt == True or has_skirt == True:
                if pose_type == "mediapipe":
                    mask_img = flood_fill(mask_img, keypoints[12], AODAI_LABELS["upper clothes"], [AODAI_LABELS["dress"]], list_of_boundary_labels)
                    mask_img = flood_fill(mask_img, keypoints[11], AODAI_LABELS["upper clothes"], [AODAI_LABELS["dress"]], list_of_boundary_labels)
                else:
                    mask_img = flood_fill(mask_img, keypoints[5], AODAI_LABELS["upper clothes"], [AODAI_LABELS["dress"]], list_of_boundary_labels)
                    mask_img = flood_fill(mask_img, keypoints[2], AODAI_LABELS["upper clothes"], [AODAI_LABELS["dress"]], list_of_boundary_labels)
            if np.any(mask_img == AODAI_LABELS["dress"]):
                mask_img[mask_img == AODAI_LABELS["dress"]] = AODAI_LABELS["skirt"]
    
        mask_img = assign_limbs_points(mask_img, keypoints, AODAI_LABELS, "dwpose")
        color_mask_img = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), dtype=np.uint8)
        for label_id, color in enumerate(AODAI_PALETTE):
            color_mask_img[mask_img == label_id] = color
    
    # for keypoint in left_arm_points:
    #     x, y = keypoint
    #     if x < mask_img.shape[1] and x > 0 and y < mask_img.shape[0] and y > 0:
    #         cv2.circle(color_mask_img, (int(x), int(y)), 4, (255, 0, 0), thickness=-1)
    
    # for keypoint in point_augment([keypoints[7], keypoints[6]]):
    #     x, y = keypoint
    #     if x < mask_img.shape[1] and x > 0 and y < mask_img.shape[0] and y > 0:
    #         cv2.circle(color_mask_img, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

    # for keypoint in [keypoints[11], keypoints[14]]:
    #     x, y = keypoint
    #     if x < mask_img.shape[1] and x > 0 and y < mask_img.shape[0] and y > 0:
    #         cv2.circle(color_mask_img, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
    cv2.imwrite(os.path.join(output_dir, f'{mask_path}_mask.jpg'), cv2.cvtColor(color_mask_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f'{mask_path}.png'), mask_img)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, nargs="+", required=False, help="list of text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    parser.add_argument("--dataset", type=str, default="ATR", required=False, help="dataset type")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_dir_path = args.input_image_dir
    # text_prompt_list = ["Face", "Neck", "Hair", "Arm", "Leg", "Upper clothes", "Hat", "Coat", "Skirt", "Pants", "Socks", "Sunglasses", "Glove", "Dress", "Scarf", "Jumpsuits", "Shoe"]

    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path
    dataset_type = args.dataset
    if dataset_type == "ATR":
        text_prompt_list = ["Face", "Hair", "Ear", "Arm", "Hat", "Leg", "Sunglasses", "Dress", "Skirt", "Maxi skirt" "Shirt", "T-shirt", "Tanktop", "Sweater", "Blouse", "Coat", "Vest", "Pants", "Shorts", "Socks", "Shoe", "Boot", "Belt", "Bag", "Scarf"]
    else:
        text_prompt_list = ["Leg", "Arm", "Pants", "Dress", "Shoe", "Face", "Neck", "Ear", "Hair"]

    # make dir
    os.makedirs(output_dir, exist_ok=True)

	# load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    model = model.to(device)

	# initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    is_skipped = True
    get_pose(image_dir_path, "pose_results/images", "pose_results/keypoints")
    for image_path in natsorted(os.listdir(image_dir_path)):
        if image_path.split('.')[0] in ["2500_20", "2500_24"]:
            is_skipped = False
        else:
            is_skipped = True
        if is_skipped:
            continue
        print(f"Current image is: {image_path}")
        image_output_dir = os.path.join(output_dir, image_path.split('.')[0])
        input_path = os.path.join(image_dir_path, image_path)
        os.makedirs(image_output_dir, exist_ok=True)
        # load image
        image_pil, image = load_image(input_path)

        # visualize raw image
        image_pil.save(os.path.join(image_output_dir, "raw_image.jpg"))
        image2 = cv2.imread(input_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)    
        predictor.set_image(image2)   
        size = image_pil.size
        H, W = size[1], size[0]
        masks_list = []
        boxes_filt_list = []
        pred_phrases_list = []
        for text_prompt in text_prompt_list:
            # run grounding dino model
            if text_prompt in ["Scarf"]:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold + 0.1, text_threshold, device=device
		        )
            elif text_prompt in ["Bag"]:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold + 0.075, text_threshold, device=device
		        )
            elif text_prompt in ["Socks", "Sunglasses"]:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold + 0.05, text_threshold, device=device
		        )
            elif text_prompt in ["Shoe", "Ear"]:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold + 0.025, text_threshold, device=device
		        )
            elif text_prompt in ["Shirt", "T-shirt", "Tanktop", "Sweater", "Blouse", "Coat", "Vest", "Maxi skirt", "Skirt", "Arm"]:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold - 0.1, text_threshold, device=device
		        )
            else:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
		        )
            # Check if boxes_filt is empty before proceeding
            if boxes_filt.size(0) == 0:
                print(f"No boxes found for prompt: {text_prompt}. Skipping...")
                boxes_filt_list.append([])
                pred_phrases_list.append([])
                continue # Skip to the next prompt
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]      
            boxes_filt = boxes_filt.cpu()
            boxes_filt_list.append(boxes_filt)
            pred_phrases_list.append(pred_phrases)
        
        # if image_path.split('.')[0] in ["DKNI0051"]:
        #     boxes_filt_list, pred_phrases_list = nms_filter(boxes_filt_list, pred_phrases_list, cover_threshold=1.0)
        # else:
        if dataset_type != "AODAI":
            boxes_filt_list, pred_phrases_list = nms_filter(boxes_filt_list, pred_phrases_list, cover_threshold=0.8)
        else:
            boxes_filt_list, pred_phrases_list = nms_filter(boxes_filt_list, pred_phrases_list, cover_threshold=1.0)
        for boxes_filt, pred_phrases in zip(boxes_filt_list, pred_phrases_list):
            if len(pred_phrases) > 0 and (pred_phrases[0].split("(")[0] == "arm") and dataset_type != "ATR":
                masks_list.append([])
                continue
            boxes_filt_tensor = torch.Tensor(boxes_filt)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt_tensor, image2.shape[:2]).to(device)        
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
		    )
            masks_list.append(masks)
        # draw output image
        image = image2
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for masks in masks_list:
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for boxes_filt, pred_phrases in zip(boxes_filt_list, pred_phrases_list):
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box, plt.gca(), label)       
        plt.axis('off')
        plt.savefig(
            os.path.join(image_output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
		)
        
        image = cv2.imread(os.path.join(image_output_dir, "grounded_sam_output.jpg"))
        image = cv2.resize(image, (W, H))
        cv2.imwrite(os.path.join(image_output_dir, "grounded_sam_output.jpg"), image)

        mask_img = save_mask_data(image_output_dir, image_path.split('.')[0], masks_list, boxes_filt_list, pred_phrases_list, H, W, dataset_type)

        mask_process(image_output_dir, image_path.split('.')[0], boxes_filt_list, masks_list, pred_phrases_list, predictor, device, mask_img, os.path.join("pose_results/keypoints", f"{image_path.split(".")[0]}_keypoints.json"), dataset_type, "dwpose", False)

