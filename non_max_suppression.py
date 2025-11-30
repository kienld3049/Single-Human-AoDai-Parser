import numpy as np
import re

def iou(box1, box2):
    """Tính Intersection over Union (IoU) giữa 2 bounding box"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Tính tọa độ giao nhau
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Tính diện tích vùng giao
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Tính diện tích của 2 box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Tính IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def is_box_covering_others(big_box, other_boxes, cover_threshold=0.9):
    """
    Kiểm tra xem hộp lớn có bao trùm phần lớn diện tích của các hộp còn lại không.
    - cover_threshold: Ngưỡng diện tích (tỷ lệ) để xác định bao trùm.
    """
    x1_big, y1_big, x2_big, y2_big = big_box
    big_area = (x2_big - x1_big) * (y2_big - y1_big)

    covered_count = 0
    invalid_boxes = 0
    for box in other_boxes:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)

        x1g, y1g, x2g, y2g = big_box

        # Tính tọa độ giao nhau
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
    
        # Tính diện tích vùng giao
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        if inter_area / box_area >= cover_threshold:
            covered_count += 1
        if inter_area == 0:
            invalid_boxes += 1

    return covered_count >= (len(other_boxes) - invalid_boxes) * cover_threshold

def parse_label_score(label_score):
    """Tách nhãn và score từ chuỗi dạng 'label(score)'"""
    match = re.match(r"(.+)\(([\d.]+)\)", label_score)
    if match:
        label, score = match.groups()
        return label, float(score)
    return None, 0.0

def non_max_suppression(boxes, scores, labels, iou_threshold=0.5):
    """Áp dụng NMS trên toàn bộ hộp từ tất cả các nhãn"""
    if len(boxes) == 0:
        return [], [], []

    # Chuyển danh sách thành NumPy array
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    # Sắp xếp theo độ tin cậy giảm dần
    indices = np.argsort(scores)[::-1]
    selected_boxes = []
    selected_scores = []
    selected_labels = []

    while len(indices) > 0:
        best_idx = indices[0]
        best_box = boxes[best_idx]
        selected_boxes.append(best_box)
        selected_scores.append(scores[best_idx])
        selected_labels.append(labels[best_idx])

        # Tính IoU giữa box tốt nhất và các box còn lại
        remaining_boxes = boxes[indices[1:]]
        ious = np.array([iou(best_box, box) for box in remaining_boxes])

        # Giữ lại các box có IoU nhỏ hơn ngưỡng
        indices = indices[1:][ious < iou_threshold]

    return selected_boxes, selected_scores, selected_labels

def nms_filter(boxes_filt_list, pred_phrases_list, cover_threshold=0.9):
    # Gộp tất cả hộp lại thành danh sách chung
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for boxes, phrases in zip(boxes_filt_list, pred_phrases_list):
        for box, phrase in zip(boxes, phrases):
            label, score = parse_label_score(phrase)
            if label is not None:
                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(label)
    
    # Áp dụng NMS trên toàn bộ hộp
    iou_threshold = 0.7
    filtered_boxes, filtered_scores, filtered_labels = non_max_suppression(all_boxes, all_scores, all_labels, iou_threshold)

    if filtered_boxes:
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in filtered_boxes]
        max_area_idx = np.argmax(areas)
        max_box = filtered_boxes[max_area_idx]

        remaining_boxes = filtered_boxes[:max_area_idx] + filtered_boxes[max_area_idx + 1:]
        if is_box_covering_others(max_box, remaining_boxes, cover_threshold):
            del filtered_boxes[max_area_idx]
            del filtered_scores[max_area_idx]
            del filtered_labels[max_area_idx]

    # Phân loại lại hộp sau khi lọc
    filtered_boxes_filt_list = []
    filtered_pred_phrases_list = []
    
    # Tạo dictionary để gom hộp theo nhãn
    filtered_dict = {}
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        if label not in filtered_dict:
            filtered_dict[label] = []
        filtered_dict[label].append((box, score))
    
    # Phân loại lại theo danh sách ban đầu
    for original_labels in pred_phrases_list:
        temp_boxes = []
        temp_phrases = []
        
        for label_score in original_labels:
            label, _ = parse_label_score(label_score)
            if label in filtered_dict and len(filtered_dict[label]) > 0:
                box, score = filtered_dict[label].pop(0)
                temp_boxes.append(box)
                temp_phrases.append(f"{label}({score:.2f})")
        
        filtered_boxes_filt_list.append(temp_boxes)
        filtered_pred_phrases_list.append(temp_phrases)

    # Danh sách nhãn cần sắp xếp từ trái sang phải
    sort_labels = {"shoe", "arm", "leg"}
    
    # Tạo danh sách riêng cho các hộp cần sắp xếp và các hộp khác
    sorted_boxes = []
    sorted_phrases = []
    other_boxes = []
    other_phrases = []
    
    for boxes, phrases in zip(filtered_boxes_filt_list, filtered_pred_phrases_list):
        temp_sorted = []
        temp_other = []
        
        for box, phrase in zip(boxes, phrases):
            label, _ = parse_label_score(phrase)
            if label in sort_labels:
                temp_sorted.append((box, phrase))
            else:
                temp_other.append((box, phrase))
        
        # Sắp xếp các hộp cần sắp xếp theo x1 tăng dần
        temp_sorted.sort(key=lambda x: x[0][0])
    
        # Lưu kết quả
        sorted_boxes.append([b[0] for b in temp_sorted] + [b[0] for b in temp_other])
        sorted_phrases.append([b[1] for b in temp_sorted] + [b[1] for b in temp_other])
    
    return sorted_boxes, sorted_phrases
