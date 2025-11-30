from annotator.dwpose import DWposeDetector
import os
import numpy as np
import math
from natsort import natsorted
import json
import cv2
import time

pose = DWposeDetector()

def process_image(image_path, output_image_path, output_json_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    new_width = 368

    new_height = int((new_width / width) * height)

    # Resize ảnh
    image_rgb = cv2.resize(image, (new_width, new_height))
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    out, bodies, process_time = pose(image_rgb)

    if not out.any():
        print(f"Khong co pose trong anh: {image_path}")
        return process_time

    # Lấy keypoints 2D
    candidate = bodies['candidate']
    subset = bodies['subset']
    candidate = np.array(candidate)
    new_subset = []
    new_subset.append(subset[0])
    subset = np.array(new_subset)
    pose_keypoints_2d = []

    midpoint_9_12 = [
        (candidate[8][0] + candidate[11][0]) / 2,
        (candidate[8][1] + candidate[11][1]) / 2
    ]
    candidate = candidate.tolist()
    if len(candidate) < 19:
        candidate.append(midpoint_9_12)
    else:
        candidate.insert(18, midpoint_9_12)
    candidate = np.array(candidate)
    print(candidate)
    
    subset = subset.tolist()
    if subset[0][8] == -1 or subset[0][11] == -1:
        subset[0].append(-1)
    else:
        subset[0].append(18)
    subset = np.array(subset)
    print(subset)

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 19], [9, 19], [9, 10], \
               [10, 11], [12, 19], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 85, 255]]

    canvas = np.zeros_like(image)
    stickwidth = 4

    for i in range(18):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(width)
            X = candidate[index.astype(int), 1] * float(height)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    # for i in range(19):
    #     for n in range(len(subset)):
    #         index = int(subset[n][i])
    #         if index == -1:
    #             if i < 18 and i != 1:
    #                 pose_keypoints_2d.extend([0.0, 0.0])
    #             continue
    #         x, y = candidate[index][0:2]
    #         x = int(x * width)
    #         y = int(y * height)
    #         if i < 18 and i != 1:
    #             pose_keypoints_2d.extend([candidate[index][0] * width, candidate[index][1] * height])
    #         cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    
    for i in range(19):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                # print(f"Invalid point: {i}, {candidate[index][0]}, {candidate[index][1]}")
                # if candidate[i][1] < 1.0:
                #    candidate[i][1] = 2.0
                # pose_keypoints_2d.extend([candidate[i][0] * width, candidate[i][1] * height])
                pose_keypoints_2d.extend([0.0, 0.0])
                continue
            x, y = candidate[index][0:2]
            x = int(x * width)
            y = int(y * height)
            pose_keypoints_2d.extend([candidate[index][0] * width, candidate[index][1] * height])
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    mid_hip = pose_keypoints_2d[-2:]
    second_part = pose_keypoints_2d[16:]
    pose_keypoints_2d = pose_keypoints_2d[:16]
    pose_keypoints_2d.extend(mid_hip)
    pose_keypoints_2d.extend(second_part[:-2])
    cv2.imwrite(output_image_path, canvas)

    # for landmark in candidate:
    #     pose_keypoints_2d.extend([landmark[0] * width, landmark[1] * height])

    pose_data = {
        "people": [
            {
                "pose_keypoints_2d": pose_keypoints_2d
            }
        ]
    }

    with open(output_json_path, 'w') as json_file:
        json.dump(pose_data, json_file, indent=4)
    
    print(f"Thoi gian xu ly anh {image_path} la: {process_time}ms")
    return process_time


def get_pose(input_directory, output_image_directory, output_keypoint_directory):
    # input_directory = 'ATR_images'
    # output_image_directory = 'ATR_results/images'
    # output_keypoint_directory ='ATR_results/keypoints'

    os.makedirs(output_image_directory, exist_ok=True)
    os.makedirs(output_keypoint_directory, exist_ok=True)

    total_time = 0
    for i, filename in enumerate(natsorted(os.listdir(input_directory))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            input_path = os.path.join(input_directory, filename)
            output_image_path = os.path.join(output_image_directory, f"{os.path.splitext(filename)[0]}_rendered.png")
            output_json_path = os.path.join(output_keypoint_directory, f"{os.path.splitext(filename)[0]}_keypoints.json")

            process_time = process_image(input_path, output_image_path, output_json_path)
            if i > 0:
                total_time += process_time
    
    average_time = total_time / (len(os.listdir(input_directory)) - 1)
    print(f"Thoi gian chay 1 anh: {average_time:.2f}ms")

