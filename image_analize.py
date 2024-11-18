import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from detect.hole.hole_detect import HoleDetectModel
from detect.hole_detect_process import HoleDetectConvector
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel
import json
import time
import matplotlib.pyplot as plt

# Load configuration file
with open('./RemoteServer/remote.json') as f:
    config = json.load(f)

# Initialize models
hole_detect = HoleDetectModel(config['yolov5'])
with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:
    workpieceJsonData = dict(json.load(f))
stand_workpiece = StandWorkpiece(workpieceJsonData)
push_back = PushBackModel(config['push_back'], stand_workpiece)

# Sort points for perspective transform
def sort_points(points):
    points = sorted(points, key=lambda p: p[1])
    top_points = sorted(points[:2], key=lambda p: p[0])
    bottom_points = sorted(points[2:], key=lambda p: p[0])
    return [top_points[0], bottom_points[0], bottom_points[1], top_points[1]]

# Draw contours and find centers for perspective transformation
def draw_contours(img, contours):
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    points = np.array([point[0] for point in hull])
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(points)
    centers = kmeans.cluster_centers_.astype(int)
    sorted_centers = sort_points(centers)
    return sorted_centers

# Apply perspective transformation
def apply_perspective_transform(img, points):
    src_points = np.array(points, dtype=np.float32)
    dst_points = np.array([[0, 0], [0, 400], [400, 400], [400, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_img = cv2.warpPerspective(img, matrix, (400, 400))
    return transformed_img

# Check the distances between points
def check_distances(points, min_distance=50):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < min_distance:
                return False
    return True

# 計算 IOU（Intersection Over Union）
# 修改 calculate_iou 前先添加一些調試輸出
def calculate_iou(box1, box2):
    print(f"Original Box: {box1}, Warped Box: {box2}")  # 調試輸出
    
    # 確認 box1 和 box2 是四個元素的列表
    if len(box1) < 4 or len(box2) < 4:
        print("Error: Box does not contain four elements.")
        return 0  # 如果 box1 或 box2 不是四個值，返回 0 的 IOU
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection
    iou = intersection / union if union != 0 else 0
    return iou


# 計算中心距離
def calculate_center_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    return distance

# 初始化指標數據
iou_list_original = []
iou_list_warped = []
center_distance_original = []
center_distance_warped = []
processing_time_original = []
processing_time_warped = []

# Open the video file
cap = cv2.VideoCapture('./test_video/paper_test02.mp4')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1120, 630))

    ### 原始影像處理 ###
    start_time = time.time()  # 記錄處理時間
    # Detect holes in the original frame
    height, width, _ = frame.shape
    hole_detail_original, det_original = hole_detect.detect(frame)
    result_of_convert_original = HoleDetectConvector.convert(hole_detail_original, (width, height), (width, height))
    hole_convert_original = result_of_convert_original['hole']
    result_of_pushback_original = push_back.getPushbackPosition(hole_convert_original, 1.0)
    push_back_frame_original = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback_original)

    # 顯示原始偵測結果
    cv2.imshow('Original Video - Hole Detection', push_back_frame_original)

    # 記錄處理時間
    processing_time_original.append(time.time() - start_time)

    ### 應用透視轉換處理 ###
    start_time = time.time()  # 記錄處理時間
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 30
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if filtered_contours:
        sorted_centers = draw_contours(frame, filtered_contours)

        if sorted_centers and check_distances(sorted_centers):
            # Apply perspective transformation
            warped_frame = apply_perspective_transform(frame, sorted_centers)
            cv2.imshow("Warped Perspective", warped_frame)

            # Detect holes in the warped frame
            height, width, _ = warped_frame.shape
            hole_detail_warped, det_warped = hole_detect.detect(warped_frame)
            result_of_convert_warped = HoleDetectConvector.convert(hole_detail_warped, (width, height), (width, height))
            hole_convert_warped = result_of_convert_warped['hole']
            result_of_pushback_warped = push_back.getPushbackPosition(hole_convert_warped, 1.0)
            push_back_frame_warped = HoleDetectConvector.getPushbackFrame(warped_frame, result_of_pushback_warped)

            # 顯示透視轉換後偵測結果
            cv2.imshow('Warped Perspective - Hole Detection', push_back_frame_warped)

            # 記錄處理時間
            processing_time_warped.append(time.time() - start_time)

            # 比較 IOU 和中心距離
            for original_box, warped_box in zip(hole_convert_original, hole_convert_warped):
                iou_list_original.append(calculate_iou(original_box, warped_box))
                center_distance_original.append(calculate_center_distance(original_box, warped_box))

    # Press 'q' to exit the loop
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 繪製比較結果
x = np.arange(len(processing_time_original))

plt.figure(figsize=(12, 8))

# 處理時間比較
plt.subplot(2, 2, 1)
plt.plot(x, processing_time_original, 'b-', label="Original Processing Time")
plt.plot(x, processing_time_warped, 'r-', label="Warped Processing Time")
plt.legend()
plt.title("Processing Time Comparison")

# IOU 比較
plt.subplot(2, 2, 2)
plt.plot(x, iou_list_original, 'b-', label="IOU Comparison")
plt.legend()
plt.title("IOU Comparison")

# 中心距離比較
plt.subplot(2, 2, 3)
plt.plot(x, center_distance_original, 'g-', label="Center Distance Comparison")
plt.legend()
plt.title("Center Distance Comparison")

plt.tight_layout()
plt.show()
