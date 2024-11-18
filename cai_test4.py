import os
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from detect.hole.hole_detect import HoleDetectModel
from detect.hole_detect_process import HoleDetectConvector
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back_cai import PushBackModel
import json

# Set OMP_NUM_THREADS to avoid KMeans memory leak
os.environ["OMP_NUM_THREADS"] = "1"

# Load configuration file
try:
    with open('./RemoteServer/remote.json') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit()

# Initialize models
hole_detect = HoleDetectModel(config['yolov5'])
try:
    with open(os.path.join(config['stand']['parent_foleder'], f"{config['stand']['stand_tool_name']}.json")) as f:
        workpieceJsonData = json.load(f)
except Exception as e:
    print(f"Error loading workpiece data: {e}")
    sys.exit()

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
    kmeans = KMeans(n_clusters=4, random_state=0).fit(points)
    centers = kmeans.cluster_centers_.astype(int)
    sorted_centers = sort_points(centers)
    return sorted_centers

# Apply perspective transformation
def apply_perspective_transform(img, points):
    src_points = np.array(points, dtype=np.float32)
    dst_points = np.array([[0, 0], [0, 400], [400, 400], [400, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_img = cv2.warpPerspective(img, matrix, (400, 400))
    return transformed_img, matrix

# Calculate weighted points based on distance from the center
def compute_weighted_points(points, height, width):
    center = np.array([width // 2, height // 2])
    distances = np.linalg.norm(points - center, axis=1)
    max_distance = np.sqrt((width // 2) ** 2 + (height // 2) ** 2)
    weights = 1 - (distances / max_distance)
    weighted_points = points * weights[:, np.newaxis]
    return weighted_points

# Inverse perspective transformation with weighting
def inverse_perspective_transform_with_weights(points, matrix, height, width):
    inv_matrix = np.linalg.inv(matrix)
    original_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), inv_matrix)
    original_points = original_points.reshape(-1, 2)
    weighted_points = compute_weighted_points(original_points, height, width)
    return weighted_points

# Check the distances between points
def check_distances(points, min_distance=50):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < min_distance:
                return False
    return True

# Open the video file
cap = cv2.VideoCapture('./test_video/paper_test05.mp4')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1120, 630))

    # Detect holes in the original frame
    height, width, _ = frame.shape
    hole_detail, det_original = hole_detect.detect(frame)
    result_of_convert_original = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
    hole_convert_original = result_of_convert_original['hole']
    result_of_pushback_original = push_back.getPushbackPosition(hole_convert_original, 1.0)
    push_back_frame_original = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback_original)

    cv2.imshow('Original Video - Hole Detection', push_back_frame_original)

    # Convert frame to HSV for perspective transform
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
            warped_frame, matrix = apply_perspective_transform(frame, sorted_centers)
            cv2.imshow("Warped Perspective", warped_frame)

            # Detect holes in the warped frame
            height, width, _ = warped_frame.shape
            hole_detail, det_warped = hole_detect.detect(warped_frame)
            result_of_convert_warped = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
            hole_convert_warped = result_of_convert_warped['hole']
            result_of_pushback_warped = push_back.getPushbackPosition(hole_convert_warped, 1.0)
            push_back_frame_warped = HoleDetectConvector.getPushbackFrame(warped_frame, result_of_pushback_warped)

            cv2.imshow('Warped Perspective - Hole Detection', push_back_frame_warped)

            # 比較透視變換前後的偵測結果
            if 'holes' in det_warped:
                detected_points = np.array(det_warped['holes'], dtype=np.float32)
                original_points = inverse_perspective_transform_with_weights(detected_points, matrix, height, width)

                # 确保 original_points 是有效的
                if original_points.size > 0:
                    # 在原始影像上標註透視變換後映射回的孔洞
                    for point in original_points:
                        # 确保点在图像内
                        if (0 <= point[0] < width) and (0 <= point[1] < height):
                            cv2.circle(push_back_frame_original, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # 用红色标记变换后的孔洞
                        else:
                            print(f"Point {point} is out of bounds!")

            # 顯示對比結果
            cv2.imshow('Original Video - Hole Detection (After Perspective)', push_back_frame_original)

    # Press 'q' to exit the loop
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
