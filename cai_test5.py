import os
import sys
import time
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from detect.hole_detect_process_cai import HoleDetectConvector
from detect.hole.hole_detect import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel
from detect.push_back.hole_push_back_cai import PushBackModel1

sys.path.append(os.getcwd())

# 加載配置
with open('./RemoteServer/remote.json') as f:
    config = json.load(f)

# 初始化模型
hole_detect = HoleDetectModel(config['yolov5'])
with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:
    workpieceJsonData = dict(json.load(f))
stand_workpiece = StandWorkpiece(workpieceJsonData)
push_back_model1 = PushBackModel(config['push_back'], stand_workpiece)
push_back_model2 = PushBackModel1(config['push_back'], stand_workpiece)

# 設定參數
cap = cv2.VideoCapture('./test_video/9045.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

# 初始化變數
start_time = time.time()
processing_times_model1 = []
processing_times_model2 = []
timestamps = []
detections_model1 = 0
detections_model2 = 0
total_frame_time_model1 = 0
total_frame_time_model2 = 0

photo_count = 0
frame_count = 0

hole_1 = hole_stage2_1 = hole_match_1 = else_1 = 0
hole_2 = hole_stage2_2 = hole_match_2 = else_2 = 0

# 新增Kalman濾波器和移動平均的參數
hole_positions_buffer = []
max_buffer_length = 5
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4) * 0.03

previous_position = None

def smooth_positions(new_position, buffer, max_length=5):
    buffer.append(new_position)
    if len(buffer) > max_length:
        buffer.pop(0)
    smoothed_position = np.mean(buffer, axis=0)
    return smoothed_position

def kalman_smooth(kalman, new_position):
    measured = np.array([[np.float32(new_position[0])], [np.float32(new_position[1])]])
    kalman.correct(measured)
    prediction = kalman.predict()
    return (int(prediction[0]), int(prediction[1]))

def limit_detection_to_roi(frame, position, roi_size=50):
    if position:
        x, y = position
        roi = frame[max(0, y-roi_size):min(y+roi_size, frame.shape[0]), 
                    max(0, x-roi_size):min(x+roi_size, frame.shape[1])]
        return roi
    return frame

def check_position_consistency(current_position, previous_position, threshold_distance=30):
    if previous_position is not None:
        distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
        if distance > threshold_distance:
            return previous_position
    return current_position

# 開始處理影片幀
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (1120, 630))
    height, width, _ = frame.shape

    cv2.putText(frame, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    hole_detail, det = hole_detect.detect(frame)
    cv2.imshow('detect', det)

    result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
    hole_convert = result_of_convert['hole']
    wrench_convert = result_of_convert['wrench']

    if hole_convert:
        photo_count += 1
        first_hole = hole_convert[0]  # 提取第一個孔洞（假設為列表的第一個元素）
        if 'x' in first_hole and 'y' in first_hole:
            hole_position = (first_hole['x'], first_hole['y'])

            # 平滑處理
            smoothed_position = smooth_positions(hole_convert, hole_positions_buffer, max_buffer_length)
            kalman_position = kalman_smooth(kalman, smoothed_position)
            roi_frame = limit_detection_to_roi(frame, kalman_position)
            stable_position = check_position_consistency(kalman_position, previous_position)
            previous_position = stable_position

            print(f"Stable position: {stable_position}")

    # 計算推回位置（使用之前的模型邏輯）
    start_time_model1 = time.time()
    result_of_pushback1 = push_back_model1.getPushbackPosition(hole_convert, 1.0)
    processing_time_model1 = time.time() - start_time_model1
    processing_times_model1.append(processing_time_model1)
    total_frame_time_model1 += processing_time_model1
    detections_model1 += 1

    start_time_model2 = time.time()
    result_of_pushback2 = push_back_model2.getPushbackPosition(hole_convert, 1.0)
    processing_time_model2 = time.time() - start_time_model2
    processing_times_model2.append(processing_time_model2)
    total_frame_time_model2 += processing_time_model2
    detections_model2 += 1

    timestamps.append(time.time() - start_time)

    # 顯示偵測結果並加上幀數和累積數據
    cv2.putText(frame, f'Cumulative (Model1): {hole_1}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('test result Model1', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 顯示圖表
plt.figure(figsize=(10, 6))
plt.plot(timestamps, processing_times_model1, label='PushBackModel1', marker='o')
plt.plot(timestamps, processing_times_model2, label='PushBackModel2', marker='x')
plt.title("Push Back Calculation Time Over Video")
plt.xlabel("Time (seconds)")
plt.ylabel("Processing Time (seconds)")
plt.legend()
plt.savefig('processing_times_comparison.png')
plt.show()

print(f"Total frames processed: {frame_count}")
print(f"Total photos captured: {photo_count}")
print(f"Total time for PushBackModel1: {total_frame_time_model1:.4f} seconds")
print(f"Total time for PushBackModel2: {total_frame_time_model2:.4f} seconds")
print(f"Total detections for PushBackModel1: {detections_model1}")
print(f"Total detections for PushBackModel2: {detections_model2}")
