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

# 移動閾值設定
MIN_MOVEMENT_THRESHOLD = 5  # 移動幅度過小的閾值
MAX_MOVEMENT_THRESHOLD = 50  # 移動幅度過大的閾值

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

# 讀取影片
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

# 照片與幀計數初始化
photo_count = 0
frame_count = 0

# 顏色計數初始化 (累積計算)
hole_1 = hole_stage2_1 = hole_match_1 = else_1 = 0
hole_2 = hole_stage2_2 = hole_match_2 = else_2 = 0

# 定義移動幅度檢查函數
def calculate_center_distance(prev_hole, current_hole):
    if not prev_hole or not current_hole:
        return 0, "normal"

    if 'coordinate' in prev_hole and 'coordinate' in current_hole:
        prev_center = np.array(prev_hole['coordinate']['middle'])
        current_center = np.array(current_hole['coordinate']['middle'])
    else:
        print("警告：缺少坐標資料")
        return 0, "normal"

    distance = np.linalg.norm(current_center - prev_center)

    if distance < MIN_MOVEMENT_THRESHOLD:
        return distance, "too small"
    elif distance > MAX_MOVEMENT_THRESHOLD:
        return distance, "too large"
    return distance, "normal"

# 定義移動幅度檢查函數
def calculate_center_distance(prev_hole, current_hole):
    # 如果 prev_hole 或 current_hole 为空或结构不符合预期，则直接返回
    if not prev_hole or not current_hole:
        return 0, "normal"

    # 檢查 prev_hole 和 current_hole 是否包含 'coordinate' 鍵
    if 'coordinate' in prev_hole and 'coordinate' in current_hole:
        # 取出中心點
        prev_center = np.array(prev_hole['coordinate']['middle'])
        current_center = np.array(current_hole['coordinate']['middle'])
    else:
        # 如果沒有坐標資料，返回距離 0 並視為正常
        print("警告：缺少坐標資料")
        return 0, "normal"

    # 計算兩個中心點之間的距離
    distance = np.linalg.norm(current_center - prev_center)

    # 判斷移動幅度
    if distance < MIN_MOVEMENT_THRESHOLD:
        return distance, "too small"
    elif distance > MAX_MOVEMENT_THRESHOLD:
        return distance, "too large"
    return distance, "normal"

# 開始處理影片幀
prev_hole_detail = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # 記錄幀數

    frame = cv2.resize(frame, (1120, 630))
    height, width, _ = frame.shape

    # 孔洞偵測
    hole_detail, det = hole_detect.detect(frame)
    result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
    hole_convert = result_of_convert['hole']
    wrench_convert = result_of_convert['wrench']

    hole_detect_convector = HoleDetectConvector()

    # 檢查 `hole_convert` 是否為空，並在非空的情況下進行計算
    if hole_convert:
        if prev_hole_detail:
            distance, movement_status = calculate_center_distance(prev_hole_detail, hole_convert)
            
            # 根據移動幅度進行不同處理
            if movement_status == "too small":
                # 小幅移動，使用平滑更新
                hole_convert = hole_detect_convector.update_with_smoothing(hole_convert)
            elif movement_status == "too large":
                # 大幅移動，直接使用當前偵測結果
                pass
            else:
                # 正常移動，進行一般平滑處理
                hole_convert = hole_detect_convector.update_with_smoothing(hole_convert)

        # 更新前一幀的孔洞資料
        prev_hole_detail = hole_convert.copy()
    else:
        print("未檢測到孔洞，跳過此幀")
        prev_hole_detail = None

    # 計算 PushBackModel1 的推回位置
    start_time_model1 = time.time()
    result_of_pushback1 = push_back_model1.getPushbackPosition(hole_convert, 1.0)
    processing_time_model1 = time.time() - start_time_model1
    processing_times_model1.append(processing_time_model1)
    total_frame_time_model1 += processing_time_model1
    detections_model1 += 1

    # 計算 PushBackModel2 的推回位置
    start_time_model2 = time.time()
    result_of_pushback2 = push_back_model2.getPushbackPosition(hole_convert, 1.0)
    processing_time_model2 = time.time() - start_time_model2
    processing_times_model2.append(processing_time_model2)
    total_frame_time_model2 += processing_time_model2
    detections_model2 += 1

    timestamps.append(time.time() - start_time)

    # 顯示錯誤訊息及兩個模型的結果
    wrench_wrong_screw_index = HoleDetectConvector.wrenchOnWrongScrewPosition(wrench_convert, result_of_pushback1, "1")
    print("wrong:", wrench_wrong_screw_index)

    # 顯示推回模型結果
    push_back_frame1, _ = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback1)
    push_back_frame2, _ = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback2)

    cv2.imshow('test result Model1', push_back_frame1)
    cv2.imshow('test result Model2', push_back_frame2)

    # 按 'q' 退出
    if cv2.waitKey(1) == ord('q'):
        break

# 關閉影片資源
cap.release()
cv2.destroyAllWindows()

# 繪製處理時間圖表
plt.figure(figsize=(10, 6))
plt.plot(timestamps, processing_times_model1, label='PushBackModel1', marker='o')
plt.plot(timestamps, processing_times_model2, label='PushBackModel2', marker='x')
plt.title("Push Back Calculation Time Over Video")
plt.xlabel("Time (seconds)")
plt.ylabel("Processing Time (seconds)")
plt.legend()
plt.savefig('processing_times_comparison.png')
plt.show()
