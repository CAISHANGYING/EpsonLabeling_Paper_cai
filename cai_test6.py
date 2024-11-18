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

# 開始處理影片幀
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # 記錄幀數

    frame = cv2.resize(frame, (1120, 630))
    height, width, _ = frame.shape

    # 在影像上顯示當前幀數
    cv2.putText(frame, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 孔洞偵測
    hole_detail, det = hole_detect.detect(frame)
    cv2.imshow('detect', det)

    result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
    hole_convert = result_of_convert['hole']
    wrench_convert = result_of_convert['wrench']

    # 如果偵測到孔洞，則認為拍攝到了照片，並進行計數
    if hole_convert:
        photo_count += 1

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

    # 計算兩個模型在當前幀的顏色次數
    push_back_frame1, color_count_1 = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback1)
    all1 = color_count_1['hole'] + color_count_1['hole_stage2'] + color_count_1['hole_match'] + color_count_1['else']

    push_back_frame2, color_count_2 = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback2)
    all2 = color_count_2['hole'] + color_count_2['hole_stage2'] + color_count_2['hole_match'] + color_count_2['else']

    # 累積計算
    hole_1 += color_count_1['hole']
    hole_stage2_1 += color_count_1['hole_stage2']
    hole_match_1 += color_count_1['hole_match']
    else_1 += color_count_1['else']

    hole_2 += color_count_2['hole']
    hole_stage2_2 += color_count_2['hole_stage2']
    hole_match_2 += color_count_2['hole_match']
    else_2 += color_count_2['else']

    # 顯示推回模型結果並加上幀數和當下偵測的 all1, all2 及累積數據
    cv2.putText(push_back_frame1, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(push_back_frame2, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(push_back_frame1, f'Current Total (Model1): {all1}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(push_back_frame2, f'Current Total (Model2): {all2}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 顯示累積的數量
    cumulative_model1 = hole_1 + hole_stage2_1 + hole_match_1 + else_1
    cumulative_model2 = hole_2 + hole_stage2_2 + hole_match_2 + else_2
    cv2.putText(push_back_frame1, f'Cumulative (Model1): {cumulative_model1}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(push_back_frame2, f'Cumulative (Model2): {cumulative_model2}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

# 繪製兩個模型的顏色計數柱狀圖
plt.figure(figsize=(10, 6))
categories = ['hole', 'hole_stage2', 'hole_match', 'else']
counts_model1 = [hole_1, hole_stage2_1, hole_match_1, else_1]
counts_model2 = [hole_2, hole_stage2_2, hole_match_2, else_2]
bar_width = 0.35

# 設定位置
indices = np.arange(len(categories))

# 繪製柱狀圖
plt.bar(indices, counts_model1, bar_width, label='PushBackModel1', color='blue')
plt.bar(indices + bar_width, counts_model2, bar_width, label='PushBackModel2', color='orange')

# 設定標籤與標題
plt.xlabel('Hole Status')
plt.ylabel('Counts')
plt.title('Counts of Different Hole Statuses for Two Models')
plt.xticks(indices + bar_width / 2, categories)
plt.legend()

# 顯示數值標籤
for i, (count1, count2) in enumerate(zip(counts_model1, counts_model2)):
    plt.text(i, count1 + 1, str(count1), ha='center', va='bottom')
    plt.text(i + bar_width, count2 + 1, str(count2), ha='center', va='bottom')

# 儲存並顯示柱狀圖
plt.savefig('color_counts_comparison_between_models.png')
plt.show()

# 輸出幀數、照片次數、總處理時間、偵測次數和總數
print(f"Total frames processed: {frame_count}")
print(f"Total photos captured: {photo_count}")
print(f"Total time for PushBackModel1: {total_frame_time_model1:.4f} seconds")
print(f"Total time for PushBackModel2: {total_frame_time_model2:.4f} seconds")
print(f"Total detections for PushBackModel1: {detections_model1}")
print(f"Total detections for PushBackModel2: {detections_model2}")
print(f"Total cumulative count for PushBackModel1: {cumulative_model1}")
print(f"Total cumulative count for PushBackModel2: {cumulative_model2}")
