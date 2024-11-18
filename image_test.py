import os 
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import cv2

# 設置 matplotlib 和其他環境參數
matplotlib.use('TkAgg')
matplotlib.rcParams['backend'] = 'TkAgg'
plt.rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.getcwd())

from detect.hole_detect_process import HoleDetectConvector
from detect.hole.hole_detect import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel
from detect.push_back.hole_push_back_cai import PushBackModel1

# 加載配置文件
with open('./RemoteServer/remote_0531.json') as f:
    config = json.load(f)

# 初始化檢測模型
hole_detect = HoleDetectModel(config['yolov5'])
with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:
    workpieceJsonData = dict(json.load(f))
stand_workpiece = StandWorkpiece(workpieceJsonData)

# 初始化兩個推回模型
push_back_model1 = PushBackModel(config['push_back'], stand_workpiece)
push_back_model2 = PushBackModel1(config['push_back'], stand_workpiece)

# 加載測試圖像
frame = cv2.imread('./test_image/boundary_test_11.jpg')
frame = cv2.resize(frame, (1440, 810))

# 計時開始，並設置時間存儲列表
start_time = time.time()
processing_times_model1 = []
processing_times_model2 = []
timestamps = []

# 記錄每個模型的偵測次數
detections_model1 = 0
detections_model2 = 0

cv2.imshow('detect', frame)
total_frame_time_model1 = 0
total_frame_time_model2 = 0

# 開始處理循環
while True:
    current_time = time.time()
    if current_time - start_time > 300:  # 300秒後停止
        break

    # 檢測孔洞
    height, width, _ = frame.shape
    hole_detail, det = hole_detect.detect(frame)
    cv2.imshow('detect', det)

    result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
    hole_convert = result_of_convert['hole']
    boundary_convert = list(result_of_convert['boundary'])
    wrench_convert = result_of_convert['wrench']

    # 交替使用兩個模型
    if boundary_convert != []:
        stand_boundary = list(stand_workpiece.getBoundaryDetail())
        stand_boundary = HoleDetectConvector.boundary_convert(stand_boundary, [stand_workpiece.getShape()[1], stand_workpiece.getShape()[0]], (width, height))
        matrix, hole_transform = HoleDetectConvector.holePerspectiveTransform(hole_convert, boundary_convert, np.array(stand_boundary, dtype=np.float32))

        # 測試 PushBackModel1
        start_time_model1 = time.time()
        result_of_pushback1 = push_back_model1.getPushbackPosition(hole_transform, 1.0, matrix=matrix)
        processing_time_model1 = time.time() - start_time_model1
        processing_times_model1.append(processing_time_model1)
        total_frame_time_model1 += processing_time_model1
        detections_model1 += 1  # 偵測次數加1

        # 測試 PushBackModel2
        start_time_model2 = time.time()
        result_of_pushback2 = push_back_model2.getPushbackPosition(hole_transform, 1.0, matrix=matrix)
        processing_time_model2 = time.time() - start_time_model2
        processing_times_model2.append(processing_time_model2)
        total_frame_time_model2 += processing_time_model2
        detections_model2 += 1  # 偵測次數加1
    else:
        # 測試 PushBackModel1
        start_time_model1 = time.time()
        result_of_pushback1 = push_back_model1.getPushbackPosition(hole_convert, 1.0)
        processing_time_model1 = time.time() - start_time_model1
        processing_times_model1.append(processing_time_model1)
        total_frame_time_model1 += processing_time_model1
        detections_model1 += 1  # 偵測次數加1

        # 測試 PushBackModel2
        start_time_model2 = time.time()
        result_of_pushback2 = push_back_model2.getPushbackPosition(hole_convert, 1.0)
        processing_time_model2 = time.time() - start_time_model2
        processing_times_model2.append(processing_time_model2)
        total_frame_time_model2 += processing_time_model2
        detections_model2 += 1  # 偵測次數加1

    timestamps.append(current_time - start_time)

    # 顯示結果
    push_back_frame1, _, _, _3 = HoleDetectConvector.getPushbackFrameEachStep(frame, result_of_pushback1)
    push_back_frame2, _, _, _4 = HoleDetectConvector.getPushbackFrameEachStep(frame, result_of_pushback2)
    cv2.imshow('test result Model1', push_back_frame1)
    cv2.imshow('test result Model2', push_back_frame2)

    # 按 'q' 退出
    if cv2.waitKey(100) == ord('q'):
        break

# 關閉所有窗口
cv2.destroyAllWindows()

# 繪製處理時間圖表
plt.plot(timestamps, processing_times_model1, label='PushBackModel1', marker='o')
plt.plot(timestamps, processing_times_model2, label='PushBackModel2', marker='x')
plt.title("Push Back Calculation Time Over 10 Minutes")
plt.xlabel("Time (seconds)")
plt.ylabel("Processing Time (seconds)")
plt.legend()
plt.savefig('processing_times_comparison.png')

# 輸出總處理時間和偵測次數
print(f"Total time for PushBackModel1: {total_frame_time_model1:.4f} seconds")
print(f"Total time for PushBackModel2: {total_frame_time_model2:.4f} seconds")
print(f"Total detections for PushBackModel1: {detections_model1}")
print(f"Total detections for PushBackModel2: {detections_model2}")
