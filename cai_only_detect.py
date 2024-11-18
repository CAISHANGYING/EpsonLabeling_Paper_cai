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
hole_conv = HoleDetectConvector(movement_threshold=30, large_movement_threshold=50, small_movement_threshold=5)

# 讀取影片
cap = cv2.VideoCapture('./test_video/9045.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

frame_count = 0

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

    # 按 'q' 退出
    if cv2.waitKey(1) == ord('q'):
        break

# 關閉影片資源
cap.release()
cv2.destroyAllWindows()





