import os
import sys
import json
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from detect.hole_detect_process_cai import HoleDetectConvector
from detect.hole.hole_detect import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back_cai2 import PushBackModel

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
hole_conv = HoleDetectConvector(movement_threshold=30, large_movement_threshold=50, small_movement_threshold=5)

# 讀取影片
cap = cv2.VideoCapture('./test_video/show_workpiece_test.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

# 初始化變數
frame_count = 0

# 定義函數根據 hole_detail 繪製偵測結果並標記固定ID
def draw_detection_with_fixed_ids(frame, hole_detail):
    frame_with_detections = frame.copy()

    # 初始化 PushBackModel，並初始化孔洞ID
    push_back_model1.initialize_hole_positions(hole_detail)

    # 呼叫 process_frame 函數處理每一幀並返回結果
    frame_with_fixed_ids = push_back_model1.process_frame(frame, hole_detail)

    return frame_with_fixed_ids


# 開始處理影片幀
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (1120, 630))
    height, width, _ = frame.shape

    # 孔洞偵測
    hole_detail, det = hole_detect.detect(frame)

    # 根據 hole_detail 回推畫面並標記固定ID
    frame_with_fixed_ids = draw_detection_with_fixed_ids(frame, hole_detail)

    result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
    hole_convert = result_of_convert['hole']
    print("hole Detail:", hole_convert)

    # 合併兩個畫面來同時顯示
    combined_frame = np.hstack((det, frame_with_fixed_ids))

    # 顯示兩個畫面的效果
    cv2.imshow('Original Det vs Hole Detail with Fixed IDs', combined_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
