import os
import sys
import json
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from detect.hole_detect_process_cai import HoleDetectConvector
from detect.hole.hole_detect import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back_cai1 import PushBackModel1

sys.path.append(os.getcwd())

# 加載配置
with open('./RemoteServer/remote.json') as f:
    config = json.load(f)

# 初始化模型
hole_detect = HoleDetectModel(config['yolov5'])
with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:
    workpieceJsonData = dict(json.load(f))
stand_workpiece = StandWorkpiece(workpieceJsonData)
push_back_model1 = PushBackModel1(config['push_back'], stand_workpiece)
hole_conv = HoleDetectConvector(movement_threshold=30, large_movement_threshold=50, small_movement_threshold=5)

# 讀取影片
# cap = cv2.VideoCapture('./test_video/9045.mp4')
cap = cv2.VideoCapture('./test_video/show_workpiece_test.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

# 初始化變量
frame_count = 0

def draw_detection_with_fixed_ids(frame, hole_detail):
    frame_with_detections = frame.copy()
    
    # 初次分配 ID，如果還沒有初始化
    if len(push_back_model1._hole_id_map) == 0:
        push_back_model1.initialize_hole_ids(hole_detail)
    
    # 追蹤孔洞
    push_back_model1.track_holes(hole_detail)
    hole_id_map = push_back_model1.get_hole_id_map()

    for hole_id, hole_data in hole_id_map.items():
        # 獲取中心點資料，確保正確的訪問方式
        center_point = hole_data["center"]
        center_point = (int(center_point[0]), int(center_point[1]))

        # 找到對應的孔洞框
        for hole in hole_detail:
            if hole["tag"] == hole_id:
                box = hole["box"]

                # 繪製孔洞的矩形框
                color = (0, 255, 0)  # 綠色框
                top_left = (int(box[0]), int(box[1]))
                bottom_right = (int(box[2]), int(box[3]))
                cv2.rectangle(frame_with_detections, top_left, bottom_right, color, 2)

                # 在中心點顯示ID
                cv2.putText(frame_with_detections, f"{hole_id}", center_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                break

    return frame_with_detections

# 開始處理影片序
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (1120, 630))

    # 孔洞偵測
    hole_detail, det = hole_detect.detect(frame)

    # 根據 hole_detail 回推畫面並標記固定ID
    frame_with_fixed_ids = draw_detection_with_fixed_ids(frame, hole_detail)

    # 合併兩個畫面來同時顯示
    combined_frame = np.hstack((det, frame_with_fixed_ids))

    # 顯示兩個畫面的效果
    cv2.imshow('Original Det vs Hole Detail with Fixed IDs', combined_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()