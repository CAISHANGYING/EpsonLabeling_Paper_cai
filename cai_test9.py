import os
import sys
import json
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from detect.hole_detect_process_cai import HoleDetectConvector
from detect.hole.hole_detect import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel
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
push_back_model1 = PushBackModel(config['push_back'], stand_workpiece)
hole_conv = HoleDetectConvector(movement_threshold=30, large_movement_threshold=50, small_movement_threshold=5)

# 孔洞ID對應的映射字典
hole_id_map = {}  # 孔洞ID到中心點的映射
max_hole_id = 0

# 初始化孔洞順序
initial_hole_positions = {}  # 儲存每個孔洞的初始位置和ID

# 初始化孔洞順序
def initialize_hole_order(hole_detail):
    global initial_hole_positions
    sorted_holes = sorted(hole_detail, key=lambda h: (h['box'][1], h['box'][0]))  # 依照左上角排序
    for idx, hole in enumerate(sorted_holes, start=1):
        box = hole["box"]
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        initial_hole_positions[idx] = {"center": (center_x, center_y), "box": box, "lost_count": 0}

# 根據初始位置來保持孔洞ID的穩定
def draw_detection_with_fixed_ids(frame, hole_detail):
    global initial_hole_positions

    frame_with_detections = frame.copy()

    updated_hole_positions = {}

    for hole in hole_detail:
        box = hole["box"]
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        center_point = (center_x, center_y)

        # 找到最接近的初始化位置
        matched_id = None
        min_distance = float('inf')
        for hole_id, data in initial_hole_positions.items():
            existing_center = data["center"]
            distance = np.linalg.norm(np.array(existing_center) - np.array(center_point))
            if distance < 20 and distance < min_distance:
                matched_id = hole_id
                min_distance = distance
        
        # 如果找到匹配ID，則更新位置，否則保持初始位置不變
        if matched_id is not None:
            updated_hole_positions[matched_id] = {"center": center_point, "box": box}
        else:
            # 丟失的孔洞會被保留
            matched_id = None

    # 處理丟失的孔洞
    for hole_id, data in initial_hole_positions.items():
        if hole_id not in updated_hole_positions:
            # 如果孔洞在當前幀沒有被偵測到，則保持丟失狀態
            updated_hole_positions[hole_id] = data

    # 更新ID映射
    initial_hole_positions = updated_hole_positions

    # 繪製結果
    for hole_id, data in initial_hole_positions.items():
        box = data["box"]
        center_point = data["center"]

        # 繪製孔洞的矩形框
        color = (0, 255, 0)  # 綠色表示正常
        if hole_id not in updated_hole_positions:
            color = (0, 0, 255)  # 紅色表示丟失
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(frame_with_detections, top_left, bottom_right, color, 2)

        # 在中心點顯示ID
        cv2.putText(frame_with_detections, f"{hole_id}", center_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame_with_detections

# 模擬使用範例
cap = cv2.VideoCapture('./test_video/show_workpiece_test.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1120, 630))
    height, width, _ = frame.shape

    # 孔洞偵測
    hole_detail, det = hole_detect.detect(frame)

    # 當偵測到的孔洞數量減少時，執行初始化
    if len(hole_detail) < len(initial_hole_positions):
        initialize_hole_order(hole_detail)

    # 根據 hole_detail 繪製畫面並標記固定ID
    frame_with_fixed_ids = draw_detection_with_fixed_ids(frame, hole_detail)

    # 合併兩個畫面來同時顯示
    combined_frame = np.hstack((det, frame_with_fixed_ids))

    # 顯示畫面
    cv2.imshow('Original Det vs Hole Detail with Fixed IDs', combined_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
