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

# 讀取影片
# cap = cv2.VideoCapture('./test_video/90.mp4')
cap = cv2.VideoCapture('./test_video/show_workpiece_test.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

# 初始化變數
frame_count = 0
hole_id_map = {}  # 孔洞ID到中心點的映射

# 定義函數根據 hole_detail 繪製偵測結果並標記固定ID
def draw_detection_with_fixed_ids(frame, hole_detail):
    frame_with_detections = frame.copy()
    global hole_id_map

    # 初始化現有ID，如果是第一幀，按照左上角排序分配ID
    if len(hole_id_map) == 0:
        sorted_holes = sorted(hole_detail, key=lambda h: (h['box'][1], h['box'][0]))
        for idx, hole in enumerate(sorted_holes, start=1):
            box = hole["box"]
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            hole_id_map[idx] = {"center": (center_x, center_y), "lost_count": 0}

    updated_hole_id_map = {}
    assigned_ids = set()

    for hole in hole_detail:
        # 獲取框的坐標
        box = hole["box"]
        conf = hole.get("conf", 0.5)
        
        # 計算中心點座標
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        center_point = (center_x, center_y)

        # 嘗試匹配現有ID
        matched_id = None
        min_distance = float('inf')
        for hole_id, data in hole_id_map.items():
            existing_center = data["center"]
            distance = np.linalg.norm(np.array(existing_center) - np.array(center_point))
            if distance < 20 and distance < min_distance and hole_id not in assigned_ids:
                matched_id = hole_id
                min_distance = distance

        # 如果找到匹配的ID，則更新位置
        if matched_id is not None:
            updated_hole_id_map[matched_id] = {"center": center_point, "lost_count": 0}
            assigned_ids.add(matched_id)
        else:
            # 如果沒有匹配的ID，則查找未分配的最小ID進行分配
            for i in range(1, len(hole_detail) + 1):
                if i not in assigned_ids and i not in hole_id_map:
                    updated_hole_id_map[i] = {"center": center_point, "lost_count": 0}
                    assigned_ids.add(i)
                    matched_id = i
                    break

        # 繪製孔洞的矩形框
        color = (0, 255, 0)  # 綠色框
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(frame_with_detections, top_left, bottom_right, color, 2)

        # 在中心點顯示ID
        cv2.putText(frame_with_detections, f"{matched_id}", center_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 保留未被檢測到的孔洞，並增加 lost_count
    for hole_id, data in hole_id_map.items():
        if hole_id not in updated_hole_id_map:
            data["lost_count"] += 1
            if data["lost_count"] <= 10:  # 如果孔洞失蹤的幀數小於等於10，則保留其位置
                updated_hole_id_map[hole_id] = data

                # 繪製失蹤孔洞的矩形框和ID
                center_point = data["center"]
                cv2.putText(frame_with_detections, f"{hole_id}", (int(center_point[0]), int(center_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 紅色顯示失蹤的孔洞ID

    # 更新全局ID映射
    hole_id_map = updated_hole_id_map

    return frame_with_detections

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
    print("hole Detail:",hole_convert)

    # 合併兩個畫面來同時顯示
    combined_frame = np.hstack((det, frame_with_fixed_ids))

    # 顯示兩個畫面的效果
    cv2.imshow('Original Det vs Hole Detail with Fixed IDs', combined_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
