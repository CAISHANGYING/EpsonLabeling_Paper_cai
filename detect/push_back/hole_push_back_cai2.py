import random
import cv2
import numpy as np
from workpiece.calculator import WorkpieceCalculator

class PushBackModel:
    def __init__(self, config: dict, standWorkpiece: StandWorkpiece = None) -> None:
        self._config = config
        self._standWorkpiece = standWorkpiece
        self._rollAngle = 0
        self._pitchAngle = 0
        self._hole_id_map = {}  # 用來儲存孔洞ID和其位置
        self.initial_hole_positions = {}  # 儲存每個孔洞的初始位置和ID

    def setConfig(self, config: dict) -> None:
        self._config = config

    def setStandWorkpiece(self, standWorkpiece: StandWorkpiece) -> None:
        self._standWorkpiece = standWorkpiece

    def setImageRotateAngle(self, rollAngle: float, pitchAngle: float) -> None:
        self._rollAngle = rollAngle
        self._pitchAngle = pitchAngle

    def initialize_hole_positions(self, hole_detail: dict) -> None:
        """ 初始化孔洞的 ID 和中心點位置，這樣它們即使被遮蔽也能保持固定 ID """
        self.initial_hole_positions.clear()
        for idx, hole in enumerate(hole_detail, start=1):
            box = hole["box"]
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            self.initial_hole_positions[idx] = {"center": (center_x, center_y), "box": box}

    def draw_detection_with_fixed_ids(self, frame, hole_detail):
        """ 根據孔洞中心點匹配固定的 ID 並繪製 """
        frame_with_detections = frame.copy()
        updated_hole_id_map = {}
        assigned_ids = set()

        for hole in hole_detail:
            box = hole["box"]
            conf = hole.get("conf", 0.5)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            center_point = (center_x, center_y)

            # 匹配 ID
            matched_id = None
            min_distance = float('inf')
            for hole_id, data in self.initial_hole_positions.items():
                existing_center = data["center"]
                distance = np.linalg.norm(np.array(existing_center) - np.array(center_point))
                if distance < 20 and hole_id not in assigned_ids:
                    matched_id = hole_id
                    min_distance = distance

            if matched_id is not None:
                updated_hole_id_map[matched_id] = {"center": center_point, "box": box}
                assigned_ids.add(matched_id)

            # 繪製矩形框
            color = (0, 255, 0)  # 綠色框
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(frame_with_detections, top_left, bottom_right, color, 2)

            # 顯示 ID
            cv2.putText(frame_with_detections, f"{matched_id}", center_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # 顯示中心點
            cv2.circle(frame_with_detections, center_point, 5, (0, 0, 255), -1)  # 中心紅色圓點

        # 更新 ID 映射
        self._hole_id_map = updated_hole_id_map

        return frame_with_detections

    def process_frame(self, frame, hole_detail):
        """ 將孔洞檢測結果與畫框ID繪製到畫面中 """
        self.draw_detection_with_fixed_ids(frame, hole_detail)
        return frame
