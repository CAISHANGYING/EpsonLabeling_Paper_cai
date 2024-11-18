import numpy as np
from math import pi, cos, sin
from workpiece.stand_workpiece import StandWorkpiece
from workpiece.calculator import WorkpieceCalculator

class PushBackModel1:
    def __init__(self, config: dict, stand_workpiece: StandWorkpiece):
        self._config = config
        self._stand_workpiece = stand_workpiece
        self._hole_id_map = {}  # 孔洞 ID 到中心點的映射
        self._initial_positions = {}  # 孔洞的初始位置
        self._distance_threshold = 100  # 移動空間的距離閾值
        self._max_lost_frames = 10  # 最大失踪幀數
        self._relative_distances = {}  # 相對距離
        self._relative_angles = {}  # 相對角度
        self._stability_score = {}  # 穩定性評分

        # 初始化孔洞 ID 和相關資訊
        self.initialize_hole_ids()

    def initialize_hole_ids(self):
        """
        初始化孔洞 ID 和中心座標，根據 stand_workpiece 提供的初始資料進行設置。
        """
        # 從 stand_workpiece 獲取孔洞詳細資料
        hole_detail = self._stand_workpiece.getHoleDetail()
        # 獲取孔洞的中心座標列表和 ID 列表
        hole_centers, hole_ids = self._stand_workpiece.getCenterCoordinateList()

        # 初始化孔洞 ID 映射和初始位置
        for idx, center in enumerate(hole_centers):
            hole_id = hole_ids[idx]
            box = hole_detail.get(hole_id, {}).get("box", [0, 0, 0, 0])  # 設置默認值避免 KeyError
            self._hole_id_map[hole_id] = {"center": center, "lost_count": 0, "box": box}
            self._initial_positions[hole_id] = center
            self._stability_score[hole_id] = 1.0  # 初始化穩定性評分

        # 初始化孔洞之間的距離和角度
        self._initialize_relative_features()


    def _initialize_relative_features(self):
        """ 計算並儲存所有孔洞之間的初始距離和角度。"""
        hole_ids = list(self._hole_id_map.keys())
        for i in range(len(hole_ids)):
            for j in range(i + 1, len(hole_ids)):
                id1, id2 = hole_ids[i], hole_ids[j]
                center1, center2 = self._hole_id_map[id1]["center"], self._hole_id_map[id2]["center"]
                distance = np.linalg.norm(np.array(center1) - np.array(center2))
                angle = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
                self._relative_distances[(id1, id2)] = distance
                self._relative_distances[(id2, id1)] = distance
                self._relative_angles[(id1, id2)] = angle
                self._relative_angles[(id2, id1)] = angle

    def track_holes(self, hole_detail: list):
        """ 追踪孔洞並為每個孔洞保持固定的 ID。"""
        updated_hole_id_map = {}

        # 更新孔洞的實際檢測
        for hole in hole_detail:
            box = hole["box"]
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            # 匹配初始位置和相對位置特徵
            matched_id = None
            min_distance = float('inf')

            for hole_id, data in self._hole_id_map.items():
                initial_center = self._initial_positions[hole_id]
                
                # 計算與初始位置的距離
                initial_distance = np.linalg.norm(np.array(center) - np.array(initial_center))

                # 相對位置關係匹配
                neighbor_match_score = self._calculate_neighbor_match_score(hole_id, center)

                # 通過多種特徵匹配
                if (initial_distance < self._distance_threshold and 
                    neighbor_match_score > 0.8 and 
                    initial_distance < min_distance):

                    matched_id = hole_id
                    min_distance = initial_distance

            # 更新匹配的孔洞
            if matched_id is not None:
                updated_hole_id_map[matched_id] = {"center": center, "lost_count": 0, "box": box}
                hole["tag"] = matched_id
                # 更新穩定性評分
                self._stability_score[matched_id] = min(self._stability_score[matched_id] + 0.1, 1.0)
            else:
                # 如果沒有找到匹配的孔洞，保留原有的 ID 不變
                pass

        # 保留未被檢測到的孔洞，並增加 lost_count
        for hole_id, data in self._hole_id_map.items():
            if hole_id not in updated_hole_id_map:
                data["lost_count"] += 1
                # 如果孔洞失踪的幀數小於一定閾值，保留它的記錄
                if data["lost_count"] <= self._max_lost_frames:
                    updated_hole_id_map[hole_id] = data
                    # 減少穩定性評分
                    self._stability_score[hole_id] = max(self._stability_score[hole_id] - 0.1, 0.0)

        # 更新全局孔洞 ID 映射
        self._hole_id_map = updated_hole_id_map

    def _calculate_neighbor_match_score(self, hole_id, center):
        """ 通過相對位置和距離特徵來計算當前中心與已知孔洞的匹配得分。"""
        score = 0
        neighbor_ids = [k for k in self._initial_positions.keys() if k != hole_id]
        
        for neighbor_id in neighbor_ids:
            initial_distance = self._relative_distances.get((hole_id, neighbor_id), None)
            if initial_distance is None:
                continue

            if neighbor_id not in self._hole_id_map:
                continue

            current_distance = np.linalg.norm(np.array(center) - np.array(self._hole_id_map[neighbor_id]["center"]))
            distance_diff = abs(current_distance - initial_distance)

            # 如果距離差小於一定閾值，則增加匹配分數
            if distance_diff < 20:
                score += 1

        return score / len(neighbor_ids) if len(neighbor_ids) > 0 else 0

    def get_hole_id_map(self):
        """ 獲取當前的孔洞 ID 到中心點的映射。"""
        return {hole_id: {"center": data["center"], "box": data["box"]}
                for hole_id, data in self._hole_id_map.items() if data["lost_count"] <= self._max_lost_frames}
