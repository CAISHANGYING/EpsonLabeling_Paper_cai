import math

import cv2
import numpy as np
from scipy.spatial import ConvexHull

class HoleDetectConvector:

    
    def __init__(self, movement_threshold=30,large_movement_threshold = 50, small_movement_threshold=5):
        # 記錄前一幀的偵測結果
        self.prev_hole_details = None
        self.movement_threshold = movement_threshold  # 設定大範圍移動的閾值
        self.large_movement_threshold = large_movement_threshold
        self.small_movement_threshold = small_movement_threshold
    def calculate_movement(self, prev_coordinates, current_coordinates):
        """
        計算中心點之間的距離，並判斷移動幅度
        """
        prev_center_x = (prev_coordinates['left_top'][0] + prev_coordinates['right_bottom'][0]) / 2
        prev_center_y = (prev_coordinates['left_top'][1] + prev_coordinates['right_bottom'][1]) / 2

        current_center_x = (current_coordinates['left_top'][0] + current_coordinates['right_bottom'][0]) / 2
        current_center_y = (current_coordinates['left_top'][1] + current_coordinates['right_bottom'][1]) / 2

        distance = math.sqrt((current_center_x - prev_center_x) ** 2 + (current_center_y - prev_center_y) ** 2)

        if distance >= self.large_movement_threshold:
            movement_status = "large"
        elif distance <= self.small_movement_threshold:
            movement_status = "small"
        else:
            movement_status = "normal"

        return distance, movement_status
       
    def update_with_smoothing(self, current_hole_details):
        """
        當移動緩慢時，根據前一幀的偵測結果來平滑當前偵測結果，增強穩定性。
        """
        if self.prev_hole_details is None:
            self.prev_hole_details = current_hole_details
            return current_hole_details

        smoothed_hole_details = {}
        for key, current_value in current_hole_details.items():
            if key in self.prev_hole_details:
                prev_coords = self.prev_hole_details[key]['coordinate']
                current_coords = current_value['coordinate']
                
                # 計算移動距離和移動幅度
                distance, movement_status = self.calculate_movement(prev_coords, current_coords)

                if movement_status == "small":
                    smoothed_x = 0.7 * prev_coords['middle'][0] + 0.3 * current_coords['middle'][0]
                    smoothed_y = 0.7 * prev_coords['middle'][1] + 0.3 * current_coords['middle'][1]
                    smoothed_hole_details[key] = {
                        'coordinate': {
                            'middle': [smoothed_x, smoothed_y]
                        },
                        **self.prev_hole_details[key]
                    }
                else:
                    smoothed_hole_details[key] = current_value
            else:
                smoothed_hole_details[key] = current_value

        self.prev_hole_details = smoothed_hole_details
        return smoothed_hole_details
    
    def check_large_movement(self, current_hole_details):
        """
        當偵測到大範圍移動時，跳過平滑直接使用當前偵測結果。
        """
        if self.prev_hole_details is None:
            self.prev_hole_details = current_hole_details
            return False  # 初次偵測無法計算移動量，視為非大範圍移動

        for key, current_value in current_hole_details.items():
            if key in self.prev_hole_details:
                prev_coords = self.prev_hole_details[key]['coordinate']['middle']
                current_coords = current_value['coordinate']['middle']
                distance = math.sqrt((current_coords[0] - prev_coords[0]) ** 2 + (current_coords[1] - prev_coords[1]) ** 2)

                # 如果超過移動閾值，視為大範圍移動
                if distance >= self.movement_threshold:
                    self.prev_hole_details = current_hole_details
                    return True

        return False
    

    @staticmethod
    def _holeIdCreate(idNumber):

        holeId = ""

        count = idNumber

        while count > 0:

            remainder = (count - 1) % 26

            holeId = chr(97 + remainder) + holeId

            count = (count - 1) // 26

        return holeId

    @staticmethod
    def applyPerspectiveTransform(point, matrix):

        point = np.array([point], dtype=np.float32)
        point = np.array([point])
        transformed_point = cv2.perspectiveTransform(point, matrix)

        coordinates = transformed_point[0][0].tolist()

        return [coordinates[0], coordinates[1]]

    @staticmethod
    def holePerspectiveTransform(hole_detail, boundary_detail, transforms_points, matrix=None):
        if matrix is None:
            matrix = cv2.getPerspectiveTransform(np.array(boundary_detail, dtype=np.float32), transforms_points)

        for key, value in hole_detail.items():
            coordinates = value['coordinate']
            coordinates_real = {k: v[:] for k, v in coordinates.items()}

            hole_detail[key]['coordinate_real'] = coordinates_real
            for coord_key in coordinates:
                transformed_coord = HoleDetectConvector.applyPerspectiveTransform(coordinates[coord_key], matrix)
                
                # 檢查變換後座標的誤差，避免過大偏移
                original_coord = coordinates_real[coord_key]
                error_distance = math.sqrt((transformed_coord[0] - original_coord[0]) ** 2 + (transformed_coord[1] - original_coord[1]) ** 2)
                
                if error_distance < 50:  # 設定誤差容忍範圍
                    coordinates[coord_key] = transformed_coord
                else:
                    print(f"Skipping transformation for {coord_key} due to high error distance: {error_distance}")

            middle = coordinates['middle']
            width = coordinates['right_bottom'][0] - coordinates['left_top'][0]
            height = coordinates['right_bottom'][1] - coordinates['left_top'][1]

            hole_detail[key]['xywh'] = [middle[0], middle[1], width, height]
            hole_detail[key]['width'] = width
            hole_detail[key]['height'] = height

        return matrix, hole_detail

    # def holePerspectiveTransform(hole_detail, boundary_detail, transforms_points, matrix = None) -> (np.array, dict):

    #     if matrix is None:

    #         matrix = cv2.getPerspectiveTransform(np.array(boundary_detail, dtype=np.float32), transforms_points)

    #     # print("Matrix: ", matrix)

    #     for key, value in hole_detail.items():


    #         coordinates = value['coordinate']
    #         coordinates_real = {k: v[:] for k, v in coordinates.items()} # deep copy

    #         hole_detail[key]['coordinate_real'] = coordinates_real

    #         # 对每个坐标点应用透视变换
    #         for coord_key in coordinates:

    #             coordinates[coord_key] = HoleDetectConvector.applyPerspectiveTransform(coordinates[coord_key], matrix)

    #         # 更新 xywh 信息
    #         middle = coordinates['middle']
    #         width = coordinates['right_bottom'][0] - coordinates['left_top'][0]
    #         height = coordinates['right_bottom'][1] - coordinates['left_top'][1]

    #         hole_detail[key]['xywh'] = [middle[0], middle[1], width, height]
    #         hole_detail[key]['width'] = width
    #         hole_detail[key]['height'] = height



    #     return matrix, hole_detail

    @staticmethod
    def sortBoundary(points):
        if not points:
            return np.array([], dtype=np.float32)

        points = np.array(points, dtype=np.float32)

        # 計算凸包，獲得凸包點
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # 計算質心
        center = np.mean(hull_points, axis=0)

        # 以極角從中心點進行排序
        sorted_points = sorted(hull_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))

        # 將順時針排序的點轉換為 numpy 陣列
        sorted_points = np.array(sorted_points)

        # 將左上角的點作為起始點
        top_left_index = np.argmin([p[0] + p[1] for p in sorted_points])
        sorted_points = np.roll(sorted_points, -top_left_index, axis=0)

        return sorted_points



    @staticmethod
    def boundary_convert(boundaryList, imageSize, showPlantSize):

        convertX = showPlantSize[0] / imageSize[0]
        convertY = showPlantSize[1] / imageSize[1]

        boundaryDetail = []

        for item in boundaryList:

            x = item[0] * convertX
            y = item[1] * convertY

            boundaryDetail.append([x, y])

        return boundaryDetail


    @staticmethod
    def convert(detectResult, imageSize, showPlantSize):

        holeDetail = {}
        wrenchDetail = {}
        boundaryDetail = []

        wrenchCount = 1
        holeCount = 1

        convertX = showPlantSize[0] / imageSize[0]
        convertY = showPlantSize[1] / imageSize[1]

        if detectResult:

            for item in detectResult:

                tmp = item["box"]

                x = ((tmp[0] + tmp[2]) / 2) * convertX
                y = ((tmp[1] + tmp[3]) / 2) * convertY
                w = (tmp[2] - tmp[0]) * convertX
                h = (tmp[3] - tmp[1]) * convertY

                if item["tag"] == "wrench":

                    wrenchDetail["Wn" + str(wrenchCount)] = {
                        "tag": "0",
                        "coordinate": {
                            "left_top": [tmp[0] * convertX, tmp[1] * convertY],
                            "right_bottom": [tmp[2] * convertX, tmp[3] * convertY],
                            "middle": [x * 0.5, y * 1.5],
                        },
                        "width": w,
                        "height": h,
                        "xywh": [x, y, w, h],
                        "status": item["tag"]
                    }

                    wrenchCount += 1

                elif item["tag"] == "boundary":

                    boundaryDetail.append((x,y))

                else:

                    holeDetail[HoleDetectConvector._holeIdCreate(holeCount)] = {

                        "tag": "0",
                        "coordinate": {
                            "left_top": [tmp[0] * convertX, tmp[1] * convertY],
                            "right_bottom": [tmp[2] * convertX, tmp[3] * convertY],
                            "middle": [x, y],
                        },
                        "width": w,
                        "height": h,
                        "xywh": [x, y, w, h],
                        "status": item["tag"]
                    }

                    holeCount += 1
        return {
            "hole": holeDetail,
            "wrench": wrenchDetail,
            "boundary": HoleDetectConvector.sortBoundary(boundaryDetail)
        }

    @staticmethod
    def _isCover(wrenchCoordinate : dict, pushbackCoordinate : dict) -> bool:

        wrenchLeftTop = wrenchCoordinate["left_top"]
        wrenchRightBottom = wrenchCoordinate["right_bottom"]

        pushbackLeftTop = pushbackCoordinate["left_top"]
        pushbackRightBottom = pushbackCoordinate["right_bottom"]

        if (wrenchLeftTop[0] <= pushbackLeftTop[0] and wrenchLeftTop[1] <= pushbackLeftTop[1] and
                wrenchRightBottom[0] >= pushbackRightBottom[0] and wrenchRightBottom[1] >= pushbackRightBottom[1]):

            return True

        return False

    @staticmethod
    def wrenchOnWrongScrewPosition(wrenchDetail : dict, pushbackDetail : dict, nowScrewIndex : str) -> dict:

        if nowScrewIndex == "done":

            return {}

        wrongPositionDict = {}

        for wrenchKey, wrenchValue in wrenchDetail.items():

            wrenchCoordinate = wrenchValue["coordinate"]

            for pushbackKey, pushbackValue in pushbackDetail.items():

                if nowScrewIndex != pushbackValue["tag"]:

                    pushbackCoordinate = pushbackValue["coordinate"]

                    if HoleDetectConvector._isCover(wrenchCoordinate, pushbackCoordinate):

                        if wrenchKey not in wrongPositionDict.keys():

                            wrongPositionDict[wrenchKey] = {

                                "wrenchCoordinate": wrenchCoordinate,
                                "holeTag": pushbackValue["tag"],
                                "pushbackCoordinate": pushbackCoordinate
                            }

                        else:

                            if int(wrongPositionDict[wrenchKey]["holeTag"]) > int(pushbackValue["tag"]):

                                wrongPositionDict[wrenchKey] = {

                                    "wrenchCoordinate": wrenchCoordinate,
                                    "holeTag": pushbackValue["tag"],
                                    "pushbackCoordinate": pushbackCoordinate
                                }


        return wrongPositionDict

    @staticmethod
    def covertToResultType(result, screenSize, heightAdjust=0):

        if result is None: return {}

        factorX = 1
        factorY = 1

        convertResult = {}

        for key, value in result.items():

            convertResult[key] = {

                "tag": value["tag"],
                "coordinate" : {
                    "left_top": [value["coordinate"]["left_top"][0] / factorX, (value["coordinate"]["left_top"][1] + heightAdjust) / factorY],
                    "right_bottom": [value["coordinate"]["right_bottom"][0] / factorX, (value["coordinate"]["right_bottom"][1] + heightAdjust) / factorY],
                    "middle": [value["coordinate"]["middle"][0] / factorX, (value["coordinate"]["middle"][1] + heightAdjust) / factorY],
                },
                "width": value["width"] / factorX,
                "height": (value["height"] + heightAdjust) / factorY,
                "xywh": [value["xywh"][0] / factorX, (value["xywh"][1] + heightAdjust) / factorY, value["xywh"][2] / factorX, (value["xywh"][3] + heightAdjust) / factorY],
                "status": value["status"]
            }

        return convertResult

    @staticmethod
    def covertToResultTypeForServer(result, screenSize, heightAdjust=0):

        factorX = screenSize[0]
        factorY = screenSize[1]

        convertResult = {}

        for key, value in result.items():
            convertResult[key] = {

                "tag": value["tag"],
                "coordinate": {
                    "left_top": [value["coordinate"]["left_top"][0] / factorX,
                                 (value["coordinate"]["left_top"][1] + heightAdjust) / factorY],
                    "right_bottom": [value["coordinate"]["right_bottom"][0] / factorX,
                                     (value["coordinate"]["right_bottom"][1] + heightAdjust) / factorY],
                    "middle": [value["coordinate"]["middle"][0] / factorX,
                               (value["coordinate"]["middle"][1] + heightAdjust) / factorY],
                },
                "width": value["width"] / factorX,
                "height": (value["height"] + heightAdjust) / factorY,
                "xywh": [value["xywh"][0] / factorX, (value["xywh"][1] + heightAdjust) / factorY,
                         value["xywh"][2] / factorX, (value["xywh"][3] + heightAdjust) / factorY],
                "status": value["status"]
            }

        return convertResult

    @staticmethod
    def _draw(frame, coordinate, x_move, y_move, lw, color, tag, debug=False):

        cv2.rectangle(frame, (int(coordinate['left_top'][0]), int(coordinate['left_top'][1])),
                      (int(coordinate['right_bottom'][0]), int(coordinate['right_bottom'][1])), color, 2)

        p1, p2 = ((int(coordinate['middle'][0]) + x_move), (int(coordinate['middle'][1]) + y_move)), (
            (int(coordinate['middle'][0]) - x_move), (int(coordinate['middle'][1]) - y_move))

        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(str(tag), 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

        if debug is False:
            cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # filled

            cv2.putText(frame, tag, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                        lw / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

        return frame

    @staticmethod
    def getPushbackFrame(frame: np.ndarray, hole_result_convert: dict, nowHole: int = -1, debug=False) -> tuple:
        frame = frame.copy()
        x_move = 25
        y_move = -25
        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

        # 初始化 color_count 字典來紀錄顏色的次數
        color_count = {
            'hole': 0,        # 對應 (50, 108, 66)
            'hole_stage2': 0, # 對應 (255, 149, 0)
            'hole_match': 0,  # 對應 (170, 0, 255)
            'else': 0         # 對應 (0, 234, 255)
        }

        for _, value in hole_result_convert.items():
            tag = value['tag']

            if 'coordinate_real' in value and debug is False:
                coordinate = value['coordinate_real']
            else:
                coordinate = value['coordinate']

            if nowHole == -1 or int(tag) == nowHole:
                tag = tag.zfill(2)

                # 根據 status 設定顏色並增加顏色次數
                if value['status'] == 'hole':
                    color = (50, 108, 66)
                    color_count['hole'] += 1
                elif value['status'] == 'hole_stage2':
                    color = (255, 149, 0)
                    color_count['hole_stage2'] += 1
                elif value['status'] == 'hole_match':
                    color = (170, 0, 255)
                    color_count['hole_match'] += 1
                else:
                    color = (0, 234, 255)
                    color_count['else'] += 1

                # 使用 _draw 方法繪製
                frame = HoleDetectConvector._draw(frame, coordinate, x_move, y_move, lw, color, tag, debug)

        # 返回 frame 和 color_count 作為統計結果
        return frame, color_count

    @staticmethod
    def getPushbackFrameEachStep(frame : np.ndarray, hole_result_convert : dict, nowHole : int = -1, debug=False) -> np.ndarray:

        frame = frame.copy()
        frame1 = frame.copy()
        frame2 = frame.copy()
        frame3 = frame.copy()

        x_move = 25
        y_move = -25

        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

        for _, value in hole_result_convert.items():

            tag = value['tag']

            if 'coordinate_real' in value and debug is False:

                coordinate = value['coordinate_real']

            else:

                coordinate = value['coordinate']

            if nowHole == -1 or int(tag) == nowHole:

                tag = tag.zfill(2)

                if value['status'] == 'hole':

                    color = (50, 108, 66)

                elif value['status'] == 'hole_stage2':

                    color = (255, 149, 0)

                elif value['status'] == 'hole_match':

                    color = (170, 0, 255)

                else:

                    color = (0, 234, 255)

                if value['status'] == 'hole':

                    frame = HoleDetectConvector._draw(frame, coordinate, x_move, y_move, lw, color, tag, debug)
                    frame1 = HoleDetectConvector._draw(frame1, coordinate, x_move, y_move, lw, color, tag, debug)
                    frame2 = HoleDetectConvector._draw(frame2, coordinate, x_move, y_move, lw, color, tag, debug)
                    frame3 = HoleDetectConvector._draw(frame3, coordinate, x_move, y_move, lw, color, tag, debug)

                if value['status'] == 'hole_stage2':

                    frame1 = HoleDetectConvector._draw(frame1, coordinate, x_move, y_move, lw, color, tag, debug)
                    frame2 = HoleDetectConvector._draw(frame2, coordinate, x_move, y_move, lw, color, tag, debug)
                    frame3 = HoleDetectConvector._draw(frame3, coordinate, x_move, y_move, lw, color, tag, debug)

                if value['status'] == 'hole_match':

                    frame2 = HoleDetectConvector._draw(frame2, coordinate, x_move, y_move, lw, color, tag, debug)
                    frame3 = HoleDetectConvector._draw(frame3, coordinate, x_move, y_move, lw, color, tag, debug)

                if value['status'] == 'hole_pushback':

                    frame3 = HoleDetectConvector._draw(frame3, coordinate, x_move, y_move, lw, color, tag, debug)

        return frame, frame1, frame2, frame3