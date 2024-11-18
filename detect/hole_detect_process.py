import math

import cv2
import numpy as np
from scipy.spatial import ConvexHull

class HoleDetectConvector:

    @staticmethod
    def _holeIdCreate(idNumber):
        holeId = []
        
        while idNumber > 0:
            idNumber -= 1  # 先減去1來確保正確的字母對應
            remainder = idNumber % 26
            holeId.append(chr(97 + remainder))
            idNumber //= 26

        return ''.join(reversed(holeId))  # 一次性反轉並合併成字串


    @staticmethod
    def applyPerspectiveTransform(point, matrix):
        point = np.array([[point]], dtype=np.float32)  # 一次性轉換為正確的形狀
        transformed_point = cv2.perspectiveTransform(point, matrix)

        return transformed_point[0][0].tolist()  # 直接返回座標列表


    @staticmethod
    def holePerspectiveTransform(hole_detail, boundary_detail, transforms_points, matrix=None) -> (np.array, dict):
        # 如果 matrix 為空，則計算透視變換矩陣
        if matrix is None:
            matrix = cv2.getPerspectiveTransform(np.array(boundary_detail, dtype=np.float32), transforms_points)

        for key, value in hole_detail.items():
            coordinates = value['coordinate']
            # 直接將座標拷貝保存到 coordinate_real，這樣可以避免每次手動複製
            hole_detail[key]['coordinate_real'] = {k: v[:] for k, v in coordinates.items()}

            # 對每個座標點應用透視變換
            for coord_key, coord_value in coordinates.items():
                coordinates[coord_key] = HoleDetectConvector.applyPerspectiveTransform(coord_value, matrix)

            # 計算更新 xywh 資訊
            left_top = coordinates['left_top']
            right_bottom = coordinates['right_bottom']
            middle = coordinates['middle']

            width = right_bottom[0] - left_top[0]
            height = right_bottom[1] - left_top[1]

            # 更新寬度、高度和 xywh
            hole_detail[key]['width'] = width
            hole_detail[key]['height'] = height
            hole_detail[key]['xywh'] = [middle[0], middle[1], width, height]

        return matrix, hole_detail


    @staticmethod
    def sortBoundary(points):
        if not points:
            return np.array([], dtype=np.float32)

        points = np.array(points, dtype=np.float32)

        # 計算凸包，獲得凸包點
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # 找到質心（中心點）
        center = np.mean(hull_points, axis=0)

        # 按照點到中心的極角排序，確保是順時針方向
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])

        sorted_points = sorted(hull_points, key=angle_from_center)

        # 確保起點為 top-left
        top_left_index = np.argmin([p[0] + p[1] for p in sorted_points])
        sorted_points = np.roll(sorted_points, -top_left_index, axis=0)

        return np.array(sorted_points, dtype=np.float32)


    @staticmethod
    def boundary_convert(boundaryList, imageSize, showPlantSize):
        # 計算縮放因子
        convertX = showPlantSize[0] / imageSize[0]
        convertY = showPlantSize[1] / imageSize[1]

        # 將 boundaryList 轉換為 numpy array
        boundaryArray = np.array(boundaryList, dtype=np.float32)

        # 將每個點的 x 和 y 坐標進行縮放
        boundaryArray[:, 0] *= convertX
        boundaryArray[:, 1] *= convertY

        # 返回處理後的邊界點列表
        return boundaryArray.tolist()


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

                # 預計算轉換後的中間點、寬度和高度
                x = ((tmp[0] + tmp[2]) / 2) * convertX
                y = ((tmp[1] + tmp[3]) / 2) * convertY
                w = (tmp[2] - tmp[0]) * convertX
                h = (tmp[3] - tmp[1]) * convertY

                coordinates = {
                    "left_top": [tmp[0] * convertX, tmp[1] * convertY],
                    "right_bottom": [tmp[2] * convertX, tmp[3] * convertY],
                    "middle": [x, y],
                }

                if item["tag"] == "wrench":
                    wrenchDetail[f"Wn{wrenchCount}"] = {
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
                    boundaryDetail.append((x, y))

                else:  # 處理 "hole"
                    holeDetail[HoleDetectConvector._holeIdCreate(holeCount)] = {
                        "tag": "0",
                        "coordinate": coordinates,
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
    def _isCover(wrenchCoordinate: dict, pushbackCoordinate: dict) -> bool:
        wrenchLeftTop = wrenchCoordinate["left_top"]
        wrenchRightBottom = wrenchCoordinate["right_bottom"]

        pushbackLeftTop = pushbackCoordinate["left_top"]
        pushbackRightBottom = pushbackCoordinate["right_bottom"]

        # 直接返回判斷結果
        return (wrenchLeftTop[0] <= pushbackLeftTop[0] and wrenchLeftTop[1] <= pushbackLeftTop[1] and
                wrenchRightBottom[0] >= pushbackRightBottom[0] and wrenchRightBottom[1] >= pushbackRightBottom[1])


    @staticmethod
    def wrenchOnWrongScrewPosition(wrenchDetail: dict, pushbackDetail: dict, nowScrewIndex: str) -> dict:
        if nowScrewIndex == "done":
            return {}

        wrongPositionDict = {}

        for wrenchKey, wrenchValue in wrenchDetail.items():
            wrenchCoordinate = wrenchValue["coordinate"]

            for pushbackKey, pushbackValue in pushbackDetail.items():
                if nowScrewIndex != pushbackValue["tag"]:
                    pushbackCoordinate = pushbackValue["coordinate"]

                    if HoleDetectConvector._isCover(wrenchCoordinate, pushbackCoordinate):
                        # 如果 wrenchKey 不在字典中，或者 holeTag 的數值更小，則更新
                        if (wrenchKey not in wrongPositionDict or 
                            int(wrongPositionDict[wrenchKey]["holeTag"]) > int(pushbackValue["tag"])):
                            
                            wrongPositionDict[wrenchKey] = {
                                "wrenchCoordinate": wrenchCoordinate,
                                "holeTag": pushbackValue["tag"],
                                "pushbackCoordinate": pushbackCoordinate
                            }

        return wrongPositionDict


    @staticmethod
    def convertToResultType(result, screenSize, heightAdjust=0):
        if result is None:
            return {}

        factorX, factorY = screenSize[0], screenSize[1]  # 根據螢幕大小計算比例因子
        convertResult = {}

        def adjust_coordinate(coordinate):
            return [
                coordinate[0] / factorX,
                (coordinate[1] + heightAdjust) / factorY
            ]

        for key, value in result.items():
            coordinates = value["coordinate"]

            convertResult[key] = {
                "tag": value["tag"],
                "coordinate": {
                    "left_top": adjust_coordinate(coordinates["left_top"]),
                    "right_bottom": adjust_coordinate(coordinates["right_bottom"]),
                    "middle": adjust_coordinate(coordinates["middle"]),
                },
                "width": value["width"] / factorX,
                "height": (value["height"] + heightAdjust) / factorY,
                "xywh": [
                    value["xywh"][0] / factorX,
                    (value["xywh"][1] + heightAdjust) / factorY,
                    value["xywh"][2] / factorX,
                    (value["xywh"][3] + heightAdjust) / factorY,
                ],
                "status": value["status"]
            }

        return convertResult

    @staticmethod
    def convertToResultTypeForServer(result, screenSize, heightAdjust=0):
        factorX, factorY = screenSize[0], screenSize[1]
        convertResult = {}

        def adjust_coordinate(coordinate):
            return [
                coordinate[0] / factorX,
                (coordinate[1] + heightAdjust) / factorY
            ]

        for key, value in result.items():
            coordinates = value["coordinate"]

            convertResult[key] = {
                "tag": value["tag"],
                "coordinate": {
                    "left_top": adjust_coordinate(coordinates["left_top"]),
                    "right_bottom": adjust_coordinate(coordinates["right_bottom"]),
                    "middle": adjust_coordinate(coordinates["middle"]),
                },
                "width": value["width"] / factorX,
                "height": (value["height"] + heightAdjust) / factorY,
                "xywh": [
                    value["xywh"][0] / factorX,
                    (value["xywh"][1] + heightAdjust) / factorY,
                    value["xywh"][2] / factorX,
                    (value["xywh"][3] + heightAdjust) / factorY,
                ],
                "status": value["status"]
            }

        return convertResult

    @staticmethod
    def _draw(frame, coordinate, x_move, y_move, lw, color, tag, debug=False):
        # 畫出矩形框
        cv2.rectangle(frame, 
                    (int(coordinate['left_top'][0]), int(coordinate['left_top'][1])), 
                    (int(coordinate['right_bottom'][0]), int(coordinate['right_bottom'][1])), 
                    color, 2)

        # 計算文字位置和移動偏移量
        middle_x, middle_y = int(coordinate['middle'][0]), int(coordinate['middle'][1])
        p1 = (middle_x + x_move, middle_y + y_move)
        p2 = (middle_x - x_move, middle_y - y_move)

        # 計算字型厚度和文字大小
        tf = max(lw - 1, 1)  # font thickness
        text_size = cv2.getTextSize(str(tag), 0, fontScale=lw / 3, thickness=tf)[0]
        text_w, text_h = text_size

        # 判斷文字是否應該放在矩形框內或者外
        outside = p1[1] - text_h >= 3
        p2 = (p1[0] + text_w, p1[1] - text_h - 3) if outside else (p1[0] + text_w, p1[1] + text_h + 3)

        if not debug:
            # 畫出填充矩形並顯示文字
            cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(frame, tag, 
                        (p1[0], p1[1] - 2 if outside else p1[1] + text_h + 2), 
                        0, lw / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

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

        # 狀態與顏色的映射
        status_color_map = {
            'hole': (50, 108, 66),
            'hole_stage2': (255, 149, 0),
            'hole_match': (170, 0, 255),
            'else': (0, 234, 255)
        }

        for _, value in hole_result_convert.items():
            tag = value['tag']

            # 確定使用的座標
            coordinate = value['coordinate_real'] if 'coordinate_real' in value and not debug else value['coordinate']

            if nowHole == -1 or int(tag) == nowHole:
                tag = tag.zfill(2)

                # 選擇顏色並增加顏色次數
                color = status_color_map.get(value['status'], status_color_map['else'])
                color_count[value['status'] if value['status'] in color_count else 'else'] += 1

                # 使用 _draw 方法繪製
                frame = HoleDetectConvector._draw(frame, coordinate, x_move, y_move, lw, color, tag, debug)

        # 返回 frame 和 color_count 作為統計結果
        return frame, color_count


@staticmethod
def getPushbackFrameEachStep(frame: np.ndarray, hole_result_convert: dict, nowHole: int = -1, debug=False) -> tuple:
    # 複製出 4 個 frame，用於不同階段的繪製
    frame = frame.copy()
    frames = [frame.copy() for _ in range(4)]

    x_move = 25
    y_move = -25
    lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

    for _, value in hole_result_convert.items():
        tag = value['tag']

        # 根據 debug 狀態選擇使用的座標
        coordinate = value['coordinate_real'] if 'coordinate_real' in value and not debug else value['coordinate']

        if nowHole == -1 or int(tag) == nowHole:
            tag = tag.zfill(2)

            # 根據 status 設定顏色
            color_map = {
                'hole': (50, 108, 66),
                'hole_stage2': (255, 149, 0),
                'hole_match': (170, 0, 255),
                'hole_pushback': (0, 234, 255)
            }
            color = color_map.get(value['status'], (0, 234, 255))

            # 根據不同狀態繪製對應的 frame
            if value['status'] == 'hole':
                for i in range(4):
                    frames[i] = HoleDetectConvector._draw(frames[i], coordinate, x_move, y_move, lw, color, tag, debug)
            elif value['status'] == 'hole_stage2':
                for i in range(1, 4):
                    frames[i] = HoleDetectConvector._draw(frames[i], coordinate, x_move, y_move, lw, color, tag, debug)
            elif value['status'] == 'hole_match':
                for i in range(2, 4):
                    frames[i] = HoleDetectConvector._draw(frames[i], coordinate, x_move, y_move, lw, color, tag, debug)
            elif value['status'] == 'hole_pushback':
                frames[3] = HoleDetectConvector._draw(frames[3], coordinate, x_move, y_move, lw, color, tag, debug)

    return tuple(frames)
