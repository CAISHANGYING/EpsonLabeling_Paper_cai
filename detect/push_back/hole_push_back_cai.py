import random
from math import pi, sin, cos

import cv2
import numpy as np

from detect.hole_detect_process import HoleDetectConvector
from workpiece.stand_workpiece import StandWorkpiece
from workpiece.calculator import WorkpieceCalculator
from numba import jit, njit, prange

import concurrent.futures

import timeit

import threading

class PushBackModel1:

    _config : dict = None
    _standWorkpiece : StandWorkpiece = None

    _threadingResult = []

    _rollAngle = 0
    _pitchAngle = 0

    def __init__(self, config : dict, standWorkpiece : StandWorkpiece = None ) -> None:

        self._config = config
        self._standWorkpiece = standWorkpiece

    def setConfig(self, config : dict) -> None:

        self._config = config

    def setStandWorkpiece(self, standWorkpiece : StandWorkpiece) -> None:

        self._standWorkpiece = standWorkpiece

    def getStandWorkpieceName(self) -> str:

        return self._standWorkpiece.getName() if self._standWorkpiece is not None else "None"

    def setImageRotateAngle(self, rollAngle : float, pitchAngle : float) -> None:

        self._rollAngle = rollAngle
        self._pitchAngle = pitchAngle

    # 已修改
    def _isAngleMatched(self, angle: float, mathAngle: float) -> bool:
        differentValue = 2

        # 對負角度進行處理
        if angle < 0 and mathAngle < 0:
            return (mathAngle - differentValue) <= angle <= (mathAngle + differentValue)
        
        # 對非負角度進行處理
        return (mathAngle - differentValue) <= angle <= (mathAngle + differentValue)

    # 已修改 
    def _isRatioMatched(self, ratio: float, mathRatio: float) -> bool:
        # 提前計算不同值範圍
        shape = self._standWorkpiece.getShape()
        differentValue = self._config['ratio_match_deviation'] * (shape[0] + shape[1]) / 2
        
        # 直接返回布爾結果
        return (mathRatio - differentValue) <= ratio <= (mathRatio + differentValue)

    
    # 已修改 
    def _isCoverGapMatched(self, gap: int, matchGap: int) -> bool:
        differentValue = 1
        # 直接返回判斷結果
        return (matchGap - differentValue) <= gap <= (matchGap + differentValue)


    # 已修改 
    def _countCover(self, holeDetail: dict, middle: list, lengthVector: list) -> int:
        count = 0
        lengthVectorLen = WorkpieceCalculator.calLength(lengthVector)  # 計算一次 lengthVector 的長度

        for tag, value in holeDetail.items():
            # 計算座標差向量並計算其長度
            diffVector = [value['coordinate']['middle'][0] - middle[0], value['coordinate']['middle'][1] - middle[1]]
            diffLength = WorkpieceCalculator.calLength(diffVector)  # 計算差距長度

            # 判斷是否在範圍內
            if 0 < diffLength <= lengthVectorLen:
                count += 1

        return count

    # 已修改 
    def _lengthCreate(self, holeDetail: dict) -> dict:
        lengthDict = {}

        # 提取每個標籤的座標中心，避免重複存取
        coordinates = {tag: value['coordinate']['middle'] for tag, value in holeDetail.items()}

        for tag1, middle1 in coordinates.items():
            lengthDict[tag1] = []

            for tag2, middle2 in coordinates.items():
                if tag1 != tag2:  # 只有當標籤不同時才進行計算
                    # 計算長度向量
                    lengthVector = [middle2[0] - middle1[0], middle2[1] - middle1[1]]

                    # 計算覆蓋範圍並添加到結果中
                    lengthDict[tag1].append({
                        'to': tag2,
                        'vector': lengthVector,
                        'cover': self._countCover(holeDetail, middle1, lengthVector)
                    })

        return lengthDict

    def _degressToRadians(self, degress : float) -> float:

        return degress * pi / 180

    # 已修改 
    def findMatch(self, angle: float, ratio: float, coverGap: int) -> dict:
        # 提前提取長度關係資料，減少重複調用
        lengthRelationship = self._standWorkpiece.getLengthRelationship()
        resultDict = {}
        minDiff = float('inf')  # 儲存最小差異值

        for key, value in lengthRelationship.items():
            for item in value:
                # 先檢查是否匹配角度、比例和覆蓋間隙
                if (self._isAngleMatched(angle, item['angle']) and
                    self._isRatioMatched(ratio, item['ratio']) and
                    self._isCoverGapMatched(coverGap, item['cover'])):

                    # 計算角度和比例的平方差值來進行比較
                    angleDiff = (angle - item['angle']) ** 2
                    ratioDiff = (ratio - item['ratio']) ** 2
                    totalDiff = angleDiff + ratioDiff

                    # 更新最接近的匹配結果
                    if totalDiff < minDiff:
                        minDiff = totalDiff
                        resultDict = item

        return resultDict

    
    # 已修改
    def inStandWorkpieceRelationship(self, relationship: list, length1: dict, length2: dict) -> dict:
        # 提前計算不變的值
        length1Vector = length1['vector']
        length2Vector = length2['vector']

        angle = WorkpieceCalculator.calAngle(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        ratio = WorkpieceCalculator.calRatio(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        coverGap = length1['cover'] - length2['cover']

        # 直接進行一次性比對
        for info in relationship:
            if (self._isAngleMatched(angle, info['angle']) and 
                self._isRatioMatched(ratio, info['ratio']) and 
                self._isCoverGapMatched(coverGap, info['cover_gap'])):
                return info  # 匹配成功立即返回，減少迴圈運行

        return {}  # 若無匹配結果，返回空字典

    # 已修改
    def findLengthMatch(self, length1: dict, length2: dict, length3: dict) -> list:
        # 提前計算重複使用的變量
        length1Vector = length1['vector']
        length2Vector = length2['vector']
        
        angle12 = WorkpieceCalculator.calAngle(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        ratio12 = WorkpieceCalculator.calRatio(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        coverGap12 = length1['cover'] - length2['cover']

        findKeyList = []
        lengthRelationship = self._standWorkpiece.getLengthRelationship()

        # 使用 set() 來加快檢索速度
        for key, value in lengthRelationship.items():
            for info in value:
                if (self._isAngleMatched(angle12, info['angle']) and 
                    self._isRatioMatched(ratio12, info['ratio']) and 
                    self._isCoverGapMatched(coverGap12, info['cover_gap'])):
                    findKeyList.append(key)
                    break  # 提前跳出內部迴圈

        if not findKeyList:
            return []

        verifyList = []

        # 避免多餘的複製操作
        for key in findKeyList:
            standWorkpieceLengthRelationship = lengthRelationship[key]

            match12 = self.inStandWorkpieceRelationship(standWorkpieceLengthRelationship, length1, length2)
            if not match12:
                continue
            
            # 使用 shallow copy 以減少內存負擔
            tempRelationship = standWorkpieceLengthRelationship[:]
            tempRelationship.remove(match12)

            match13 = self.inStandWorkpieceRelationship(tempRelationship, length1, length3)
            if not match13:
                continue
            
            tempRelationship.remove(match13)

            match23 = self.inStandWorkpieceRelationship(tempRelationship, length2, length3)
            if not match23:
                continue

            # 驗證條件
            if (match23['length_1']['to'] != match12['length_2']['to'] or 
                match23['length_2']['to'] != match13['length_2']['to']):
                continue

            verifyList.append({
                'match12': match12,
                'match13': match13,
                'match23': match23
            })

        return verifyList


    # 已修改 
    def findLengthMatchKnowStart(self, findRelationship: list, length1: dict, length2: dict, length3: dict) -> dict:
        # 尋找 match12，提前返回無匹配情況
        match12 = self.inStandWorkpieceRelationship(findRelationship, length1, length2)
        if not match12:
            return {}

        findRelationship.remove(match12)  # 找到匹配後立即移除以提高後續查找效率

        # 尋找 match13，提前返回無匹配情況
        match13 = self.inStandWorkpieceRelationship(findRelationship, length1, length3)
        if not match13:
            return {}

        findRelationship.remove(match13)  # 同樣找到匹配後移除

        # 尋找 match23，提前返回無匹配情況
        match23 = self.inStandWorkpieceRelationship(findRelationship, length2, length3)
        if not match23:
            return {}

        # 檢查條件是否滿足，如果不滿足直接返回
        if not (match23['length_1']['to'] == match12['length_2']['to'] and match23['length_2']['to'] == match13['length_2']['to']):
            return {}

        # 匹配成功，返回所有結果
        return {
            'match12': match12,
            'match13': match13,
            'match23': match23
        }

    # 已修改 
    def findNearHole(self, holeDetail: dict, doneIdList: list, magnification: float, position: list) -> dict:
        # 提前計算允許偏差，避免每次迴圈重複計算
        allowance = self._config['hole_match_deviation'] * (sum(self._standWorkpiece.getShape())) * magnification * 0.75

        resultId = ""
        result = {}

        # 只對未完成的洞進行查找
        keys = (key for key in holeDetail.keys() if key not in doneIdList)

        for key in keys:
            hole = holeDetail[key]
            hole_coord = hole["coordinate"]

            left_top_x, left_top_y = hole_coord["left_top"]
            right_bottom_x, right_bottom_y = hole_coord["right_bottom"]

            # 確保 position 在 left_top 和 right_bottom 之間，考慮 allowance 誤差
            if (left_top_x - allowance <= position[0] <= right_bottom_x + allowance) and \
            (left_top_y - allowance <= position[1] <= right_bottom_y + allowance):

                result = hole.copy()  # 複製 hole 資料
                resultId = key

                result["status"] = "hole_match"
                break  # 找到最近的 hole 後，直接跳出迴圈

        if not resultId:
            return {}

        return {
            "id": resultId,
            "result": result
        }

    def verifyHole(self, pushbackHole : dict, position : list):

        for key, value in pushbackHole.items():

            if (value["coordinate"]["left_top"][0] <= position[0]) and (position[0] <= value["coordinate"]["right_bottom"][0]) and \
                (value["coordinate"]["left_top"][1] <= position[1]) and (position[1] <= value["coordinate"]["right_bottom"][1]):

                return False

        return True
    # 已修改 
    def pushback(self, holeDetail: dict, nowPushback: dict, doneTagList: list, doneIdList: list, magnification: float, rotateAngle: float, matrix_inv: np.ndarray = None) -> dict:
        count = len(nowPushback)

        start1 = doneTagList[0]
        start2 = doneTagList[-1]

        startHole = next(result for result in nowPushback.values() if result["tag"] == start1)

        # 使用列表解析和 next 函數取代多重過濾操作
        length_ = [result for result in self._standWorkpiece.getLengthRelationship()[start1]
                if result['length_1']['to'] == start2 and result['length_2']['to'] not in doneTagList]

        stand_holeDetail = self._standWorkpiece.getHoleDetail()
        temp_push_back = {}

        # 儲存角度的弧度形式，避免多次重複計算
        roll_angle_rad = self._degressToRadians(self._rollAngle)
        pitch_angle_rad = self._degressToRadians(self._pitchAngle)
        rotate_angle_rad = self._degressToRadians(rotateAngle)

        cos_rotate = cos(rotate_angle_rad)
        sin_rotate = sin(rotate_angle_rad)

        for length in length_:
            i = length['length_2']['to']
            vector = length['length_2']['vector']

            # 預先計算 bv1 和 bv2
            bv1 = vector[0] / np.cos(roll_angle_rad)
            bv2 = vector[1] / np.cos(pitch_angle_rad)

            # 優化旋轉向量計算，減少重複呼叫 cos 和 sin
            v1 = (bv1 * cos_rotate - bv2 * sin_rotate) * magnification
            v2 = (bv1 * sin_rotate + bv2 * cos_rotate) * magnification

            middle = [startHole["coordinate"]["middle"][0] + v1, startHole["coordinate"]["middle"][1] + v2]
            middle_old = middle.copy()

            # 檢查是否符合條件
            if not self.verifyHole(nowPushback, middle_old):
                return {}

            nearHole = self.findNearHole(holeDetail, doneIdList, magnification, middle_old)

            if nearHole:
                nearHoleId = nearHole['id']
                nearHoleResult = nearHole['result']

                doneIdList.append(nearHoleId)
                doneTagList.append(str(i))

                nowPushback[nearHoleId] = nearHoleResult
                nowPushback[nearHoleId]["tag"] = str(i)

                count += 1
            else:
                # 儲存洞的寬和高的計算
                stand_hole = stand_holeDetail[self._standWorkpiece.getHoleIdTagConvertDict()[str(i)]]
                width = stand_hole['width'] * magnification
                height = stand_hole['height'] * magnification

                doneTagList.append(str(i))

                temp_push_back[str(i)] = {
                    "tag": str(i),
                    "coordinate": {
                        "left_top": [middle[0] - width / 2, middle[1] - height / 2],
                        "right_bottom": [middle[0] + width / 2, middle[1] + height / 2],
                        "middle": middle,
                    },
                    "width": width,
                    "height": height,
                    "xywh": [middle[0], middle[1], width, height],
                    "status": "hole_pushback",
                }

                if matrix_inv is not None:
                    middle = HoleDetectConvector.applyPerspectiveTransform(middle, matrix_inv)
                    coordinates = temp_push_back[str(i)]["coordinate"]

                    temp_push_back[str(i)]["coordinate_real"] = {
                        "left_top": HoleDetectConvector.applyPerspectiveTransform(coordinates["left_top"], matrix_inv),
                        "right_bottom": HoleDetectConvector.applyPerspectiveTransform(coordinates["right_bottom"], matrix_inv),
                        "middle": middle,
                    }

        # 使用 dict.update 方法直接合併字典，避免多次重新分配
        nowPushback.update(temp_push_back)

        return {
            "count": count,
            "real_pushback": nowPushback,
            "temp_pushback": temp_push_back,
            "pushback": nowPushback
        }
    # 已修改
    def getPushbackPositionThreading(self, holeDetail: dict, magnificationAgain : float, threadingCount:int):

        self._threadingResult = []

        tryTime = int(self._config['try_time'] * 100)

        eachThreadTryTime = tryTime // threadingCount

        if len(holeDetail) < 4: return {}

        lengthCreateResult = self._lengthCreate(holeDetail)

        startHoleList = list(lengthCreateResult.keys())

    # 使用 ThreadPoolExecutor 進行多線程優化
        with concurrent.futures.ThreadPoolExecutor(max_workers=threadingCount) as executor:
            futures = [
                executor.submit(self.getPushbackPositionSplit, eachThreadTryTime, holeDetail, lengthCreateResult, startHoleList, magnificationAgain)
                for _ in range(threadingCount)
            ]
            concurrent.futures.wait(futures)

        maxCount = 0
        maxFinalPushback = {}

        for item in self._threadingResult:
                if item and item['count'] > maxCount:
                    maxCount = item['count']
                    maxFinalPushback = item['maxFinalPushback']


        return maxFinalPushback
    # 已修改
    def getPushbackPositionSplit(self, eachThreadTryTime, holeDetail: dict, lengthCreateResult, startHoleList, magnificationAgain: float):
        magnificationAllowance = 0.13
        rotateAngleAllowance = 7

        if len(holeDetail) < 4:
            return {}

        finalMaxHit = 0
        maxFinalPushback = {}

        # 事先準備隨機選取的數據，避免頻繁的 random 操作
        shuffledStartHoleList = random.sample(startHoleList, len(startHoleList))

        for tryCount in range(eachThreadTryTime):
            startHoleTag = shuffledStartHoleList[tryCount % len(shuffledStartHoleList)]  # 循環利用隨機選取的 hole tag
            lengthDetail = lengthCreateResult[startHoleTag]

            # 將 shuffle 降低到每個迴圈只進行一次，避免每次都進行 shuffle
            random.shuffle(lengthDetail[:3])  # 只隨機打亂前三個

            # 提前進行匹配檢查
            matchResult = self.findLengthMatch(lengthDetail[0], lengthDetail[1], lengthDetail[2])
            if len(matchResult) == 0:
                continue

            for match in matchResult:
                # 將標準件和原件的匹配提前處理，減少重複計算
                standStartHole = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[match["match12"]["from"]]]
                startHole = holeDetail[startHoleTag].copy()

                standLength1 = match["match12"]["length_1"]
                standLength2 = match["match12"]["length_2"]
                standLength3 = match["match13"]["length_2"]

                standHole1 = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[standLength1["to"]]]
                standHole2 = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[standLength2["to"]]]
                standHole3 = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[standLength3["to"]]]

                pushbackResult = {}
                doneIdList = []
                doneTagList = []
                magnificationList = []
                rotateAngleList = []

                # 減少不必要的複製操作，儘量只在需要的地方進行 copy
                startHole["tag"] = match["match12"]["from"]
                pushbackResult[startHoleTag] = startHole

                pushbackResult[lengthDetail[0]['to']] = holeDetail[lengthDetail[0]['to']].copy()
                pushbackResult[lengthDetail[0]['to']]["tag"] = match["match12"]["length_1"]["to"]

                pushbackResult[lengthDetail[1]['to']] = holeDetail[lengthDetail[1]['to']].copy()
                pushbackResult[lengthDetail[1]['to']]["tag"] = match["match12"]["length_2"]["to"]

                pushbackResult[lengthDetail[2]['to']] = holeDetail[lengthDetail[2]['to']].copy()
                pushbackResult[lengthDetail[2]['to']]["tag"] = match["match13"]["length_2"]["to"]

                doneTagList.extend([match["match12"]["from"], match["match12"]["length_1"]["to"],
                                    match["match12"]["length_2"]["to"], match["match13"]["length_2"]["to"]])

                doneIdList.extend([startHoleTag, lengthDetail[0]['to'], lengthDetail[1]['to'], lengthDetail[2]['to']])

                # 計算放大率和旋轉角度的增量，優化計算流程
                magnificationList.extend([
                    (startHole["width"] / standStartHole["width"] + startHole["height"] / standStartHole["height"]) / 2,
                    (pushbackResult[lengthDetail[0]['to']]["width"] / standHole1["width"] + pushbackResult[lengthDetail[0]['to']]["height"] / standHole1["height"]) / 2,
                    (pushbackResult[lengthDetail[1]['to']]["width"] / standHole2["width"] + pushbackResult[lengthDetail[1]['to']]["height"] / standHole2["height"]) / 2,
                    (pushbackResult[lengthDetail[2]['to']]["width"] / standHole3["width"] + pushbackResult[lengthDetail[2]['to']]["height"] / standHole3["height"]) / 2
                ])

                rotateAngleList.extend([
                    WorkpieceCalculator.calAngle(standLength1["vector"], lengthDetail[0]["vector"], self._rollAngle, self._pitchAngle),
                    WorkpieceCalculator.calAngle(standLength2["vector"], lengthDetail[1]["vector"], self._rollAngle, self._pitchAngle),
                    WorkpieceCalculator.calAngle(standLength3["vector"], lengthDetail[2]["vector"], self._rollAngle, self._pitchAngle)
                ])

                # 提前終止不符合條件的情況
                if max(magnificationList) - min(magnificationList) > magnificationAllowance or \
                max(rotateAngleList) - min(rotateAngleList) > rotateAngleAllowance:
                    continue

                # 批次化匹配的流程
                for i in range(3, len(lengthDetail)):
                    # 建立一個關聯長度的副本
                    findRelationLengthList = relationLengthList.copy()
                    findLength3 = lengthDetail[i]  # 選取當前迴圈的長度

                    # 基於已知的兩個長度進行匹配
                    newMatch = self.findLengthMatchKnowStart(findRelationLengthList, findLength1, findLength2, findLength3)

                    # 如果沒有匹配結果，則繼續下一個迴圈
                    if len(newMatch) == 0:
                        continue

                    # 檢查新的匹配是否和之前的匹配一致，避免無效的重複匹配
                    if newMatch["match12"]["length_1"] != beforeMatch["match23"]['length_1']: 
                        continue
                    if newMatch['match12']['length_2'] != beforeMatch['match23']['length_2']:
                        continue

                    # 將匹配結果填入 pushbackResult 並更新對應的標籤
                    pushbackResult[findLength3['to']] = holeDetail[findLength3['to']].copy()
                    pushbackResult[findLength3['to']]["tag"] = newMatch["match23"]["length_2"]["to"]

                    # 從標準工件中找出對應的孔並計算放大率和旋轉角度
                    standHole = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[newMatch["match23"]["length_2"]["to"]]]
                    magnificationList.append((pushbackResult[findLength3['to']]["width"] / standHole["width"] + pushbackResult[findLength3['to']]["height"] / standHole["height"]) / 2)
                    rotateAngleList.append(WorkpieceCalculator.calAngle(newMatch["match23"]["length_2"]["vector"], findLength3["vector"], self._rollAngle, self._pitchAngle))

                    # 檢查放大率和旋轉角度差異是否在允許範圍內
                    magnificationGap = max(magnificationList) - min(magnificationList)
                    rotateAngleGap = max(rotateAngleList) - min(rotateAngleList)
                    if magnificationGap > magnificationAllowance or rotateAngleGap > rotateAngleAllowance:
                        stop = True  # 終止匹配
                        break

                    # 更新處理過的 ID 和標籤，準備進行下一輪匹配
                    doneIdList.append(findLength3['to'])
                    doneTagList.append(newMatch["match23"]["length_2"]["to"])

                    # 更新關聯長度列表，去除已匹配的長度
                    relationLengthList = [value for value in relationLengthList if value["length_1"]["to"] != newMatch["match12"]["length_1"]["to"]]

                    # 更新匹配前後的變數，準備進行下一輪匹配
                    beforeMatch = newMatch
                    findLength1 = findLength2
                    findLength2 = findLength3
                # 統計結果並更新最佳結果
                count = len(pushbackResult)
                if count > finalMaxHit:
                    finalMaxHit = count
                    maxFinalPushback = pushbackResult

                    if finalMaxHit == len(self._standWorkpiece.getHoleDetail()):
                        break

        print(finalMaxHit)

        # 收集結果
        if finalMaxHit >= len(self._standWorkpiece.getHoleDetail()) / 3:
            self._threadingResult.append({"count": finalMaxHit, "maxFinalPushback": maxFinalPushback})
        else:
            self._threadingResult.append({})

    # 已修改
    def getPushbackPosition(self, holeDetail: dict, magnificationAgain : float, tryTime : int = None, matrix : np.ndarray = None):
        if tryTime is None:
            tryTime = int(self._config['try_time'] * 100)

        magnificationAllowance = 0.13
        rotateAngleAllowance = 7

        if len(holeDetail) < 4 or self._standWorkpiece is None: return {}

        lengthCreateResult = self._lengthCreate(holeDetail)
        startHoleList = list(lengthCreateResult.keys())

        if matrix is not None:
            matrix_inv = np.linalg.inv(matrix)
        else:
            matrix_inv = None

        finalMaxHit = 0
        maxFinalPushback = {}

        # 只隨機打亂一次 startHoleList 和 lengthDetail，減少每次迴圈中的開銷
        random.shuffle(startHoleList)

        for tryCount in range(0, tryTime):

            startHoleTag = random.choice(startHoleList)
            lengthDetail = lengthCreateResult[startHoleTag]
            random.shuffle(lengthDetail)

            length1 = lengthDetail[0]
            length2 = lengthDetail[1]
            length3 = lengthDetail[2]

            matchResult = self.findLengthMatch(length1, length2, length3)
            if len(matchResult) == 0: continue

            for match in matchResult:

                standHoleDetail = self._standWorkpiece.getHoleDetail()
                standStartHole = standHoleDetail[self._standWorkpiece.getHoleIdTagConvertDict()[match["match12"]["from"]]]
                startHole = holeDetail[startHoleTag].copy()

                standLength1 = match["match12"]["length_1"]
                standLength2 = match["match12"]["length_2"]
                standLength3 = match["match13"]["length_2"]

                standHole1 = standHoleDetail[self._standWorkpiece.getHoleIdTagConvertDict()[standLength1["to"]]]
                standHole2 = standHoleDetail[self._standWorkpiece.getHoleIdTagConvertDict()[standLength2["to"]]]
                standHole3 = standHoleDetail[self._standWorkpiece.getHoleIdTagConvertDict()[standLength3["to"]]]

                pushbackResult = {}
                doneIdList = []
                doneTagList = []
                magnificationList = []
                rotateAngleList = []

                startHole["tag"] = match["match12"]["from"]
                pushbackResult[startHoleTag] = startHole

                pushbackResult[length1['to']] = holeDetail[length1['to']].copy()
                pushbackResult[length1['to']]["tag"] = match["match12"]["length_1"]["to"]

                pushbackResult[length2['to']] = holeDetail[length2['to']].copy()
                pushbackResult[length2['to']]["tag"] = match["match12"]["length_2"]["to"]

                pushbackResult[length3['to']] = holeDetail[length3['to']].copy()
                pushbackResult[length3['to']]["tag"] = match["match13"]["length_2"]["to"]

                doneTagList = [match["match12"]["from"], match["match12"]["length_1"]["to"],
                            match["match12"]["length_2"]["to"], match["match13"]["length_2"]["to"]]
                doneIdList = [startHoleTag, length1['to'], length2['to'], length3['to']]

                # 放大倍率計算
                magnificationList.append((startHole["width"] / standStartHole["width"] + startHole["height"] / standStartHole["height"]) / 2)
                magnificationList.append((pushbackResult[length1['to']]["width"] / standHole1["width"] + pushbackResult[length1['to']]["height"] / standHole1["height"]) / 2)
                magnificationList.append((pushbackResult[length2['to']]["width"] / standHole2["width"] + pushbackResult[length2['to']]["height"] / standHole2["height"]) / 2)
                magnificationList.append((pushbackResult[length3['to']]["width"] / standHole3["width"] + pushbackResult[length3['to']]["height"] / standHole3["height"]) / 2)

                # 緩存角度計算
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength1["vector"], length1["vector"], self._rollAngle, self._pitchAngle))
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength2["vector"], length2["vector"], self._rollAngle, self._pitchAngle))
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength3["vector"], length3["vector"], self._rollAngle, self._pitchAngle))

                magnificationMax = max(magnificationList)
                magnificationMin = min(magnificationList)
                rotateAngleMax = max(rotateAngleList)
                rotateAngleMin = min(rotateAngleList)

                if magnificationMax - magnificationMin > magnificationAllowance or rotateAngleMax - rotateAngleMin > rotateAngleAllowance:
                    continue

                findLength1 = length2
                findLength2 = length3
                findLength3 = {}

                beforeMatch = match
                stop = False

                relationLengthList = self._standWorkpiece.getLengthRelationship()[match["match12"]["from"]].copy()
                relationLengthList = [value for value in relationLengthList if value["length_1"]["to"] != match["match12"]["length_1"]["to"]]

                for i in range(3, len(lengthDetail)):

                    findRelationLengthList = relationLengthList.copy()
                    findLength3 = lengthDetail[i]

                    newMatch = self.findLengthMatchKnowStart(findRelationLengthList, findLength1, findLength2, findLength3)

                    if len(newMatch) == 0: continue
                    if newMatch["match12"]["length_1"] != beforeMatch["match23"]['length_1']: continue
                    if newMatch['match12']['length_2'] != beforeMatch['match23']['length_2']: continue

                    standHole = standHoleDetail[self._standWorkpiece.getHoleIdTagConvertDict()[newMatch["match23"]["length_2"]["to"]]]

                    pushbackResult[findLength3['to']] = holeDetail[findLength3['to']].copy()
                    pushbackResult[findLength3['to']]["tag"] = newMatch["match23"]["length_2"]["to"]
                    pushbackResult[findLength3['to']]["status"] = "hole_stage2"

                    # 更新放大倍率與旋轉角度
                    newMagnification = (pushbackResult[findLength3['to']]["width"] / standHole["width"] + pushbackResult[findLength3['to']]["height"] / standHole["height"]) / 2
                    magnificationList.append(newMagnification)
                    rotateAngleList.append(WorkpieceCalculator.calAngle(newMatch["match23"]["length_2"]["vector"], findLength3["vector"], self._rollAngle, self._pitchAngle))

                    magnificationMax = max(magnificationMax, newMagnification)
                    magnificationMin = min(magnificationMin, newMagnification)
                    rotateAngleMax = max(rotateAngleMax, rotateAngleList[-1])
                    rotateAngleMin = min(rotateAngleMin, rotateAngleList[-1])

                    if magnificationMax - magnificationMin > magnificationAllowance or rotateAngleMax - rotateAngleMin > rotateAngleAllowance:
                        stop = True
                        break

                    doneIdList.append(findLength3['to'])
                    doneTagList.append(newMatch["match23"]["length_2"]["to"])

                    relationLengthList = [value for value in relationLengthList if value["length_1"]["to"] != newMatch["match12"]["length_1"]["to"]]
                    beforeMatch = newMatch
                    findLength1 = findLength2
                    findLength2 = findLength3

                if stop: continue

                # pushback stage 3 - pushback no matching hole
                if len(pushbackResult) != len(self._standWorkpiece.getHoleDetail()):
                    magnification = sum(magnificationList) / len(magnificationList)
                    rotateAngle = sum(rotateAngleList) / len(rotateAngleList)

                    result = self.pushback(holeDetail, pushbackResult, doneTagList, doneIdList, magnification, rotateAngle, matrix_inv)
                    if result == {}: continue

                    count = result['count']
                    pushbackResult = result['pushback']
                else:
                    count = len(pushbackResult)

                if count > finalMaxHit:
                    finalMaxHit = count
                    maxFinalPushback = pushbackResult

                    if finalMaxHit == len(self._standWorkpiece.getHoleDetail()):
                        break

        if finalMaxHit >= len(self._standWorkpiece.getHoleDetail()) / 3:
            return maxFinalPushback

        return {}

    # 已修改
    def getPushbackFrame(self, frame: np.ndarray, hole_result_convert: dict, nowHole: int = -1) -> np.ndarray:
        frame = frame.copy()

        x_move = 25
        y_move = -25

        # 提前計算寬度、字型大小等參數
        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)
        tf = max(lw - 1, 1)  # 字型粗細

        # 直接在迴圈外判斷是否需要對特定的 `nowHole` 進行處理
        if nowHole != -1:
            # 如果指定了特定的 nowHole，過濾只處理這個洞
            hole_result_convert = {k: v for k, v in hole_result_convert.items() if int(v['tag']) == nowHole}

        for _, value in hole_result_convert.items():
            tag = value['tag'].zfill(2)  # 用兩位數字來填充 tag
            coordinate = value['coordinate']

            # 繪製矩形框
            color = (50, 108, 66)
            cv2.rectangle(
                frame,
                (int(coordinate['left_top'][0]), int(coordinate['left_top'][1])),
                (int(coordinate['right_bottom'][0]), int(coordinate['right_bottom'][1])),
                color, lw
            )

            # 計算中間位置並移動
            middle_x, middle_y = int(coordinate['middle'][0]), int(coordinate['middle'][1])
            p1 = (middle_x + x_move, middle_y + y_move)
            p2 = (middle_x - x_move, middle_y - y_move)

            # 提前計算文字的寬度和高度
            w, h = cv2.getTextSize(str(tag), 0, fontScale=lw / 3, thickness=tf)[0]

            # 判斷是否文字要放在矩形框內或外
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            # 繪製文字背景和文字
            cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # 填充矩形
            cv2.putText(frame, tag, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, 
                        (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

        return frame
