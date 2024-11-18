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

class PushBackModel:

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

    # 已修改: 基本角度匹配函數
    def _isAngleMatched(self, angle: float, mathAngle: float) -> bool:
        differentValue = 4
        if angle < 0 and mathAngle < 0:
            return (mathAngle - differentValue) <= angle <= (mathAngle + differentValue)
        return (mathAngle - differentValue) <= angle <= (mathAngle + differentValue)

    # 已修改: 基本比例匹配函數
    def _isRatioMatched(self, ratio: float, mathRatio: float) -> bool:
        shape = self._standWorkpiece.getShape()
        differentValue = self._config['ratio_match_deviation'] * (shape[0] + shape[1]) / 2
        return (mathRatio - differentValue) <= ratio <= (mathRatio + differentValue)

    # 已修改: 基本覆蓋範圍匹配函數
    def _isCoverGapMatched(self, gap: int, matchGap: int) -> bool:
        differentValue = 1
        return (matchGap - differentValue) <= gap <= (matchGap + differentValue)

    # ----------------------------------------
    # 新增條件: 距離匹配
    def _isDistanceMatched(self, middle1: list, middle2: list, minDistance: float) -> bool:
        distance = np.sqrt((middle1[0] - middle2[0]) ** 2 + (middle1[1] - middle2[1]) ** 2)
        return distance >= minDistance

    # 新增條件: 角度變化梯度匹配
    def _isGradientMatched(self, previousAngle: float, currentAngle: float, maxGradient: float) -> bool:
        return abs(currentAngle - previousAngle) <= maxGradient
    # ----------------------------------------

    def _countCover(self, holeDetail: dict, middle: list, lengthVector: list) -> int:
        count = 0
        lengthVectorLen = WorkpieceCalculator.calLength(lengthVector)

        for tag, value in holeDetail.items():
            diffVector = [value['coordinate']['middle'][0] - middle[0], value['coordinate']['middle'][1] - middle[1]]
            diffLength = WorkpieceCalculator.calLength(diffVector)

            if 0 < diffLength <= lengthVectorLen:
                count += 1

        return count

    def _lengthCreate(self, holeDetail: dict) -> dict:
        lengthDict = {}
        coordinates = {tag: value['coordinate']['middle'] for tag, value in holeDetail.items()}

        for tag1, middle1 in coordinates.items():
            lengthDict[tag1] = []
            for tag2, middle2 in coordinates.items():
                if tag1 != tag2:
                    lengthVector = [middle2[0] - middle1[0], middle2[1] - middle1[1]]
                    lengthDict[tag1].append({
                        'to': tag2,
                        'vector': lengthVector,
                        'cover': self._countCover(holeDetail, middle1, lengthVector)
                    })

        return lengthDict

    def _degressToRadians(self, degrees: float) -> float:
        return degrees * pi / 180

    def findMatch(self, angle: float, ratio: float, coverGap: int) -> dict:
        lengthRelationship = self._standWorkpiece.getLengthRelationship()
        resultDict = {}
        minDiff = float('inf')

        for key, value in lengthRelationship.items():
            for item in value:
                if (self._isAngleMatched(angle, item['angle']) and
                    self._isRatioMatched(ratio, item['ratio']) and
                    self._isCoverGapMatched(coverGap, item['cover'])):

                    # 可選條件: 檢查距離匹配
                    # if not self._isDistanceMatched(middle1, middle2, self._config['min_distance']):
                    #     continue

                    # 可選條件: 檢查角度變化梯度
                    # if not self._isGradientMatched(previousAngle, currentAngle, self._config['max_gradient']):
                    #     continue

                    angleDiff = (angle - item['angle']) ** 2
                    ratioDiff = (ratio - item['ratio']) ** 2
                    totalDiff = angleDiff + ratioDiff

                    if totalDiff < minDiff:
                        minDiff = totalDiff
                        resultDict = item

        return resultDict

    def inStandWorkpieceRelationship(self, relationship: list, length1: dict, length2: dict) -> dict:
        length1Vector = length1['vector']
        length2Vector = length2['vector']

        angle = WorkpieceCalculator.calAngle(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        ratio = WorkpieceCalculator.calRatio(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        coverGap = length1['cover'] - length2['cover']

        for info in relationship:
            if (self._isAngleMatched(angle, info['angle']) and 
                self._isRatioMatched(ratio, info['ratio']) and 
                self._isCoverGapMatched(coverGap, info['cover_gap'])):

                # 可選條件: 檢查距離
                # if not self._isDistanceMatched(middle1, middle2, self._config['min_distance']):
                #     continue

                return info

        return {}

    def findLengthMatch(self, length1: dict, length2: dict, length3: dict) -> list:
        length1Vector = length1['vector']
        length2Vector = length2['vector']
        
        angle12 = WorkpieceCalculator.calAngle(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        ratio12 = WorkpieceCalculator.calRatio(length1Vector, length2Vector, self._rollAngle, self._pitchAngle)
        coverGap12 = length1['cover'] - length2['cover']

        findKeyList = []
        lengthRelationship = self._standWorkpiece.getLengthRelationship()

        for key, value in lengthRelationship.items():
            for info in value:
                if (self._isAngleMatched(angle12, info['angle']) and 
                    self._isRatioMatched(ratio12, info['ratio']) and 
                    self._isCoverGapMatched(coverGap12, info['cover_gap'])):

                    # 可選條件: 檢查距離
                    # if not self._isDistanceMatched(middle1, middle2, self._config['min_distance']):
                    #     continue

                    findKeyList.append(key)
                    break

        if not findKeyList:
            return []

        verifyList = []

        for key in findKeyList:
            standWorkpieceLengthRelationship = lengthRelationship[key]

            match12 = self.inStandWorkpieceRelationship(standWorkpieceLengthRelationship, length1, length2)
            if not match12:
                continue
            
            tempRelationship = standWorkpieceLengthRelationship[:]
            tempRelationship.remove(match12)

            match13 = self.inStandWorkpieceRelationship(tempRelationship, length1, length3)
            if not match13:
                continue
            
            tempRelationship.remove(match13)

            match23 = self.inStandWorkpieceRelationship(tempRelationship, length2, length3)
            if not match23:
                continue

            if (match23['length_1']['to'] != match12['length_2']['to'] or 
                match23['length_2']['to'] != match13['length_2']['to']):
                continue

            verifyList.append({
                'match12': match12,
                'match13': match13,
                'match23': match23
            })

        return verifyList

    # ----------------------------------------
    # 新增: 數據正規化方法，用於提高匹配效果
    def _normalizeData(self, data: float, minValue: float, maxValue: float) -> float:
        return (data - minValue) / (maxValue - minValue)

    # 可選: 對角度和比例進行正規化處理
    # angleNormalized = self._normalizeData(angle, 0, 180)
    # ratioNormalized = self._normalizeData(ratio, 0, 1)
    # ----------------------------------------

    def findLengthMatchKnowStart(self, findRelationship: list, length1: dict, length2: dict, length3: dict) -> dict:
        match12 = self.inStandWorkpieceRelationship(findRelationship, length1, length2)
        if not match12:
            return {}

        findRelationship.remove(match12)

        match13 = self.inStandWorkpieceRelationship(findRelationship, length1, length3)
        if not match13:
            return {}

        findRelationship.remove(match13)

        match23 = self.inStandWorkpieceRelationship(findRelationship, length2, length3)
        if not match23:
            return {}

        if not (match23['length_1']['to'] == match12['length_2']['to'] and match23['length_2']['to'] == match13['length_2']['to']):
            return {}

        return {
            'match12': match12,
            'match13': match13,
            'match23': match23
        }

