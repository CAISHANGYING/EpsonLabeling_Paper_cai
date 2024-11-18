import random
from math import pi, sin, cos

import cv2
import numpy as np

from workpiece.stand_workpiece import StandWorkpiece
from workpiece.calculator import WorkpieceCalculator

import timeit

import threading

class PushBackModel:

    _config : dict = None
    _standWorkpiece : StandWorkpiece = None

    _threadingResult = []

    def __init__(self, config : dict, standWorkpiece : StandWorkpiece = None ) -> None:

        self._config = config
        self._standWorkpiece = standWorkpiece

    def setConfig(self, config : dict) -> None:

        self._config = config

    def setStandWorkpiece(self, standWorkpiece : StandWorkpiece) -> None:

        self._standWorkpiece = standWorkpiece

    def getStandWorkpieceName(self) -> str:

        return self._standWorkpiece.getName() if self._standWorkpiece is not None else "None"

    def _isAngleMatched(self, angle : float, mathAngle : float) -> bool:

        differentValue = 5

        if angle < 0 and mathAngle < 0:

            if (mathAngle - differentValue) >= angle and angle >= (mathAngle + differentValue):

                return True

        elif (mathAngle - differentValue) <= angle and angle <= (mathAngle + differentValue):

            return True

        return False

    def _isRatioMatched(self, ratio : float, mathRatio : float) -> bool:

        differentValue = self._config['ratio_match_deviation'] * (self._standWorkpiece.getShape()[0] + self._standWorkpiece.getShape()[1]) / 2

        if (mathRatio - differentValue) <= ratio and ratio <= (mathRatio + differentValue):

            return True

        return False

    def _isCoverGapMatched(self, gap : int, matchGap : int) -> bool:

        differentValue = 1

        if (matchGap - differentValue) <= gap and gap <= (matchGap + differentValue):

            return True

        return False

    def _countCover(self, holeDetail : dict, middle : list, lengthVector : list) -> int:

        count = 0

        for tag, value in holeDetail.items():

            if (0 < WorkpieceCalculator.calLength( [value['coordinate']['middle'][0] - middle[0], value['coordinate']['middle'][1] - middle[1]])) and \
                    (WorkpieceCalculator.calLength([value['coordinate']['middle'][0] - middle[0], value['coordinate']['middle'][1] - middle[1]]) <= WorkpieceCalculator.calLength(lengthVector)) :

                count += 1

        return count

    def _lengthCreate(self, holeDetail : dict) -> dict:

        lengthDict = {}

        for tag1, value1 in holeDetail.items():

            lengthDict[tag1] = []
            middle1 = value1['coordinate']['middle']

            for tag2, value2 in holeDetail.items():

                middle2 = value2['coordinate']['middle']

                if tag1 != tag2:

                    lengthVector = [middle2[0] - middle1[0], middle2[1] - middle1[1]]

                    lengthDict[tag1].append({
                        'to' : tag2,
                        'vector': lengthVector,
                        'cover': self._countCover(holeDetail, middle1, lengthVector)
                    })

        return lengthDict

    def _degressToRadians(self, degress : float) -> float:

        return degress * pi / 180

    def findMatch(self, angle : float, ratio : float, coverGap : int) -> dict:

        resultDict = {}

        for key, value in self._standWorkpiece.getLengthRelationship().items():

            for item in value:

                if self._isAngleMatched(angle, item['angle']) and \
                    self._isRatioMatched(ratio, item['ratio']) and \
                    self._isCoverGapMatched(coverGap, item['cover']):

                    if len(resultDict) == 0:

                        resultDict = item

                    elif (angle - resultDict['angle'] > angle - item['angle']) and \
                            (ratio - resultDict['ratio'] > ratio - item['ratio']) :

                        resultDict = item

        return resultDict

    def inStandWorkpieceRelationship(self, relationship : list, length1 : dict, length2 : dict) -> dict:

        length1Vector = length1['vector']
        length2Vector = length2['vector']

        angle = WorkpieceCalculator.calAngle(length1Vector, length2Vector)
        ratio = WorkpieceCalculator.calRatio(length1Vector, length2Vector)
        coverGap = length1['cover'] - length2['cover']

        for info in relationship:

            if self._isAngleMatched(angle, info['angle']) and \
                self._isRatioMatched(ratio, info['ratio']) and \
                self._isCoverGapMatched(coverGap, info['cover_gap']):

                return info

        return {}

    def findLengthMatch(self, length1 : dict, length2 : dict, length3 : dict) -> list:

        length1Vector = length1['vector']
        length2Vector = length2['vector']

        angle12 = WorkpieceCalculator.calAngle(length1Vector, length2Vector)
        ratio12 = WorkpieceCalculator.calRatio(length1Vector, length2Vector)
        coverGap12 = length1['cover'] - length2['cover']

        findKeyList = []

        for key, value in self._standWorkpiece.getLengthRelationship().items():

            for info in value:

                if self._isAngleMatched(angle12, info['angle']) and \
                    self._isRatioMatched(ratio12, info['ratio']) and \
                    self._isCoverGapMatched(coverGap12, info['cover_gap']):

                    findKeyList.append(key)

                    break

        if len(findKeyList) == 0: return []


        verifyList = []

        for key in findKeyList:

            standWorkpieceLengthRelationship = self._standWorkpiece.getLengthRelationship()[key].copy()

            match12 = self.inStandWorkpieceRelationship(standWorkpieceLengthRelationship, length1, length2)

            if len(match12) == 0: continue

            standWorkpieceLengthRelationship.remove(match12)

            match13 = self.inStandWorkpieceRelationship(standWorkpieceLengthRelationship, length1, length3)

            if len(match13) == 0: continue

            standWorkpieceLengthRelationship.remove(match13)

            match23 = self.inStandWorkpieceRelationship(standWorkpieceLengthRelationship, length2, length3)

            if len(match23) == 0: continue

            if match23['length_1']['to'] != match12['length_2']['to'] or match23['length_2']['to'] != match13['length_2']['to']: continue

            verifyList.append({
                'match12': match12,
                'match13': match13,
                'match23': match23
            })

        return verifyList

    def findLengthMatchKnowStart(self, findRelationship : list, length1 : dict, length2 : dict, length3 : dict) -> dict:

        match12 = self.inStandWorkpieceRelationship(findRelationship, length1, length2)

        if len(match12) == 0: return {}

        findRelationship.remove(match12)

        match13 = self.inStandWorkpieceRelationship(findRelationship, length1, length3)

        if len(match13) == 0: return {}

        findRelationship.remove(match13)

        match23 = self.inStandWorkpieceRelationship(findRelationship, length2, length3)

        if len(match23) == 0: return {}

        if match23['length_1']['to'] != match12['length_2']['to'] or match23['length_2']['to'] != match13['length_2']['to']: return {}

        return {
            'match12': match12,
            'match13': match13,
            'match23': match23
        }

    def findNearHole(self, holeDetail : dict, doneIdList : list, magnification : float, position : list) -> dict:

        allowance = self._config['hole_match_deviation'] * (self._standWorkpiece.getShape()[0] + self._standWorkpiece.getShape()[1]) * magnification * 1.1 / 2

        resultId = ""
        result = {}

        keys = [key for key in holeDetail.keys() if key not in doneIdList]

        for key in keys:

            hole = holeDetail[key]

            if (((hole["coordinate"]["left_top"][0] - allowance) <= position[0]) and (position[0] <= (hole["coordinate"]["right_bottom"][0] + allowance))) and \
                (((hole["coordinate"]["left_top"][1] - allowance) <= position[1]) and (position[1] <= (hole["coordinate"]["right_bottom"][1] + allowance))):

                result = hole.copy()
                resultId = key

                result["status"] = "hole_match"

                break

        if resultId == "" : return {}

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

    def pushback(self, holeDetail: dict, nowPushback : dict, doneTagList : list, doneIdList : list, magnification : float, rotateAngle : float):

        count = len(nowPushback)

        start1 = doneTagList[0]
        start2 = doneTagList[len(doneTagList) - 1]

        startHole = [result for result in nowPushback.values() if result["tag"] == start1][0]

        length_ = [result for result in self._standWorkpiece.getLengthRelationship()[start1] if
                   result['length_1']['to'] == start2 and result['length_2']['to'] not in doneTagList]

        for length in length_:

            i = length['length_2']['to']
            vector = length['length_2']['vector']

            bv1 = vector[0]
            bv2 = vector[1]

            v1 = (bv1 * cos(self._degressToRadians(rotateAngle)) - bv2 * sin(self._degressToRadians(rotateAngle)))
            v2 = (bv1 * sin(self._degressToRadians(rotateAngle)) + bv2 * cos(self._degressToRadians(rotateAngle)))

            v1 = v1 * magnification
            v2 = v2 * magnification

            middle = [startHole["coordinate"]["middle"][0] + v1, startHole["coordinate"]["middle"][1] + v2]

            if not self.verifyHole(nowPushback, middle): return {}

            nearHole = self.findNearHole(holeDetail, doneIdList, magnification, middle)

            if len(nearHole) != 0:

                nearHoleId = nearHole['id']
                nearHoleResult = nearHole['result']

                doneIdList.append(nearHoleId)
                doneTagList.append(str(i))

                nowPushback[nearHoleId] = nearHoleResult
                nowPushback[nearHoleId]["tag"] = str(i)

            else:

                width = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[str(i)]]['width'] * magnification
                height = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[str(i)]]['height'] * magnification

                doneTagList.append(str(i))

                nowPushback[str(i)] = {
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

        return {
            "count": count,
            "pushback": nowPushback,
        }


    def getPushbackPositionThreading(self, holeDetail: dict, magnificationAgain : float, threadingCount:int):

        self._threadingResult = []

        tryTime = int(self._config['try_time'] * 100)

        eachThreadTryTime = tryTime // threadingCount

        if len(holeDetail) < 4: return {}

        lengthCreateResult = self._lengthCreate(holeDetail)

        startHoleList = list(lengthCreateResult.keys())

        threadingList = []

        for i in range(0, threadingCount):

            threadingList.append(threading.Thread(target=self.getPushbackPositionSplit, args=(eachThreadTryTime, holeDetail, lengthCreateResult, startHoleList, magnificationAgain)))

        for i in range(0, threadingCount):

            threadingList[i].start()

        for i in range(0, threadingCount):

            threadingList[i].join()

        maxCount = 0
        maxFinalPushback = {}

        for item in self._threadingResult:

            if item == {}: continue

            if item['count'] > maxCount :

                maxCount = item['count']
                maxFinalPushback = item['maxFinalPushback']


        return maxFinalPushback

    def getPushbackPositionSplit(self, eachThreadTryTime, holeDetail: dict, lengthCreateResult, startHoleList, magnificationAgain : float):

        magnificationAllowance = 0.13
        rotateAngleAllowance = 7

        if len(holeDetail) < 4: return {}

        finalMaxHit = 0
        maxFinalPushback = {}

        for tryCount in range(0, eachThreadTryTime):

            startHoleTag = random.choice(startHoleList)
            lengthDetail = lengthCreateResult[startHoleTag]

            random.shuffle(lengthDetail)


            length1 = lengthDetail[0]
            length2 = lengthDetail[1]
            length3 = lengthDetail[2]

            matchResult = self.findLengthMatch(length1, length2, length3)

            if len(matchResult) == 0: continue

            for match in matchResult:

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

                magnificationList.append((startHole["width"] / standStartHole["width"] + startHole["height"] / standStartHole["height"]) / 2)
                magnificationList.append((pushbackResult[length1['to']]["width"] / standHole1["width"] + pushbackResult[length1['to']]["height"] / standHole1["height"]) / 2)
                magnificationList.append((pushbackResult[length2['to']]["width"] / standHole2["width"] + pushbackResult[length2['to']]["height"] / standHole2["height"]) / 2)
                magnificationList.append((pushbackResult[length3['to']]["width"] / standHole3["width"] + pushbackResult[length3['to']]["height"] / standHole3["height"]) / 2)

                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength1["vector"], length1["vector"]))
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength2["vector"], length2["vector"]))
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength3["vector"], length3["vector"]))

                magnificationGap = max(magnificationList) - min(magnificationList)
                rotateAngleGap = max(rotateAngleList) - min(rotateAngleList)

                if magnificationGap > magnificationAllowance or rotateAngleGap > rotateAngleAllowance: continue


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

                    pushbackResult[findLength3['to']] = holeDetail[findLength3['to']].copy()
                    pushbackResult[findLength3['to']]["tag"] = newMatch["match23"]["length_2"]["to"]

                    standHole = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[newMatch["match23"]["length_2"]["to"]]]

                    magnificationList.append((pushbackResult[findLength3['to']]["width"] / standHole["width"] + pushbackResult[findLength3['to']]["height"] / standHole["height"]) / 2)
                    rotateAngleList.append(WorkpieceCalculator.calAngle(newMatch["match23"]["length_2"]["vector"], findLength3["vector"]))

                    magnificationGap = max(magnificationList) - min(magnificationList)
                    rotateAngleGap = max(rotateAngleList) - min(rotateAngleList)

                    if magnificationGap > magnificationAllowance or rotateAngleGap > rotateAngleAllowance:

                        stop = True

                        break

                    doneIdList.append(findLength3['to'])
                    doneTagList.append(newMatch["match23"]["length_2"]["to"])

                    relationLengthList = [value for value in relationLengthList if value["length_1"]["to"] != newMatch["match12"]["length_1"]["to"]]

                    beforeMatch = newMatch

                    findLength1 = findLength2
                    findLength2 = findLength3

                if stop: continue

                result = {}
                count = 0

                if len(pushbackResult) != len(self._standWorkpiece.getHoleDetail()):

                    magnification = sum(magnificationList) / len(magnificationList)
                    rotateAngle = sum(rotateAngleList) / len(rotateAngleList)

                    result = self.pushback(holeDetail, pushbackResult, doneTagList, doneIdList, magnification, rotateAngle)

                    if len(result) == 0: continue

                    count = result['count']
                    pushbackResult = result['pushback']

                else:

                    count = len(pushbackResult)

                if count > finalMaxHit:

                    finalMaxHit = count
                    maxFinalPushback = pushbackResult

                    if finalMaxHit == len(self._standWorkpiece.getHoleDetail()):

                        break

        print(finalMaxHit)

        if finalMaxHit >= len(self._standWorkpiece.getHoleDetail()) / 3:

            self._threadingResult.append({"count":finalMaxHit, "maxFinalPushback":maxFinalPushback})

            return

        self._threadingResult.append({})

    def getPushbackPosition(self, holeDetail: dict, magnificationAgain : float, tryTime : int = None):

        if tryTime is None:

            tryTime = int(self._config['try_time'] * 100)

        magnificationAllowance = 0.13
        rotateAngleAllowance = 7

        if len(holeDetail) < 4 or self._standWorkpiece is None: return {}

        lengthCreateResult = self._lengthCreate(holeDetail)

        startHoleList = list(lengthCreateResult.keys())


        finalMaxHit = 0
        maxFinalPushback = {}

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

                magnificationList.append((startHole["width"] / standStartHole["width"] + startHole["height"] / standStartHole["height"]) / 2)
                magnificationList.append((pushbackResult[length1['to']]["width"] / standHole1["width"] + pushbackResult[length1['to']]["height"] / standHole1["height"]) / 2)
                magnificationList.append((pushbackResult[length2['to']]["width"] / standHole2["width"] + pushbackResult[length2['to']]["height"] / standHole2["height"]) / 2)
                magnificationList.append((pushbackResult[length3['to']]["width"] / standHole3["width"] + pushbackResult[length3['to']]["height"] / standHole3["height"]) / 2)

                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength1["vector"], length1["vector"]))
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength2["vector"], length2["vector"]))
                rotateAngleList.append(WorkpieceCalculator.calAngle(standLength3["vector"], length3["vector"]))

                magnificationGap = max(magnificationList) - min(magnificationList)
                rotateAngleGap = max(rotateAngleList) - min(rotateAngleList)

                if magnificationGap > magnificationAllowance or rotateAngleGap > rotateAngleAllowance: continue


                findLength1 = length2
                findLength2 = length3
                findLength3 = {}

                beforeMatch = match

                stop = False

                relationLengthList = self._standWorkpiece.getLengthRelationship()[match["match12"]["from"]].copy()

                relationLengthList = [value for value in relationLengthList if value["length_1"]["to"] != match["match12"]["length_1"]["to"]]

                ## 1

                for i in range(3, len(lengthDetail)):

                    findRelationLengthList = relationLengthList.copy()

                    findLength3 = lengthDetail[i]

                    newMatch = self.findLengthMatchKnowStart(findRelationLengthList, findLength1, findLength2, findLength3)

                    if len(newMatch) == 0: continue

                    if newMatch["match12"]["length_1"] != beforeMatch["match23"]['length_1']: continue
                    if newMatch['match12']['length_2'] != beforeMatch['match23']['length_2']: continue

                    pushbackResult[findLength3['to']] = holeDetail[findLength3['to']].copy()
                    pushbackResult[findLength3['to']]["tag"] = newMatch["match23"]["length_2"]["to"]

                    standHole = self._standWorkpiece.getHoleDetail()[self._standWorkpiece.getHoleIdTagConvertDict()[newMatch["match23"]["length_2"]["to"]]]

                    magnificationList.append((pushbackResult[findLength3['to']]["width"] / standHole["width"] + pushbackResult[findLength3['to']]["height"] / standHole["height"]) / 2)
                    rotateAngleList.append(WorkpieceCalculator.calAngle(newMatch["match23"]["length_2"]["vector"], findLength3["vector"]))

                    magnificationGap = max(magnificationList) - min(magnificationList)
                    rotateAngleGap = max(rotateAngleList) - min(rotateAngleList)

                    if magnificationGap > magnificationAllowance or rotateAngleGap > rotateAngleAllowance:

                        stop = True

                        break

                    doneIdList.append(findLength3['to'])
                    doneTagList.append(newMatch["match23"]["length_2"]["to"])

                    relationLengthList = [value for value in relationLengthList if value["length_1"]["to"] != newMatch["match12"]["length_1"]["to"]]

                    beforeMatch = newMatch

                    findLength1 = findLength2
                    findLength2 = findLength3

                if stop: continue

                result = {}
                count = 0

                if len(pushbackResult) != len(self._standWorkpiece.getHoleDetail()):

                    magnification = sum(magnificationList) / len(magnificationList)
                    rotateAngle = sum(rotateAngleList) / len(rotateAngleList)

                    result = self.pushback(holeDetail, pushbackResult, doneTagList, doneIdList, magnification, rotateAngle)

                    if len(result) == 0: continue

                    count = result['count']
                    pushbackResult = result['pushback']

                else:

                    count = len(pushbackResult)

                if count > finalMaxHit:

                    finalMaxHit = count
                    maxFinalPushback = pushbackResult

                    if finalMaxHit == len(self._standWorkpiece.getHoleDetail()):

                        break

        # print(finalMaxHit)

        if finalMaxHit >= len(self._standWorkpiece.getHoleDetail()) / 3:

            return  maxFinalPushback

        return {}

    def getPushbackFrame(self, frame : np.ndarray, hole_result_convert : dict, nowHole : int = -1) -> np.ndarray:

        frame = frame.copy()

        x_move = 25
        y_move = -25

        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

        for _, value in hole_result_convert.items():

            tag = value['tag']
            coordinate = value['coordinate']

            if nowHole == -1 or int(tag) == nowHole:

                tag = tag.zfill(2)
                color = (50, 108, 66)

                cv2.rectangle(frame, (int(coordinate['left_top'][0]), int(coordinate['left_top'][1])),
                              (int(coordinate['right_bottom'][0]), int(coordinate['right_bottom'][1])), color, 2)

                p1, p2 = ((int(coordinate['middle'][0]) + x_move), (int(coordinate['middle'][1]) + y_move)), (
                (int(coordinate['middle'][0]) - x_move), (int(coordinate['middle'][1]) - y_move))

                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(str(tag), 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

                cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, tag, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                            lw / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

        return frame