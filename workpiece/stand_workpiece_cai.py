# stand_workpiece.py
import json
import numpy as np
from detect.hole_detect_process import HoleDetectConvector

class StandWorkpiece:
    _name: str = ""
    _shape: list = []
    _realImage: list = []
    _holeDetail: dict = {}
    _lengthDetail: dict = {}
    _holeIdTagConvertDict: dict = {}
    _lengthRelationship: dict = {}
    _boundaryDetail: list = []

    def __init__(self, jsonData: dict) -> None:
        self.loadFromJson(jsonData)

    def loadFromJson(self, jsonData: dict) -> None:
        self._name = jsonData['name']
        self._shape = jsonData['shape']
        self._realImage = jsonData['real_image']
        self._holeDetail = jsonData['hole_detail']
        self._lengthDetail = jsonData['length_detail']
        self._lengthRelationship = jsonData['length_relationship']
        self._generateConvert()
        if 'boundary_detail' in jsonData:
            self._boundaryDetailConvert(jsonData['boundary_detail'])

    def _generateConvert(self) -> None:
        self._holeIdTagConvertDict = {}
        for holeId, holeValue in self._holeDetail.items():
            self._holeIdTagConvertDict[holeValue["tag"]] = holeId

    def _boundaryDetailConvert(self, boundaryDict) -> None:
        boundaryDetail = []
        for key in boundaryDict:
            boundaryDetail.append(boundaryDict[key]['coordinate']['middle'])
        self._boundaryDetail = HoleDetectConvector.sortBoundary(boundaryDetail)

    def getName(self) -> str: return self._name
    def getShape(self) -> list: return self._shape
    def getRealImage(self) -> list: return self._realImage
    def getHoleDetail(self) -> dict: return self._holeDetail
    def getLengthDetail(self) -> dict: return self._lengthDetail
    def getHoleIdTagConvertDict(self) -> dict: return self._holeIdTagConvertDict
    def getLengthRelationship(self) -> dict: return self._lengthRelationship
    def getBoundaryDetail(self) -> list:
        return self._boundaryDetail

    def getCenterCoordinateList(self) -> (list, list):
        holeCenterCoordinateList = []
        holeIdList = []
        for key in self._holeDetail.keys():
            holeCenterCoordinateList.append(self._holeDetail[key]['coordinate']['middle'])
            holeIdList.append(self._holeDetail[key]['tag'])
        return holeCenterCoordinateList, holeIdList
