import cv2
import numpy as np
import torch
import math
import tool

class AngleDetectModel:

    def __init__(self):

        pass

    def _degreeCAL(self, vx, vy):

        rad = math.atan2(vy, vx)

        if rad < 0:

            return 90 + math.degrees(rad) + 90

        else:

            return math.degrees(rad)

    def _line_pro(self, contours):

        fitline_out = cv2.fitLine(contours, cv2.DIST_L2, 0, 0.01, 0.01)

        return fitline_out[0], fitline_out[1]

    def detect(self, standTool : tool.stand_tool.StandTool, mask : np.ndarray) -> (float, float):

        # 拿標準工件 mask，計算角度
        contours_std, _ = cv2.findContours(standTool.get_mask(), 3, 2)
        contours_std_tmp = contours_std[0]

        vx_std, vy_std = self._line_pro(contours_std_tmp)
        degree_std = self._degreeCAL(vx_std, vy_std)

        # 拿現在攝像機拍的工件 mask，計算角度
        contours_com, _ = cv2.findContours(mask, 3, 2)
        contours_com_tmp = contours_com[0]

        vx_com, vy_com = self._line_pro(contours_com_tmp)
        degree_com = self._degreeCAL(vx_com, vy_com)

        return degree_std, degree_com
