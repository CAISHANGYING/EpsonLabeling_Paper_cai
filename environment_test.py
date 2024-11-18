import os
import sys

sys.path.append(os.getcwd())

import json


with open('./environment_test/environment_test.json') as f:

    config = json.load(f)

from detect.hole.hole_detect_old import HoleDetectModel
from tool.stand_tool import StandTool
from detect.push_back.hole_push_back_old_v1 import HolePushBackModel

hole_detect = HoleDetectModel(config['yolov5'])
stand_tool = StandTool(config['stand'])
push_back = HolePushBackModel(config['push_back'], stand_tool)

import cv2

frame = cv2.imread('./environment_test/S__128892932.jpg')
# frame = cv2.resize(frame, (1440, 810))
# frame = cv2.resize(frame, (810, 1440))


hole_detail, det = hole_detect.detect(frame)

cv2.imshow('detect', det)

while True:

    push_back_list, min_hit = push_back.get_push_back_position((frame.shape[0], frame.shape[1]), hole_detail)

    push_back_frame = push_back.get_hole_frame(frame, push_back_list, min_hit)

    cv2.imshow('test result', push_back_frame)

    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1000) == ord('q'):

        break

