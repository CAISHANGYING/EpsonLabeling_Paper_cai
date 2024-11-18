
import os
import sys
import time
import timeit

sys.path.append(os.getcwd())

import json
import socket

from detect.hole.hole_detect import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.hole_detect_process import HoleDetectConvector


# Create Connection : client

host = 'localhost'  # 伺服器地址
port = 13478  # 伺服器端口

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

print('Client connected to server on port %s' % port)

# log config

with open('../environment_test/environment_test.json') as f:

    config = json.load(f)

# # stand workpiece initialize
#
# with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:
#
#     workpieceJsonData = dict(json.load(f))
#
# stand_workpiece = StandWorkpiece(workpieceJsonData)

# model initialize

hole_detect = HoleDetectModel(config['yolov5'])


import cv2

# frame = cv2.imread('./environment_test/771027.jpg')
cap = cv2.VideoCapture('D:\\Program\\Python\\EpsonLabeling\\environment_test\\test.mp4')
# frame = cv2.resize(frame, (1440, 810))
# frame = cv2.resize(frame, (810, 1440))

while True:

    ret, frame = cap.read()

    if ret == False:

        break


    height, width, _ = frame.shape

    # Detect Part
    hole_detail, det = hole_detect.detect(frame)

    cv2.imshow('detect', det)


    start_time = timeit.default_timer()

    result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))

    end_time = timeit.default_timer()

    print(f"Detect {(end_time - start_time) * 1000} ms")

    # Pushback Part
    hole_convert = result_of_convert['hole']
    wrench_convert = result_of_convert['wrench']

    # Send Hole Data
    start_time = timeit.default_timer()

    holeDataStr = json.dumps({"screenSize": (width, height), "data": hole_convert})

    client.send(holeDataStr.encode('utf-8'))

    end_time = timeit.default_timer()

    print(f"Send Hole Data {(end_time - start_time) * 1000} ms")


    # count = 0
    #
    # with Pool(4) as pool:
    #
    #     pool.apply_async(
    #         runPushBack,
    #         args=(hole_convert, count)
    #     )
    #
    #     pool.close()
    #     pool.join()
    #
    #     count += 1

    if cv2.waitKey(1) == ord('q'):

        break

