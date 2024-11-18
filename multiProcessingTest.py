from multiprocessing import Process, Queue, Pool

import os
import sys
import json
import timeit


from workpiece.stand_workpiece import StandWorkpiece
from detect.hole_detect_process import HoleDetectConvector
from detect.push_back.hole_push_back import PushBackModel


sys.path.append(os.getcwd())

# log config

with open('./environment_test/environment_test.json') as f:

    config = json.load(f)


# stand workpiece initialize

with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:

    workpieceJsonData = dict(json.load(f))

stand_workpiece = StandWorkpiece(workpieceJsonData)

push_back = PushBackModel(config['push_back'], stand_workpiece)


# multiprocessing initialize

def runPushBack(hole_convert, screenSize, singleTryTime, num):

    start_time = timeit.default_timer()

    result_of_pushback = push_back.getPushbackPosition(hole_convert, 1.0, singleTryTime)

    hole_result_convert = HoleDetectConvector.covertToResultType(result_of_pushback, screenSize, 0)

    end_time = timeit.default_timer()

    print(f'#{num} Result Length: {len(hole_result_convert)}')
    print(f"#{num} Pushback {(end_time - start_time) * 1000} ms")

    return hole_result_convert

def worker(q, q_out, tryTime, processId):

    while True:

        dataJson = q.get()

        if dataJson is None:

            break

        # 在這裡處理數據...
        result = runPushBack(dataJson['data'], dataJson['screenSize'], tryTime, processId)

        q_out.put(result)

if __name__ == '__main__':

    totalTryTime = config['push_back']['try_time'] * 100

    # MultiProcessing Function

    processingCount = 8

    singleTryTime = totalTryTime // processingCount

    q = Queue()
    q_out = Queue()

    workers = []

    for i in range(processingCount):  # 創建 4 個工作進程

        p = Process(target=worker, args=(q, q_out, singleTryTime, i))
        p.start()

        workers.append(p)

    # hole detect model initialize

    from detect.hole.hole_detect import HoleDetectModel

    hole_detect = HoleDetectModel(config['yolov5'])

    # get video frame

    import cv2

    # frame = cv2.imread('./environment_test/771027.jpg')
    cap = cv2.VideoCapture('./environment_test/test.mp4')
    # frame = cv2.resize(frame, (1440, 810))
    # frame = cv2.resize(frame, (810, 1440))

    while True:

        print()

        ret, frame = cap.read()

        if ret == False:

            break


        height, width, _ = frame.shape

        hole_detail, det = hole_detect.detect(frame)

        cv2.imshow('detect', det)

        # Detect Part
        start_time = timeit.default_timer()

        result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))

        end_time = timeit.default_timer()

        print(f"Detect {(end_time - start_time) * 1000} ms")

        hole_convert = result_of_convert['hole']
        wrench_convert = result_of_convert['wrench']

        start_time = timeit.default_timer()

        start_time2 = timeit.default_timer()

        for _ in range(processingCount):

            q.put({"screenSize": (width, height), "data": hole_convert})

        end_time2 = timeit.default_timer()

        print(f"MultiProcessing Send Data {(end_time2 - start_time2) * 1000} ms")

        # get result

        start_time2 = timeit.default_timer()

        maxCount = 0
        maxFinalPushback = {}

        while True:

            if q_out.qsize() == processingCount:

                for _ in range(processingCount):

                    item = q_out.get()

                    if item == {} or item is None: continue

                    if len(item) > maxCount:

                        maxCount = len(item)
                        maxFinalPushback = item

                break

        end_time2 = timeit.default_timer()

        print(f"MultiProcessing Get Result {(end_time2 - start_time2) * 1000} ms")


        end_time = timeit.default_timer()

        print(f"MultiProcessing Pushback {(end_time - start_time) * 1000} ms")


        push_back_frame = HoleDetectConvector.getPushbackFrame(frame, maxFinalPushback,)

        cv2.imshow('test result', push_back_frame)


        if cv2.waitKey(1) == ord('q'):

            break

