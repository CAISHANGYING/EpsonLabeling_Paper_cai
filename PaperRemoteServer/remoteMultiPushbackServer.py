import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import timeit
import json
from multiprocessing import Process, Queue
import threading

import cv2
import numpy as np


from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel
from detect.hole_detect_process import HoleDetectConvector


# load config
try:

    with open('./RemoteServer/remote_0531.json') as f:

        config = json.load(f)

except FileNotFoundError:

    with open('./remote_0531.json') as f:

        config = json.load(f)


# stand workpiece initialize
with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:

    workpieceJsonData = dict(json.load(f))

stand_workpiece = StandWorkpiece(workpieceJsonData)


def worker(dataQuene, resultQuene, controlQuene, count):

    push_back = PushBackModel(config['push_back'])

    while True:

        if controlQuene.qsize() > 0:

            controlMessage = controlQuene.get()

            if controlMessage['code'] == 'setDetectWorkpiece':

                stand_workpiece = StandWorkpiece(controlMessage['data'])

                print(f"#{count} Set workpiece success, name: " + stand_workpiece.getName())

                push_back.setStandWorkpiece(stand_workpiece)

            elif controlMessage['code'] == 'setDetectConfig':

                push_back.setConfig(controlMessage['data'])

                print(f"#{count} Set detect config success, comfig: " + str(controlMessage['data']))

        dataJson = dataQuene.get()

        if dataJson is None:

            break

        # run pushback
        result = runPushBack(push_back, dataJson['data'], dataJson['screenSize'], count)

        resultQuene.put(result)


def runPushBack(push_back, hole_convert, screenSize, num):

    start_time = timeit.default_timer()

    print(f'#{num} Workpiece: {push_back.getStandWorkpieceName()}')

    result_of_pushback = push_back.getPushbackPosition(hole_convert, 1.0, 112)

    hole_result_convert = HoleDetectConvector.covertToResultType(result_of_pushback, screenSize, 0)

    end_time = timeit.default_timer()

    print(f'#{num} Result Length: {len(hole_result_convert)}')
    print(f"#{num} Pushback {(end_time - start_time) * 1000} ms")

    return hole_result_convert

class MutiPushbackServer:

    def __init__(self, host, wrenchPort, imagePort, controlPort):


        self.host = host
        self.wrenchPort = wrenchPort
        self.imagePort = imagePort
        self.controlPort = controlPort
        self.serverReady = False
        self.showMode = "Next"
        self.lockingScrewIndex = "1"

        self.isUseGlassesCamera = True

    def connect(self):

        self.wrenchServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.wrenchServer.bind((self.host, self.wrenchPort))
        self.wrenchServer.listen(1)

        print('Wrench Server started on port %s' % self.wrenchPort)
        print('IP: %s' % socket.gethostbyname(self.host))

        self.wrenchClient, address = self.wrenchServer.accept()

        self.wrenchClient.send("Wrench Server Connect\n".encode('utf-8'))

        print('Accepted connection from %s:%s' % (address[0], address[1]))

        self.controlServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controlServer.bind((self.host, self.controlPort))
        self.controlServer.listen(1)

        print()
        print('Control Server started on port %s' % self.controlPort)
        print('IP: %s' % socket.gethostbyname(self.host))

        self.controlClient, address = self.controlServer.accept()
        print('Accepted connection from %s:%s' % (address[0], address[1]))

        self.imageServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.imageServer.bind((self.host, self.imagePort))
        self.imageServer.listen(1)

        print()
        print('Image Server started on port %s' % self.imagePort)
        print('IP: %s' % socket.gethostbyname(self.host))

        self.controlClient.send("Image Server Connect\n".encode('utf-8'))

        self.imageClient, address = self.imageServer.accept()
        print('Accepted connection from %s:%s' % (address[0], address[1]))

    def start(self):

        self.frameCount = 1
        self.processingCount = 4

        self.dataQuene = Queue()
        self.resultQuene = Queue()
        self.controlQuene = Queue()

        self.workers = []

        for i in range(self.processingCount):  # 創建 4 個工作進程

            p = Process(target=worker, args=(self.dataQuene, self.resultQuene, self.controlQuene, i,))
            p.start()

            self.workers.append(p)

        self.wrenchThread = threading.Thread(target=self.handle_wrench_socket)
        self.wrenchThread.start()

        self.controlThread = threading.Thread(target=self.handle_control_socket)
        self.controlThread.start()

        self.imageThread = threading.Thread(target=self.handle_image_socket)
        self.imageThread.start()


    def stop(self):

        print("Stop Server")

        self.imageClient.close()
        self.controlClient.close()
        self.wrenchClient.close()

        for _ in range(self.processingCount):

            self.dataQuene.put(None)

        for p in self.workers:

            p.join()

    def handle_wrench_socket(self):

        while True:

            try:

                data = self.wrenchClient.recv(24)

                if not data:

                    print('Connection closed by the client')

                    self.stop()

                    break

                data = data.decode('utf-8')

                if data == "OK_wrench":

                    print(data)
                    continue

                self.lockingScrewIndex = data

                print("Get lock screw index:", data)

                self.wrenchClient.send("screw OK".encode('utf-8'))

            except ConnectionAbortedError:

                print("Wrench Control Server Closed")

                break

    def handle_control_socket(self):

        while True:

            data = self.controlClient.recv(1024)

            if not data:

                print('Connection closed by the client')

                self.stop()

                break

            data = data.decode('utf-8')
            data = json.loads(data)

            control_code = data['code']

            if control_code == 'close':

                self.stop()

                break

            elif control_code == 'setDetectWorkpiece':

                print("Receive setDetectWorkpiece command, set workpiece")

                length = int(data['data'])

                self.controlClient.send("OK1\n".encode('utf-8'))

                workpieceData = b''

                while len(workpieceData) < length:

                    packet = self.controlClient.recv(min(10240, length - len(workpieceData)))

                    if not packet: break

                    workpieceData += packet

                workpieceJsonData = json.loads(workpieceData.decode('utf-8'))

                controlMessage = {'code': 'setDetectWorkpiece', 'data': workpieceJsonData}

                for _ in range(self.processingCount):

                    self.controlQuene.put(controlMessage)

                print("Set workpiece success, name: " + stand_workpiece.getName())

                self.controlClient.send("OK2\n".encode('utf-8'))

                # wrench control
                controlMessage = {'code': 'setDetectWorkpiece', 'data': {"name": workpieceJsonData['name'], "maxScrewIndex": len(workpieceJsonData['hole_detail'])}}

                self.wrenchClient.send(json.dumps(controlMessage).encode('utf-8'))

            elif control_code == 'setDetectConfig':

                print("Receive setDetectConfig command, set detect config")

                length = int(data['data'])

                self.controlClient.send("OK1\n".encode('utf-8'))

                detectConfigData = b''

                while len(detectConfigData) < length:

                    packet = self.controlClient.recv(min(10240, length - len(detectConfigData)))

                    if not packet: break

                    detectConfigData += packet

                detectConfigJsonData = json.loads(detectConfigData.decode('utf-8'))

                controlMessage = {'code': 'setDetectConfig', 'data': detectConfigJsonData}

                for _ in range(self.processingCount):

                    self.controlQuene.put(controlMessage)

                print("Set workpiece success, name: " + stand_workpiece.getName())

                self.controlClient.send("OK2\n".encode('utf-8'))


            elif control_code == 'checkServerReady':

                print("Receive checkServerReady command, check server ready")
                print("Server Ready: " + str(self.serverReady))

                result = "true" if self.serverReady else "false"

                self.controlClient.send((result + "\n").encode('utf-8'))

            elif control_code == 'setShowMode':

                print("Receive setShowMode command, set show mode")

                self.showMode = data['data']

                print("Show Mode Set to: " + self.showMode)

                self.controlClient.send("OK1\n".encode('utf-8'))

                # wrench control
                controlMessage = {'code': 'setShowMode', 'data': self.showMode}

                self.wrenchClient.send(json.dumps(controlMessage).encode('utf-8'))

            elif control_code == 'setCameraImageSource':

                print("Receive setCameraSourceMode command, set camera source mode")

                self.isUseGlassesCamera = (True if data['data'] == "Glasses" else False)

                print("Camera Source Mode Is Glasses Camera: " + str(self.isUseGlassesCamera))

                self.controlClient.send("OK1\n".encode('utf-8'))

    def handle_image_socket(self):

        while True:

            start_time = timeit.default_timer()

            data = self.imageClient.recv(24)

            if not data:

                break

            lengthData = data.decode('utf-8')

            length = int(lengthData)

            print('ImageSize: ' + str(length) + ' Bytes, length: ' + str(len(data)))

            end_time = timeit.default_timer()

            print(f"Get Length Time {(end_time - start_time) * 1000} ms")

            self.imageClient.send("OK1\n".encode('utf-8'))

            start_time = timeit.default_timer()

            data = b''

            while len(data) < length:

                # 接收影像數據
                packet = self.imageClient.recv(min(10240, length - len(data)))

                if not packet: break

                data += packet

            end_time = timeit.default_timer()

            print("Get Data, Size = " + str(len(data)))
            print(f"Get Data Time {(end_time - start_time) * 1000} ms")


            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if not self.isUseGlassesCamera:

                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow('Camera Window', frame)

            height, width, _ = frame.shape

            # Hole Detect Part

            hole_detail, det, detect_time = hole_detect.detect(frame)

            result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))

            print(f"Detect {detect_time} ms")

            hole_convert = result_of_convert['hole']
            wrench_convert = result_of_convert['wrench']

            wrench_wrong_screw_index_dict = {}

            if self.showMode != "Detect":

                start_time = timeit.default_timer()

                dataJson = {"screenSize": (width, height), "data": hole_convert}

                for _ in range(self.processingCount):  # 將數據放入 Queue 4 次，以便每個工作進程都可以獲取一份

                    self.dataQuene.put(dataJson)

                end_time = timeit.default_timer()

                print(f"Send to Process {(end_time - start_time) * 1000} ms")

                start_time = timeit.default_timer()

                maxCount = 0
                maxFinalPushback = {}

                while True:

                    if self.resultQuene.qsize() == self.processingCount:

                        for _ in range(self.processingCount):

                            item = self.resultQuene.get()

                            if item == {} or item is None: continue

                            if len(item) > maxCount:

                                maxCount = len(item)
                                maxFinalPushback = item

                        break

                end_time = timeit.default_timer()

                print(f"MultiProcessing Get Result {(end_time - start_time) * 1000} ms")

                if self.showMode == "Wrench":

                    wrench_wrong_screw_index_dict = HoleDetectConvector.wrenchOnWrongScrewPosition(wrench_convert, maxFinalPushback, self.lockingScrewIndex)

                finalResult = HoleDetectConvector.covertToResultTypeForServer(maxFinalPushback, (width, height), 0)

                push_back_frame = HoleDetectConvector.getPushbackFrame(frame, maxFinalPushback, )

            else :

                print(f"Show Detect Hole Mode, Skip Pushback")

                finalResult = HoleDetectConvector.covertToResultTypeForServer(hole_convert, (width, height), 0)

                push_back_frame = HoleDetectConvector.getPushbackFrame(frame, hole_convert, )

            wrench_convert = HoleDetectConvector.covertToResultTypeForServer(wrench_convert, (width, height), 0)

            finalResult = {
                'pushback': finalResult,
                'wrench': wrench_convert,
                'lockingScrewIndex': self.lockingScrewIndex,
                'wrenchWrongScrewIndexDict': wrench_wrong_screw_index_dict,
            }

            start_time = timeit.default_timer()

            finalResult = json.dumps(finalResult).encode('utf-8')

            self.imageClient.send((str(len(finalResult)) + "\n").encode('utf-8'))

            self.imageClient.recv(24) # "OK2\n"

            self.imageClient.send(finalResult)

            self.imageClient.recv(24) # "OK3\n"


            cv2.imshow('Result Window', push_back_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

            self.imageClient.send("OK4\n".encode('utf-8'))

            end_time = timeit.default_timer()

            print(f"Send Result {(end_time - start_time) * 1000} ms")

            print()
            print("Frame: " + str(self.frameCount))
            print("-" * 50)
            print()
            print()

            self.frameCount += 1

    def setServerReady(self):

        self.serverReady = True

        print()
        print("Server Ready !!!")
        print()

        self.wrenchClient.send("Server Ready".encode('utf-8'))



if __name__ == '__main__':

    host = '0.0.0.0'
    imagePort = 10254
    controlPort = 10255
    wrenchPort = 10256

    server = MutiPushbackServer(host, wrenchPort, imagePort, controlPort)

    server.connect()
    server.start()

    from detect.hole.hole_detect_paper import HoleDetectModel

    hole_detect = HoleDetectModel(config['yolov5'])

    server.setServerReady()