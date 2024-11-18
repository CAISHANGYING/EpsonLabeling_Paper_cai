import socket
import timeit
import json
from multiprocessing import Process, Queue

from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel
from detect.hole_detect_process import HoleDetectConvector

# log config

with open('../environment_test/environment_test.json') as f:

    config = json.load(f)

# stand workpiece initialize

with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:

    workpieceJsonData = dict(json.load(f))

stand_workpiece = StandWorkpiece(workpieceJsonData)


def runPushBack(hole_convert, screenSize, num):

    start_time = timeit.default_timer()

    push_back = PushBackModel(config['push_back'], stand_workpiece)

    result_of_pushback = push_back.getPushbackPosition(hole_convert, 1.0)

    hole_result_convert = HoleDetectConvector.covertToResultType(result_of_pushback, screenSize, 0)

    end_time = timeit.default_timer()

    print(f'#{num} Result Length: {len(hole_result_convert)}')
    print(f"#{num} Pushback {(end_time - start_time) * 1000} ms")


def worker(q, count):

    while True:

        dataJson = q.get()

        if dataJson is None:

            break

        # 在這裡處理數據...
        runPushBack(dataJson['data'], dataJson['screenSize'], count)

if __name__ == '__main__':

    # Create Connection : Server

    host = 'localhost'  # 伺服器地址
    port = 13478  # 伺服器端口

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)

    print('Server started on port %s' % port)
    print('Server IP: %s' % socket.gethostbyname(host))

    client, address = server.accept()
    print('Accepted connection from %s:%s' % (address[0], address[1]))



    # MultiProcessing Function

    q = Queue()
    workers = []

    for i in range(4):  # 創建 4 個工作進程

        p = Process(target=worker, args=(q, i))
        p.start()

        workers.append(p)


    frameCount = 1

    while True:

        start_time = timeit.default_timer()

        data = client.recv(10240)

        dataStr = data.decode('utf-8')

        dataJson = json.loads(dataStr)

        end_time = timeit.default_timer()

        print(f"Get Data {(end_time - start_time) * 1000} ms")

        print()
        print("Frame: " + str(frameCount))
        print("-" * 50)
        print()

        for _ in range(4):  # 將數據放入 Queue 4 次，以便每個工作進程都可以獲取一份

            q.put(dataJson)

        print()
        frameCount += 1

    for _ in range(4):  # 將 None 放入 Queue 4 次，以便每個工作進程都可以接收到

        q.put(None)

    for p in workers:

        p.join()