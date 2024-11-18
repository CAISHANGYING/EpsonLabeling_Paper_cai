import json
import socket
import threading
from time import sleep


class WrenchControlSimulater:

    def __init__(self, host, port):

        self.host = host
        self.port = port

        self.screwIndexSending = False
        self.screwIndex = "1"

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.showMode = "Next"

        self.workpieceName = "None"
        self.workpieceMaxScrewIndex = "0"

        while True:

            try:

                self.socket.connect((self.host, self.port))

                break

            except ConnectionRefusedError:

                print("Waiting for Wrench Control Server...")

                sleep(2)

        self.connectSuccessfulMessage = self.socket.recv(1024)

        print(self.connectSuccessfulMessage.decode("utf-8")) # "Connected to Wrench Control Server"

        self.serverReadyMessage = self.socket.recv(1024)

        print(self.serverReadyMessage.decode("utf-8")) # "Server Ready"

        self.wrenchThread = threading.Thread(target=self.handle_wrench_socket)
        self.wrenchThread.start()

        self.screwIndexSettingThread = threading.Thread(target=self.handle_screw_index_setting)
        self.screwIndexSettingThread.start()

    def handle_wrench_socket(self):

        while True:

            try:

                data = self.socket.recv(1024)

                if not data:

                    break

                data = data.decode("utf-8")

                if data == "screw OK":

                    self.screwIndexSending = False

                    print(data) # "screw OK"

                else:

                    data = json.loads(data)

                    control_code = data['code']

                    if control_code == 'close':

                        self.stop()

                        break

                    elif control_code == 'setDetectWorkpiece':

                        print("Receive setDetectWorkpiece command, set workpiece")

                        self.workpieceName = data['data']['name']
                        self.workpieceMaxScrewIndex = data['data']['maxScrewIndex']

                        self.socket.send("OK_wrench".encode('utf-8'))

                        self.screwIndex = "1"

                    elif control_code == 'setShowMode':

                        print("Receive setShowMode command, set show mode")

                        self.showMode = data['data']

                        print("Show Mode Set to: " + self.showMode)

                        self.socket.send("OK_wrench".encode('utf-8'))

            except ConnectionResetError:

                print("Wrench Control Server Disconnected")

                break

    def handle_screw_index_setting(self):

        while True:

            if self.showMode != "Wrench" or self.screwIndexSending : continue

            print('-' * 50)
            print("Now Workpiece: " + self.workpieceName + ", Max Screw Index: " + str(self.workpieceMaxScrewIndex))
            print()

            inputScrew = input(f"Enter Screw Index (Now #{self.screwIndex}): ")

            if self.showMode != "Wrench":

                print("Skip")

                continue

            if inputScrew != "done":

                try:

                    if 1 > int(inputScrew) or int(inputScrew) > int(self.workpieceMaxScrewIndex):

                        print("Input Screw Index Error")

                        continue

                except ValueError:

                    print("Input Screw Index Error")

                    continue

            self.screwIndex = inputScrew

            self.socket.send(self.screwIndex.encode("utf-8"))

            self.screwIndexSending = True

    def stop(self):

        print("Stop Server")

        self.socket.close()


if __name__ == '__main__':

    host = 'localhost'
    wrenchPort = 10256

    wrenchControlSimulater = WrenchControlSimulater(host, wrenchPort)