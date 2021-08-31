import socket
import time
from SA3C.sac.config import Config


class Connector:
    def __init__(self, port=4001):
        self.total = ""
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", self.port))
        self.state = None
        self.lastAction = -1

    def reset(self):
        self.sendMessage("RESET\n")

        while self.state is None or self.state['terminal']:
            self.sendMessage("RESET\n")
            self.receiveMessage()

    def parse(self, tmp):
        if not tmp:
            return None

        tmp = tmp.split(";")

        arr = None
        try:
            arr = {'terminal': True if tmp[0] == 'true' else False, 'reward': float(tmp[1]), 'angle': float(tmp[2]),
                   'obs': [float(x) for x in tmp[3].split(",")], 'map': [float(x) for x in tmp[4].split(",")]}
        except (IndexError, ValueError) as e:
            return None

        if len(arr['obs']) == 4 + Config.num_pedestrians * 2 and len(arr['map']) == 100 * 100 * 3:
            return arr

        return None

    def receiveMessage(self):
        convertedBytes = ""
        while not (convertedBytes and convertedBytes[-1] == '\n'):
            try:
                receivedBytes = self.sock.recv(1024)

                if receivedBytes == 0:
                    time.sleep(0.005)
                    continue

                convertedBytes = receivedBytes.decode('utf-8')
                self.total += convertedBytes
            except OSError as e:
                print(e)
                print("Try to restore connection")
                while True:
                    try:
                        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.sock.connect(("localhost", self.port))
                        break
                    except OSError as p:
                        print("Timeout during connect ")
                        time.sleep(5)

        tmp_split = self.total.split("\n")

        parsed = self.parse(tmp_split[-1])
        if parsed is None:
            self.total = tmp_split[-1]
            self.state = self.parse(tmp_split[-2])
            assert (self.state is not None)
        else:
            self.total = ""
            self.state = parsed

        return self.state

    def sendMessage(self, m):
        while True:
            try:
                self.sock.send(m.encode())
                break
            except socket.error as e:
                print(e)
                print("Try to restore connection")
                while True:
                    try:
                        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.sock.connect(("localhost", self.port))
                        break
                    except OSError:
                        print("Timeout during connect ")
                        time.sleep(5)

    # reward, is_running = self.connection.step(action)
    def step(self, action):

        # Replace noop by our noop
        if action == -1:
            action = 1

        self.sendMessage(str(action) + "\n")
        self.receiveMessage()
        self.lastAction = action

        return self.state['reward'], not self.state['terminal']

    def one_hot(self, i):
        if i == -1:
            i = 1

        return [1 if i == index else 0 for index in range(Config.num_actions)]
