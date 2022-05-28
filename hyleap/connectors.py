"""
Author: Dikshant Gupta
Time: 13.02.22 22:23
"""
import sys
import socket
import numpy as np
import time
import os
import threading
import struct

import skimage.draw

from hyleap.utils import *
from hyleap.model import ExperienceBuffer


new_im = True


class train_connector(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.total = ""
        self.state = None
        self.lastAction = -1
        self.conn = None
        self.addr = None

        self.initialized = False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.sock.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()

        print('Training connector: Bound to port ' + str(PORT) + '...')

        self.sock.listen()
        print('Training connector: Socket now listening...')

        self.conn, self.addr = self.sock.accept()
        print('Training connector: Connection Accepted...')

        self.initialized = True

    def parse(self, tmp):
        if not tmp:
            return None

        tmp = tmp.split(";")

        assert len(tmp) == 4

        arr = None
        try:
            # vector<float> real_values;
            real_values = np.array([float(x) for x in tmp[0].split(",")])

            N = len(real_values) - 1
            for i in range(1, N):
                real_values[N - i] += y * real_values[N - i + 1]

            real_values[0] += y * real_values[1]

            # vector<float*> despot_policy;
            despot_policy = np.array([float(x) for x in tmp[1].split(",")])
            despot_policy.shape = (-1, num_actions)
            # vector<float*> observations;
            observations = np.array([float(x) for x in tmp[2].split(",")])
            observations.shape = (-1, observation_size)
            # vector<float*> histories;
            histories = np.array([float(x) for x in tmp[3].split(",")])
            histories.shape = (-1, history_size)
            histories = np.split(histories, 2, 1)

            arr = {'despot_policy': despot_policy, 'real_values': real_values,
                   'observations': observations, 'histories': histories}
        except (IndexError, ValueError) as e:
            return None

        return arr

    def receiveMessage(self):
        convertedBytes = ""
        while not (convertedBytes and convertedBytes[-1] == '\n'):
            try:
                receivedBytes = self.conn.recv(1024)

                if receivedBytes == 0:
                    time.sleep(0.005)
                    continue

                convertedBytes = receivedBytes.decode('utf-8')
                self.total += convertedBytes
            except OSError as e:
                print(e)
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

    def run(self):
        total_episodes = 0
        buf = ExperienceBuffer()

        while True:
            if not self.initialized:
                time.sleep(5)
                continue

            self.receiveMessage()
            print('Finished episode ' + str(total_episodes) + ' after '
                  + str(self.state['observations'].shape[0]) + ' steps, reward ' + str(self.state['real_values'][0]))

            message_tmp = getObsParallel(costMap, self.state['observations'])

            self.state['observations'] = message_tmp

            buf.add(self.state)

            if enableTraining:
                print(self.state)

            total_episodes += 1


class ConnectorServer(threading.Thread):
    def __init__(self, number):
        threading.Thread.__init__(self)
        self.path = "/tmp/python_unix_sockets_example" + str(number)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        if os.path.exists(self.path):
            os.remove(self.path)

        try:
            self.sock.bind(self.path)
        except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()

        print('Connector: Bound to port...')

        self.sock.listen()

        print('Connector: Socket now listening...')

    def run(self):
        while True:
            conn, addr = self.sock.accept()
            print('Connector: Connection Accepted...')
            ConnectorFull(conn).start()


class image_connector(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.total = ""
        self.path = "/tmp/python_unix_sockets_image"
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        if os.path.exists(self.path):
            os.remove(self.path)

        try:
            self.sock.bind(self.path)
        except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()

        print('ImageSocket: Bound to port...')

        self.sock.listen()

        print('ImageSocket: Socket now listening...')

        self.conn, self.addr = self.sock.accept()
        print('Training connector: Connection Accepted...')

        self.initialized = True

    def parse(self, tmp):
        if not tmp:
            return None

        tmp = tmp.split(";")

        assert len(tmp) == 3

        try:
            goal_position = [float(x) for x in tmp[0].split(",")]
            obstacle = [float(x) for x in tmp[1].split(",")]
            waypoints = np.array([float(x) for x in tmp[2].split(",")])
            waypoints.shape = (-1, 2)

            arr = {'goal_position': goal_position, 'waypoints': waypoints, 'obstacle': obstacle}
        except (IndexError, ValueError) as e:
            return None

        return arr

    def receiveMessage(self):
        convertedBytes = ""
        while not (convertedBytes and convertedBytes[-1] == '\n'):
            try:
                receivedBytes = self.conn.recv(1024)

                if receivedBytes == 0:
                    time.sleep(0.005)
                    continue

                convertedBytes = receivedBytes.decode('utf-8')
                self.total += convertedBytes
            except OSError as e:
                print(e)
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

    def run(self):
        while True:
            if not self.initialized:
                time.sleep(5)
                continue

            self.receiveMessage()

            global new_im
            global costMap
            costMapTmp = np.copy(costMap)

            self.state['waypoints'] = self.state['waypoints'].round().astype(int)

            for i in range(1, self.state['waypoints'].shape[0]):
                rowOld = self.state['waypoints'][i - 1, :]
                row = self.state['waypoints'][i, :]
                rr, cc = skimage.draw.line(rowOld[1], rowOld[0], row[1], row[0])
                for xx, xy in zip(cc, rr):
                    costMapTmp[xx + 10, xy + 10] = 0.0

            rr, cc = skimage.draw.ellipse(self.state['goal_position'][1],
                                          self.state['goal_position'][0], 5, 5, shape=costMapTmp.shape)
            for xx, xy in zip(cc, rr):
                costMapTmp[xx + 10, xy + 10] = 0.0

            rr, cc = skimage.draw.ellipse(self.state['obstacle'][1] * multiplyer,
                                          self.state['obstacle'][0] * multiplyer, 5, 5, shape=costMapTmp.shape)
            for xx, xy in zip(cc, rr):
                costMapTmp[xx + 10, xy + 10] = 200.0

            costMap = costMapTmp
            new_im = True


class ConnectorFull(threading.Thread):

    def __init__(self, conn):
        threading.Thread.__init__(self)

        self.total = b""
        self.state = None

        self.conn = conn

    def simpleParse(self, data, num_states):
        try:
            lstm_state1 = np.array(data[0:history_size // 2])
            lstm_state1 = np.tile(lstm_state1, num_states)
            lstm_state1.shape = (num_states, history_size // 2)

            lstm_state2 = np.array(data[history_size // 2: history_size])
            lstm_state2 = np.tile(lstm_state2, num_states)
            lstm_state2.shape = (num_states, history_size // 2)

            # (np.zeros([1, h_size + var_end_size]), np.zeros([1, h_size + var_end_size]))

            obs = np.array(data[history_size:])
            obs.shape = (-1, 12)

            arr = {'terminal': True, 'lstm_state': (lstm_state1, lstm_state2), 'obs': obs}

            return arr
        except (IndexError, ValueError) as e:
            return None

    def receiveMessage(self):
        while True:
            receivedBytes = self.conn.recv(4)
            if len(receivedBytes) == 4:
                break

        assert len(receivedBytes) == 4

        num_states = int(round(struct.unpack('f', receivedBytes)[0]))

        totalLen = (history_size + observation_size * num_states) * 4

        while len(self.total) < totalLen:
            try:
                receivedBytes = self.conn.recv(totalLen - len(self.total))

                if receivedBytes == 0:
                    continue

                self.total += receivedBytes
            except OSError as e:
                print(e)
                time.sleep(5)

        assert len(self.total) == totalLen
        num_elements = totalLen // 4
        format_string = str(num_elements) + 'f'
        data = struct.unpack(format_string, self.total)

        self.state = self.simpleParse(data, num_states)
        self.total = b""

        return self.state

    def buildBinaryMessage(self, state, action, value):
        res = b""

        for i in range(len(action)):
            res += state[0][i, :].astype('f').tostring()
            res += state[1][i, :].astype('f').tostring()
            res += struct.pack('2f', float(action[i]), value[i])

        return res

    def sendBinaryMessage(self, m):
        while True:
            try:
                self.conn.sendall(m)
                break
            except socket.error as e:
                print(e)
                time.sleep(5)

    def one_hot(self, i):
        if i == -1:
            i = 1

        return [1 if i == index else 0 for index in range(num_actions)]

    def run(self):
        while True:
            message = self.receiveMessage()
            processed_input = getObsParallel(costMap, message['obs'])

            # print(message)
            #  arr = { 'terminal': True, 'lstm_state': (lstm_state1, lstm_state2), 'obs': data[history_size:]}

            # action, value, updated_state = sess.run(
            #     [network.predictedAction, network.predictedValue, network.rnn_state],
            #     feed_dict={network.scalarInput: processed_input, network.trainLength: 1,
            #                network.state_in: message['lstm_state'], network.batch_size: message['obs'].shape[0]})
            action = [0]
            value = [0]
            updated_state = []

            self.sendBinaryMessage(self.buildBinaryMessage(updated_state, action, value))
