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

            action, value, updated_state = sess.run(
                [network.predictedAction, network.predictedValue, network.rnn_state],
                feed_dict={network.scalarInput: processed_input, network.trainLength: 1,
                           network.state_in: message['lstm_state'], network.batch_size: message['obs'].shape[0]})

            self.sendBinaryMessage(self.buildBinaryMessage(updated_state, action, value))
