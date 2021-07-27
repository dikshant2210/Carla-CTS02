"""
Author: Dikshant Gupta
Time: 27.07.21 21:43
"""

import socket


class Connector:
    def __init__(self, port):
        self.port = port
        self.connection = None

    def establish_connection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', self.port)
        sock.bind(server_address)
        sock.listen(1)
        self.connection, client_address = sock.accept()

    def receive_message(self):
        message = ""
        while True:
            m = self.connection.recv(1024)
            message += m
            if m[-1] == "\n":
                break
        return message

    def send_message(self, terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path):
        message = ""
        message += str(terminal) + ";" + str(reward) + ";" + str(angle) + ";"
        message += str(car_pos[0]) + ";" + str(car_pos[1]) + ";" + str(car_speed) + ";"
        for pos in pedestrian_positions:
            message += str(pos[0]) + ";" + str(pos[1]) + ";"  # Pedestrian position: (x, y)
        for wp in path:
            message += str(wp[0]) + "," + str(wp[1]) + "," + str(wp[2]) + ","  # Waypoint: (x, y, theta)
        message += "\n"
        self.connection.sendall(message)
