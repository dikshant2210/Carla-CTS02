"""
Author: Dikshant Gupta
Time: 27.05.22 23:52
"""

import os

from hyleap.model import HyLEAPNetwork
from hyleap.connectors import ConnectorServer
from hyleap.connectors import train_connector
from hyleap.connectors import image_connector


server = ConnectorServer(0)
train_connection = train_connector()
image_connection = image_connector()

network = HyLEAPNetwork()

path = '_out/hyleap'
if not os.path.exists(path):
    os.makedirs(path)

server.start()
train_connection.start()
image_connection.start()

server.join()
