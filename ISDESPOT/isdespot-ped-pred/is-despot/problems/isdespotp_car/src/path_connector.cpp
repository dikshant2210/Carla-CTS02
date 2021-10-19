//
// Created by dikshant on 23.02.21.
//

#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include "path_connector.h"

using namespace std;

void PathConnector::establish_connection() {
    socket_desc = socket(AF_INET , SOCK_STREAM , 0);

    if (socket_desc == -1)
    {
        clog << "Could not create socket\n";
    }

    server.sin_addr.s_addr = inet_addr("127.0.0.1");
    server.sin_family = AF_INET;
    server.sin_port = htons(10000);

    if (connect(socket_desc, (struct sockaddr *)&server, sizeof(server)) < 0) {
        clog << "Cannot connect to the server!\n";
        close(socket_desc);
    }
}

vector<PedHistory> PathConnector::getPredictedPath(vector <PedHistory> &ped_history) {
    string s;
    for(int i = 0; i < ped_history.size(); i++) {
        PedHistory h = ped_history[i];
        for(int j = 0; j < h.x.size(); j++) {
            s.append(to_string(h.x[j]));
            s.append(" ");
            s.append(to_string(h.y[j]));
            s.append(" ");
        }
        break;
    }

    const char *message = s.c_str();
    if (send(socket_desc, message, strlen(message), 0) < 0) {
        clog << "Send failed\n";
        close(socket_desc);
    }

    char server_reply[2048];
    if(recv(socket_desc, server_reply, 200, 0) < 0) {
        clog << "Received failed\n";
        close(socket_desc);
    }
    
    vector<PedHistory> temp;
    return  temp;
}

void PathConnector::closeConnection() {
    close(socket_desc);
}