# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:50:53 2021

@author: zeids
"""
import socket
import pickle

def send(msg):
    msg = pickle.dumps(msg)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    s.send(msg)

def receive():
    full_msg = b''
    new_msg = True    
    while True:
        msg = s.recv(16)
        if len(msg) == 0:
            return None 
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        full_msg += msg

        if len(full_msg)-HEADERSIZE == msglen:
            return pickle.loads(full_msg[HEADERSIZE:])


HEADERSIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1243))

while True:
    
    msg = 'start'
    send(msg)
    print('Message: {} sent to server!'.format(msg))
    report = receive()
    while len(report) == 0:
        report = receive()
    print(report)
    msg = 'stop'
    send(msg)
    print('Message: {} sent to server!'.format(msg))
    sheet = receive()
    while len(sheet) == 0:
        sheet = receive()
    print(sheet)      
    break
    # full_msg = b''
    # new_msg = True
    # while True:
    #     msg = s.recv(16)
    #     if new_msg:
    #         print("new msg len:",msg[:HEADERSIZE])
    #         msglen = int(msg[:HEADERSIZE])
    #         new_msg = False

    #     print(f"full message length: {msglen}")

    #     full_msg += msg

    #     print(len(full_msg))

    #     if len(full_msg)-HEADERSIZE == msglen:
    #         print("full msg recvd")
    #         print(full_msg[HEADERSIZE:])
    #         print(pickle.loads(full_msg[HEADERSIZE:]))
    #         new_msg = True
    #         full_msg = b""