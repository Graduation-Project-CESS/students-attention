# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:25:28 2021

@author: zeids
"""
import socket
import time
import pickle
import SAA
import threading
import csv


def receive():
    full_msg = b''
    new_msg = True    
    while True:
        msg = clientsocket.recv(16)
        if len(msg) == 0:
            return None 
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        full_msg += msg

        if len(full_msg)-HEADERSIZE == msglen:
            return pickle.loads(full_msg[HEADERSIZE:])

def send(data):
    msg = pickle.dumps(data)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    clientsocket.send(msg)

def send_report(number_of_reports):
    report = []
    with open('./reports/report_{}.csv'.format(number_of_reports)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            report.append(row)
        send(report)
        
def start_analysis(current_image, number_of_reports):
    while True:
        current_image = SAA.run(current_image, number_of_reports)
        send_report(number_of_reports)
        number_of_reports +=1    

def check_stop():
    while True:
        message = receive()
        if message == 'stop':
            print('stop received')
            return
 
def send_attendance_sheet(number_of_reports):
    print('in att sheet')
    SAA.generate_attendance_sheet(number_of_reports)
    attendance_sheet = []
    with open('./reports/attendance.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            attendance_sheet.append(row)
        send(attendance_sheet)          

HEADERSIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1243))
s.listen(5)
current_image = 0
global number_of_reports
number_of_reports = 0 
print("Server started at port: 1243!")

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")

    while True:
        message = receive()
        if message is not None:
            print(message)
            if message == 'start':
                try:
                    tid = threading.Thread(target = start_analysis, args = (current_image, number_of_reports, ) )
                    tid.start()
                    check_stop()
                    tid.join()
                    send_attendance_sheet(number_of_reports)
                except Exception as e:
                    print (e)

            
        
    # d = {1:"hi", 2: "there"}
    # msg = pickle.dumps(d)
    # msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
    # print(msg)
    # clientsocket.send(msg)