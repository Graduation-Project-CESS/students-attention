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
import multiprocessing
import cv2

def receive():
    full_msg = b''
    start_msg = True    
    while True:
        msg = clientsocket.recv(16)
        if len(msg) == 0:
            return None 
        if start_msg:
            msglen = int(msg[:HEADERSIZE])
            start_msg = False

        full_msg += msg

        if len(full_msg)-HEADERSIZE == msglen:
            return pickle.loads(full_msg[HEADERSIZE:])

def send(data,clientsocket):
    msg = pickle.dumps(data)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    clientsocket.send(msg)

def send_report(number_of_reports, clientsocket):
    report = []
    print(number_of_reports)
    report.append(number_of_reports)
    with open('./reports/Report{}.csv'.format(number_of_reports)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            report.append(row)
    send(report, clientsocket)


def send_images(initial_image, current_image, clientsocket):
    print('Sending image')
    images = []
    print(initial_image, '-------' , current_image)
    for i in range(initial_image, current_image):
        path = './final_output/face_image_{}.jpg'.format(i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        images.append(img)
    print(len(images))
    send(images, clientsocket)
    print('Sent images')
        
        
def start_analysis(current_image, number_of_reports, clientsocket):
    while True:
        initial_image = current_image
        current_image = SAA.run(current_image, number_of_reports[0])
        send_report(number_of_reports[0], clientsocket)
        number_of_reports[0] +=1
        send_images(initial_image, current_image, clientsocket)        

def check_stop():
    while True:
        message = receive()
        if message == 'stop':
            print('stop received')
            return
 
def send_attendance_sheet(number_of_reports, clientsocket):
    print('in att sheet')
    SAA.generate_attendance_sheet(number_of_reports)
    attendance_sheet = []
    with open('./reports/attendance.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            attendance_sheet.append(row)
        send(attendance_sheet,clientsocket)          


HEADERSIZE = 10
port = 5000
if __name__ == '__main__':
    
    
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), port))
    s.listen(5)
    current_image = 0
    number_of_reports = multiprocessing.Array("i", [0])
    # global number_of_reports
    # number_of_reports = 0 
    print("Server started at port: {} !".format(port))
    
    while True:
        # now our endpoint knows about the OTHER endpoint.
        global clientsocket
        clientsocket, address = s.accept()
        print(f"Connection from {address} has been established.")
    
        while True:
            message = receive()
            if message is not None:
                print(message)
                if message == 'start':
                    try:
                        tid = multiprocessing.Process(target=start_analysis, args=(current_image,number_of_reports,clientsocket,  ))
                        tid.start()
                        check_stop()
                        tid.terminate()
                        send_attendance_sheet(number_of_reports[0], clientsocket)
                        break
                    except Exception as e:
                        print (e)
                

            
        
    # d = {1:"hi", 2: "there"}
    # msg = pickle.dumps(d)
    # msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
    # print(msg)
    # clientsocket.send(msg)