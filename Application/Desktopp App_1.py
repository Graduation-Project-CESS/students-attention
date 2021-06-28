import sys
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QMessageBox
import socket
import pickle
import csv
import multiprocessing
import ctypes
import glob
import os
import cv2
from datetime import date

######################################################

######################################################

HEADERSIZE = 10



def receive_images(s, number_of_reports):
    directory_path = './images/Report {}'.format(number_of_reports)
    
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    
    images = receive_image_bytes(s)
    while len(images) == 0:
        images = receive_image_bytes(s)
    
    print('len images', len(images))
    for itr, image in enumerate(images):   
        cv2.imwrite('./images/Report {}/face_image_{}.jpg'.format(number_of_reports, itr),image)

def receive_image_bytes(s):
    full_msg = b''
    new_msg = True    
    while True:
        msg = s.recv(102400)  #Receiving 100 KB
        if len(msg) == 0:
            return None 
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        full_msg += msg

        if len(full_msg)-HEADERSIZE == msglen:
            return pickle.loads(full_msg[HEADERSIZE:])

def receiving_reports(s):
    print("thread started")
    while True:
        report = receive(s)
        while len(report) == 0:
            report = receive(s)
        print("report received")
        number_of_reports = report[0]
        number_of_reports +=1 #for naming purposes
        print(number_of_reports)
        report = report[1:]
        current_report_path = './Reports/Report{}.csv'.format(number_of_reports)
        with open(current_report_path, mode='w', newline='') as file:
            file_write = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for record in report:        
                file_write.writerow(record)  
        receive_images(s, number_of_reports)        

def send(msg):
    msg = pickle.dumps(msg)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    s.send(msg)

def receive(s):
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




class courses_window(QMainWindow):
    def __init__(self):
        super(courses_window,self).__init__()
        uic.loadUi("courses.ui",self)
        self.start_btn.setEnabled(False)
        self.comboBox.currentIndexChanged.connect(self.enableStart_btn)
        self.start_btn.pressed.connect(self.open_startWindow)
        
    def enableStart_btn(self,index):
        if index != 0:
            self.start_btn.setStyleSheet("background-color: rgb(47, 144, 145); \n""color: rgb(243, 243, 243); \n""border-radius: 15px;\n" )
            self.start_btn.setEnabled(True)
            
    def open_startWindow(self):  
        msg = 'start'
        send(msg)
        print('Message: {} sent to server!'.format(msg)) 
        tid.start()
        
        StartWindow=start_window()
        widget.addWidget(StartWindow)
        widget.setCurrentIndex(widget.currentIndex()+1)

        
 
class start_window(QMainWindow):
    
    def __init__(self):
        super(start_window,self).__init__()
        loadUi("started.ui",self)
        self.count = 0
        self.download_btn.setEnabled(False)
        self.show_notify_btn.pressed.connect(self.Open_Notify)
        self.stop_btn.pressed.connect(self.stop_lecture)
        self.download_btn.pressed.connect(self.Download_and_Open_Dialog)
        self.attendance_sheet = []

        
    def stop_lecture(self):
        tid.terminate()
        msg = 'stop'
        send(msg)
        print('Message: {} sent to server!'.format(msg))
        self.attendance_sheet = receive(s)
        while len(self.attendance_sheet) == 0:
            self.attendance_sheet = receive(s)
        print(self.attendance_sheet) 
        self.download_btn.setStyleSheet("background-color: rgb(47, 144, 145); \n""color: rgb(243, 243, 243); \n""border-radius: 15px;\n" )
        self.download_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: rgb(216, 216, 216); \n""color: rgb(99, 99, 99); \n" "border-radius: 15px;\n " "border-style:outset;\n" "border-color:  rgb(99, 99, 99);")
    
    def Open_Notify(self):
        NotifyWindow=notif_window()
        widget.addWidget(NotifyWindow)
        widget.setCurrentIndex(widget.currentIndex()+1)
        self.count+=1
        
    def Download_and_Open_Dialog(self):
        with open('./Downloads/attendance_sheet_{}.csv'.format(date.today()) , mode='w', newline='') as file:
            file_write = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in self.attendance_sheet:        
                file_write.writerow(row)  
       
        msg = QMessageBox()
        msg.setWindowTitle("Downloaded")
        msg.setText("Your Attendance Sheet is Downloaded!")
        msg.exec_()
        
        
class notif_window(QMainWindow):
    
    def __init__(self):
        super(notif_window,self).__init__()
        uic.loadUi("notification.ui",self)
        self.buttonNumber=0
        self.Hbox = QHBoxLayout()
        self.vbox = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setWidgetResizable(True)
        self.back_btn.pressed.connect(self.Go_back)
        self.vbox.addLayout(self.Hbox)
        self.setLayout(self.vbox)
        
        self.startToListen()
        
    def Go_back(self):
        widget.setCurrentIndex(widget.currentIndex()-1)
        widget.removeWidget(self)
        

    def startToListen(self):
        current_reports_list = glob.glob("./Reports/*.csv")
        for report in current_reports_list:
            self.Create_Viewbtn()
        
    def Create_Viewbtn(self):
        self.buttonNumber +=1
        self.view_btn = QtWidgets.QPushButton("View Report %d" % self.buttonNumber)
        self.view_btn.setObjectName("View Report%d" % self.buttonNumber)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.view_btn.setFont(font)
        self.view_btn.setStyleSheet("background-color: rgb(47, 144, 145);\n""color: rgb(243, 243, 243);\n"" border-radius: 8px; \n" "padding:8px;" )

        self.vbox.insertWidget(self.Hbox.count() - 1, self.view_btn)  
        self.view_btn.pressed.connect(self.view_file)
        
        self.sample_btn = QtWidgets.QPushButton("View Samples %d" % self.buttonNumber)
        self.sample_btn.setObjectName("View Samples %d" % self.buttonNumber)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.sample_btn.setFont(font)
        self.sample_btn.setStyleSheet("background-color: rgb(47, 144, 145);\n""color: rgb(243, 243, 243);\n"" border-radius: 8px; \n" "padding:8px;" )

        self.vbox.insertWidget(self.Hbox.count() - 1, self.sample_btn)  
        self.sample_btn.pressed.connect(self.view_samples)
      
    def view_file(self):
        sending_btn = self.sender()
        btn_name = str(sending_btn.objectName())
        btn_name=btn_name.split(sep=" ")[1]
        file_name = "./Reports/" + btn_name + ".csv"
        os.startfile(os.path.normpath(file_name))
     
    def view_samples(self):
        sending_btn = self.sender()
        btn_name = str(sending_btn.objectName())
        btn_name=btn_name.split(sep=" ")[2]        
        file_name = "./images/Report {}".format(btn_name)
        os.startfile(os.path.normpath(file_name))
#main 
if __name__ == '__main__': 
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((socket.gethostname(), 5000))
    tid = multiprocessing.Process(target=receiving_reports, args = (s, ))
    app = QtWidgets.QApplication(sys.argv)
    FirstWindow = courses_window()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(FirstWindow)
    widget.setFixedHeight(515)
    widget.setFixedWidth(750)
    widget.show()
    sys.exit(app.exec_())