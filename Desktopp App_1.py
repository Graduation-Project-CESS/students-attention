import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QDialog, QVBoxLayout, QHBoxLayout

filestoWatch  = "./Reports/Report1.xlsx"
editorProgram = 'notepad'

class courses_window(QMainWindow):
    def __init__(self):
        super(courses_window,self).__init__()
        uic.loadUi("courses.ui",self)
        self.start_btn.setEnabled(False)
        self.comboBox.currentIndexChanged.connect(self.enableStart_btn)
        self.start_btn.pressed.connect(self.open_startWindow)
        
    def enableStart_btn(self,index):
        print(index)
        if index != 0:
            self.start_btn.setStyleSheet("background-color: rgb(47, 144, 145); \n""color: rgb(243, 243, 243); \n""border-radius: 15px;\n" )
            self.start_btn.setEnabled(True)
            
    def open_startWindow(self):  
        StartWindow=start_window()
        widget.addWidget(StartWindow)
        widget.setCurrentIndex(widget.currentIndex()+1)
   
        
 
class start_window(QMainWindow):
    
    def __init__(self):
        super(start_window,self).__init__()
        loadUi("started.ui",self)
        self.count =0
        self.download_btn.setEnabled(False)
        self.show_notify_btn.pressed.connect(self.Open_Notify)
        self.stop_btn.pressed.connect(self.enable_download)
        self.download_btn.pressed.connect(self.Open_Dialog)

        
    def enable_download(self):
        self.download_btn.setStyleSheet("background-color: rgb(47, 144, 145); \n""color: rgb(243, 243, 243); \n""border-radius: 15px;\n" )
        self.download_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: rgb(216, 216, 216); \n""color: rgb(99, 99, 99); \n" "border-radius: 15px;\n " "border-style:outset;\n" "border-color:  rgb(99, 99, 99);")
    
    def Open_Notify(self):
        NotiftWindow=notif_window()
        widget.addWidget(NotiftWindow)
        widget.setCurrentIndex(widget.currentIndex()+1)
        self.count+=1
        
    def Open_Dialog(self):
        dialog=Dialog()
        widget.addWidget(dialog)      

        if self.count > 0:
            widget.setCurrentIndex(widget.currentIndex()+self.count+1)
        else:
            widget.setCurrentIndex(widget.currentIndex()+1)
            
        
        
class notif_window(QMainWindow):
    
    def __init__(self):
        super(notif_window,self).__init__()
        uic.loadUi("notification.ui",self)
        self.buttonNumber=0
        self.Hbox = QVBoxLayout()
        self.vbox = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setWidgetResizable(True)
        self.back_btn.pressed.connect(self.Go_back)
        self.vbox.addLayout(self.Hbox)
        self.setLayout(self.vbox)
        
        self.listener = QtCore.QFileSystemWatcher(self)
        self.startToListen()
        self.listener.directoryChanged.connect(self.checkFile)
        
    def Go_back(self):
        widget.setCurrentIndex(widget.currentIndex()-1)
        

    def startToListen(self):
        fileInfo = QtCore.QFileInfo(filestoWatch)
        if fileInfo.exists():
            print("stif",fileInfo.absolutePath())
            self.Create_Viewbtn()
            return
        elif fileInfo.absolutePath() not in self.listener.directories():
            print("stel",fileInfo.absolutePath())
            self.listener.addPath(fileInfo.absolutePath())

        

    def checkFile(self, path):
        fileInfo = QtCore.QFileInfo(filestoWatch)
        if fileInfo.exists():
            print("chif",fileInfo.absolutePath())
            self.Create_Viewbtn()
        else:
            print("chels:",fileInfo.absolutePath())
            # file has been [re]moved/renamed, maybe do something here...
            pass
        
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
        
        self.btn_label =QtWidgets.QLabel("New attention report")
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btn_label.setFont(font)
        self.btn_label.setStyleSheet("color: rgb(33, 82, 115);\n""font: 57 11pt \"Montserrat Medium\";")
        self.vbox.insertWidget(self.Hbox.count() - 1, self.btn_label)
        self.vbox.insertWidget(self.Hbox.count() - 1, self.view_btn)
        
        self.view_btn.pressed.connect(self.view_file)
        
    def view_file(self):
        sending_btn = self.sender()
        btn_name = str(sending_btn.objectName())
        btn_name=btn_name.split(sep=" ")[1]
        print("button name",btn_name)
        file_name= os.path.split(filestoWatch)[1]
        file_name=file_name.split(".")[0]
        print("file_name",file_name)
        if (btn_name==file_name):
            process= QtCore.QProcess(self)
            process.start(editorProgram, [filestoWatch])
            self.setEnabled(False)
            process.finished.connect(lambda: self.setEnabled(True))
        
class Dialog(QDialog):
    def __init__(self):
        super(Dialog,self).__init__()
        loadUi("dialog.ui",self) 
        self.ok_btn.pressed.connect(self.Goto_First_Window)
        self.cancel_btn.pressed.connect(self.Close_App)
        
    def Goto_First_Window(self):
        firstWindow = courses_window()
        widget.addWidget(firstWindow)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def Close_App(self):
        widget.removeWidget(self)
#main 
app = QtWidgets.QApplication(sys.argv)
FirstWindow = courses_window()
widget = QtWidgets.QStackedWidget()
widget.addWidget(FirstWindow)
widget.setFixedHeight(515)
widget.setFixedWidth(750)
widget.show()
sys.exit(app.exec_())