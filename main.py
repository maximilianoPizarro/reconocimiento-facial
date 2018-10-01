# -*- coding: utf-8 -*-

__author__      = "UNLa"
__copyright__   = "Copyright 2018, fundamentos de la teoria de la computación"
__version__ = "0.1"
__license__ = "UNLa"


from PySide import QtCore, QtGui
import sys
import cv2
import numpy as np
import threading
import time
import queue
from pyside_uicfix import loadUiType
import os

running = False
capture_thread = None
#form_class = loadUiType("simple.ui")[0]
form_class = loadUiType("view/mainwindow.ui")[0]
q = queue.Queue()
cascPath = "resources/haarcascade_frontalface_default.xml"

def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)


    while(running):
        frame = {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)

class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()



class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)       

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        
        

    def start_clicked(self):
        global running
        running = True
        capture_thread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Inciando...')
        
    def pushButton_clicked(self):
        #self.nameText.setText("safa")   set texto
        #print(self.nameText.text())    get texto
        img_name = self.nameText.text()+".png".format(0)
        frame = q.get()
        img = frame["img"]
        
        cv2.imwrite(os.path.join('resources' , img_name), img)
            

    def update_frame(self):
        if not q.empty():
            self.startButton.setText('Live')
            frame = q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)
            
            faceCascade = cv2.CascadeClassifier(cascPath)
            # Capture frame-by-frame
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
            faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def closeEvent(self, event):
        global running
        running = False



capture_thread = threading.Thread(target=grab, args = (0, q, 1920, 1080, 30))

app = QtGui.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('UNLa Fundamentos de la Teoria de la Computación')
w.show()
app.exec_()
    
    
    