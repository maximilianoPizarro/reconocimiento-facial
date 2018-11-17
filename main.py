# -*- coding: utf-8 -*-

__author__      = "UNLa"
__copyright__   = "Copyright 2018, fundamentos de la teoria de la computación"
__version__ = "0.1"
__license__ = "UNLa"


from PyQt5 import QtCore, QtGui, uic,QtWidgets
import sys
#import numpy
import cv2.cv2
#import numpy as np
import threading
#import time
#from numpy.core import multiarray
import queue
#from resources import pyside_uicfix 
import os
import pickle
from random import choice

running = False
capture_thread = None
#form_class = loadUiType("simple.ui")[0]
q = queue.Queue()
face_cascade = cv2.CascadeClassifier('resources/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('resources/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('resources/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.cv2.VideoCapture(cam)
    capture.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.cv2.CAP_PROP_FPS, fps)


    while(running):
        frame = {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)

class OwnImageWidget(QtWidgets.QWidget):
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



class MyWindowClass(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        uic.loadUi("view/mainwindow.ui",self)
        #self.setupUi(self)
        self.startButton.clicked.connect(self.start_clicked)
        self.pushButton.clicked.connect(self.agregar_foto)
        self.pushButton_2.clicked.connect(self.entrenar)
        self.pushButton_3.clicked.connect(self.detener)
        
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)       

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        
        extractAction = QtWidgets.QAction("Salir", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Salir de la aplicación')
        extractAction.triggered.connect(self.close_application)
        
        menu=self.menuBar()
        menu.addAction(extractAction)
        menu.triggered[QtWidgets.QAction].connect(self.statusbar)
        
    def statusbar(self,q):
        if(q.text()=="Inciar Busqueda"):
            self.mensajeLabel.setText("reconocer y buscar")     

    def start_clicked(self):
        global running
        running = True
        capture_thread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Inciando...')
        
    def agregar_foto(self):
        #self.nameText.setText("safa")   set texto
        #print(self.nameText.text())    get texto
        if not q.empty():
                if str(self.mensajeLabel.text())!="":
                        os.makedirs("images/"+self.nameText.text(), exist_ok=True)    
                            
                        img_name =choice("abcde")+"-img.png".format(0)
                        frame = q.get()
                        img = frame["img"]        
                        #cv2.imwrite(os.path.join('resources' , img_name), img)
                        
                        gray = cv2.cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)   
                        faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=(30, 30)
                            )                        
                        # Draw a rectangle around the faces
                        for (x, y, w, h) in faces:
                            roi = img[y:y+h, x:x+w]
                            cv2.cv2.imwrite(os.path.join("images/"+self.nameText.text() , img_name), roi)
                        self.mensajeLabel.setText("Agregado exitosamente!")
                else:
                    self.mensajeLabel.setText("")   
        else:
                self.mensajeLabel.setText("iniciar camara")  
                
    def entrenar(self):
        os.system("python faces-train.py")            
        
    def update_frame(self):
        if not q.empty():
            self.startButton.setText('Iniciado')
            frame = q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            img = cv2.cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.cv2.INTER_CUBIC)
            img = cv2.cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)
            
            # Capture frame-by-frame
            
            gray = cv2.cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)   
            faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                id_,conf = recognizer.predict(roi_gray)
                if conf>=4 and conf <= 85:
                		#print(5: #id_)
                		#print(labels[id_])
                		font = cv2.FONT_HERSHEY_SIMPLEX
                		name = labels[id_]
                		color = (255, 255, 255)
                		stroke = 2
                		cv2.cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.cv2.LINE_AA)
                img_item = "7.png"
                cv2.cv2.imwrite(img_item, roi_color)

                #cv2.putText(img, 'This one!', (x+w, y+h), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def closeEvent(self, event):
        global running
        running = False
        
    def detener(self):
        if capture_thread.is_alive():
            capture_thread._delete()
    
    def close_application(self):
        global running
        running = False
        if capture_thread.is_alive():
            capture_thread._delete()
        sys.exit()   



capture_thread = threading.Thread(target=grab, args = (0, q, 1920, 1080, 30))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindowClass(None)
    w.setWindowTitle('UNLa Fundamentos de la Teoria de la Computación')
    w.show()
    app.exec_()
    
    
    