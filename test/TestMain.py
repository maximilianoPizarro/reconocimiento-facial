# main.py

import sys

from PySide.QtUiTools import QUiLoader

from PySide.QtGui import *
from PySide.QtCore import *
import cv2

class MainWindow(QMainWindow):
    def __init__(self,window):
        super(MainWindow,self).__init__()
        window.show()
        menu=window.menuBar()
        menu.triggered[QAction].connect(self.statusbar)

    def statusbar(self,q):
        if(q.text()=="Inciar Busqueda"):
            self.mostrarcamara(window)            
        if(q.text()=="Agregar Imagen"):        
            self.pruebacamara(window)            

            
    def mostrarcamara(self,window):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        img_counter = 0
        while True:            
            ret, window.cam_frame = cam.read()
            cv2.imshow("test", window.cam_frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, window.cam_frame)
                print("{} written!".format(img_name))
                img_counter += 1
            
        cam.release()
        cv2.destroyAllWindows()
        
    def pruebacamara(self,window):
        cam=cv2.VideoCapture(0)
        ret, window.cam_frame = cam.read()
        window.cam_frame.capture(cam)
     
if __name__ == "__main__":
    app = QApplication(sys.argv)    
    file = QFile("../view/mainwindow.ui")
    file.open(QFile.ReadOnly)
    loader = QUiLoader()
    window=loader.load(file)
    mainWin = MainWindow(window)
    ret = app.exec_()
    sys.exit( ret )
