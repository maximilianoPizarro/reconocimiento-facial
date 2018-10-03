import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic

if __name__ == '__main__':
    
    app = QApplication(sys.argv)    
    w = QWidget()
    uic.loadUi("../view/mainwindow.ui", w)
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()
    
    sys.exit(app.exec_())