import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton,QVBoxLayout,QLabel,QHBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import*
from PyQt5.QtCore import *

import detector
import threading
from queue import Queue


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.neutral=QPixmap('public/neutral.png')
        self.doubt=QPixmap('public/doubt.png')
        self.question=QPixmap('public/question_sizeup.png')
        self.question=self.question.scaledToWidth(200)

        self.lbl_img=QLabel()
        self.lbl_img.setPixmap(self.neutral)

        q = Queue()

        t = threading.Thread(target=detector.detector, args=(q,))
        t.start()

        print(q.empty())
        if q.empty() == False:
            data = q.get()
            print(data)

        btn=QPushButton(self)
        btn.setIcon(QIcon('public/question.png'))
        btn.setCheckable(True)
        btn.clicked.connect(self.btn_clicked)

        vbox=QVBoxLayout()

        vbox.addWidget(self.lbl_img)
        vbox.addWidget(btn)


        self.setLayout(vbox)
        self.setWindowTitle('Untacticon UI')
        self.resize(200,300)
        self.center()
        self.show()

    def btn_clicked(self):
        #QMessageBox.about(self,"message","질문하였습니다.")
        self.lbl_img.setPixmap(self.question)
        time.sleep(1)
        #self.lbl_img.setPixmap(self.doubt)



    def center(self):
        qr=self.frameGeometry()
        cp=QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__=='__main__':
    app=QApplication(sys.argv)
    ex=MyApp()
    sys.exit(app.exec_())
