from Detector import *

modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

classFile="coco.names"
threshold=0.6
videoPath=0 #0 for webcam

detector=Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        self.m_w11 = QtWidgets.QWidget()
        self.m_w12 = QtWidgets.QWidget()
        self.m_w21 = QtWidgets.QWidget()
        self.m_w22 = QtWidgets.QWidget()

        lay = QtWidgets.QGridLayout(central_widget)

        for w, (r, c) in zip(
            (self.m_w11, self.m_w12, self.m_w21, self.m_w22),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        ):
            lay.addWidget(w, r, c)
        for c in range(2):
            lay.setColumnStretch(c, 1)
        for r in range(2):
            lay.setRowStretch(r, 1)

        lay = QtWidgets.QVBoxLayout(self.m_w11)

        self.setMinimumSize(QSize(900 , 350))    
        self.setWindowTitle("Object Detection")
    

        pybutton1 = QPushButton('Recognize', self)
        pybutton1.clicked.connect(self.clickMethodRecog)
        pybutton1.resize(300,75)
        pybutton1.move(300, 250)
        pybutton1.setStyleSheet("""background-image: url("E:/Object_Detecton/images.jfif");border: 1px solid white;color: white;padding: 16px 32px;text-align: center;font-size: 16px;
        margin: 4px 2px;border-radius:15px;""")

    def clickMethodRecog(self):
        detector.predictVideo(videoPath,threshold)

stylesheet = """
    MainWindow {
        background-image: url("E:/Object_Detecton/1.jpeg"); 
        background-repeat: no-repeat; 
        background-position: center;
        opacity: 0.6;
    }
"""

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    w = MainWindow()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())
