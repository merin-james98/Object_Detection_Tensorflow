import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from numba import jit,cuda

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QInputDialog
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize    

np.random.seed(20) #reproducable result all the runs

class Detector:
    def __init__(self):
        pass
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList=f.read().splitlines()
        #color LIst
        self.colorList=np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

    #Extracting Model   
    def downloadModel(self,modelURL):
        fileName=os.path.basename(modelURL)
        self.modelName=fileName[:fileName.index('.')]
        
        self.cacheDir="./pretrained_models"
        os.makedirs(self.cacheDir,exist_ok=True)
        get_file(fname=fileName,origin=modelURL,cache_dir=self.cacheDir,cache_subdir="checkpoints",extract=True)

    #load model   
    def loadModel(self):
        print("Loading Model"+self.modelName)
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,"saved_model"))
        print("Model"+ self.modelName+"loaded Successfully")
        
    #Bounding boxs around the objects  
    def createBoundigBox(self,image,threshold):
        inputTensor=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor,dtype=tf.uint8)
        inputTensor=inputTensor[tf.newaxis,...]
        
        detections=self.model(inputTensor)
        bboxs=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()
                
        imH,imW,imC=image.shape
        
        bboxIdx=tf.image.non_max_suppression(bboxs,classScores, max_output_size=50,iou_threshold=0.6, score_threshold=0.6)
        
        if len(bboxs)!=0:
            for i in bboxIdx:
                bbox=tuple(bboxs[i].tolist())
                classConfidence=round(100*classScores[i])
                classIndex=classIndexes[i]
                
                classLabeltext=self.classesList[classIndex].upper()
                classColor=self.colorList[classIndex]
                
                displayText='{}:{}%'.format(classLabeltext,classConfidence)
                
                ymin,xmin,ymax,xmax=bbox
                
                ymin,xmin,ymax,xmax=(xmin*imW, xmax*imW,ymin*imH,ymax*imH)
                ymin,xmin,ymax,xmax=int(xmin),int(xmax),int(ymin),int(ymax)

                cv2.putText(image,displayText,(xmax+10,ymax+20),cv2.FONT_HERSHEY_PLAIN,1,classColor,2)
                
                lineWidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))
                
                cv2.line(image,(xmin,ymin),(xmin+lineWidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmin,ymin),(xmin,ymin+lineWidth),classColor,thickness=5)
                
                cv2.line(image,(xmax,ymin),(xmax-lineWidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmax,ymin),(xmax,ymin+lineWidth),classColor,thickness=5)
                
                cv2.line(image,(xmin,ymax),(xmin+lineWidth,ymax),classColor,thickness=5)
                cv2.line(image,(xmin,ymax),(xmin,ymax-lineWidth),classColor,thickness=5)
                
                cv2.line(image,(xmax,ymax),(xmax-lineWidth,ymax),classColor,thickness=5)
                cv2.line(image,(xmax,ymax),(xmax,ymax-lineWidth),classColor,thickness=5)             
        return image
                
     #Object Detection using Webcam           
    def predictVideo(self,videoPath,threshold): 
    
        cap=cv2.VideoCapture(videoPath)
        
        cap.set(3, 1280)
        cap.set(4, 720)
        if(cap.isOpened()==False):
            print("Error Opening File...")
            return
        (success,image)=cap.read()
        startTime=0
        while success:
            
            currentTime=time.time()
            fps=1/(currentTime-startTime)
            startTime=currentTime
            bboximage=self.createBoundigBox(image,threshold)
            cv2.putText(bboximage,"FPS: "+str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            
            cv2.imshow("Result ", bboximage)
            key=cv2.waitKey(1) & 0xFF
            if key==ord("q"):
                break
            (success,image)=cap.read()
        cap.release()
        cv2.destroyAllWindows()
        


        
 
