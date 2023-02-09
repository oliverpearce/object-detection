import cv2
import numpy as np
import time

np.random.seed(200) #classes aren't random coloured each time
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    # read coco.names file and add all the different objects/classes
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList = np.random.uniform(low = 0, high = 255, size = (len(self.classesList), 3))

        # print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read() # ret is if camera is usable, frame is frame

        while ret:
            classLabelIDs, confidences, bboxs = self.net.detect(frame, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = f'{classLabel}:{classConfidence:.2f}'

                    x, y, w, h = bbox

                    cv2.rectangle(frame, (x,y), (x + w, y + h), color = classColor, thickness = 1)
                    cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)

                    # draw corners on the boxes!
                    lineWidth = min(int(w * 0.3), int(h*0.3))
                    
                    # top left
                    cv2.line(frame, (x, y), (x + lineWidth, y), classColor, thickness = 5)
                    cv2.line(frame, (x, y), (x, y + lineWidth), classColor, thickness = 5)

                    # top right
                    cv2.line(frame, (x + w, y), (x + w - lineWidth, y), classColor, thickness = 5)
                    cv2.line(frame, (x + w, y), (x + w, y + lineWidth), classColor, thickness = 5)

                    # bottom left
                    cv2.line(frame, (x, y + h), (x + lineWidth, y + h), classColor, thickness = 5)
                    cv2.line(frame, (x, y + h), (x, y + h - lineWidth), classColor, thickness = 5)

                    # bottom right
                    cv2.line(frame, (x + w, y + h), (x + w - lineWidth, y + h), classColor, thickness = 5)
                    cv2.line(frame, (x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness = 5)

            cv2.imshow('friendship window', frame)

            if cv2.waitKey(1) == ord('q'): # every ms check if q is being pressed
                break

            (success, frame) = cap.read()

        cv2.destroyAllWindows()

            