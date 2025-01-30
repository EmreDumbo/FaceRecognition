import cv2
import numpy as np 
from imutils.video import VideoStream
import time
import imutils
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascades", type=str, default="cascades", 
                help="path to input dir. containing haar cascades")
args = vars(ap.parse_args())

detectorPaths = {
    "face" : "haarcascade_frontalface_default.xml",
    "eyes" : "haarcascade_eye.xml",
}
detectors = {}
for (name, path) in detectorPaths.items():
    path = os.path.sep.join([args["cascades"], path])
    detectors[name] = cv2.CascadeClassifier(path)

vs = VideoStream(src=0).start()
font = cv2.FONT_HERSHEY_SIMPLEX
prevFrameTime = 0 
newFrameTime = 0


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newFrameTime = time.time()
    fps = int(1/(newFrameTime - prevFrameTime))
    prevFrameTime = newFrameTime
    fps = str(fps)
    cv2.putText(frame, fps, (7,70), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
    faceCascade = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags= cv2.CASCADE_SCALE_IMAGE)
    for(fX, fY, fW, fH) in faceCascade:
        faceROI = gray[fY: fY + fH , fX: fX + fW]
        eyeCascade = detectors["eyes"].detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10, minSize=(15,15), flags = cv2.CASCADE_SCALE_IMAGE)
        for (eX, eY, eW, eH) in eyeCascade:
            ptA = (fX + eX, fY + eY)
            ptB = (fX + eX + eW, fY + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)        
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) == 27:
        break  
cv2.destroyAllWindows()
vs.stop()