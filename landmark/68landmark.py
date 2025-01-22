from imutils import face_utils
import numpy as np
import argparse 
import imutils
import dlib 
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/emre/Desktop/staj/landmark/Shape Predictor 68 Face Landmarks.dat")

image = cv2.imread("images/eyedetect.jpg")
image = imutils.resize(image, width=500)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rects = detector(rgb, 1)

for rect in (rects):
    shape = predictor(rgb, rect)
    shape = face_utils.shape_to_np(shape)
    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    
    for (x,y) in shape:
        cv2.circle(image, (x,y), 1, (0,0,255), -1)

    cv2.imshow("output", image)
    for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y1:y1 + h1, x1:x1 + w1]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        
        cv2.imshow(name, roi)

cv2.waitKey(0)
