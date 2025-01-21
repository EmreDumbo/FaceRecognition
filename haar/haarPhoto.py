import cv2
import numpy as np 

def detectFaces(image, faceCascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30,30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

    return image

def detectEyes(image, eyeCascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)

    return image

def main():
    faceCascade = cv2.CascadeClassifier('/Users/emre/Desktop/staj/cascades/haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('/Users/emre/Desktop/staj/cascades/haarcascade_eye.xml')

    img = cv2.imread('/Users/emre/Desktop/staj/images/eyedetect.jpg')

    imgFaces = detectFaces(img.copy(), faceCascade)

    imgFacesEyes = detectEyes(imgFaces, eyeCascade)

    cv2.imshow('Face and Eye Detection', imgFacesEyes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
