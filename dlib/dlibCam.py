import dlib 
import cv2
import time

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
fps = cam.get(cv2.CAP_PROP_FPS)
font = cv2.FONT_HERSHEY_SIMPLEX
prevFrameTime = 0 
newFrameTime = 0

while True:
    retVal , img = cam.read()
    rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    newFrameTime = time.time()
    fps = int(1/(newFrameTime - prevFrameTime))
    prevFrameTime = newFrameTime
    fps = str(fps)
    cv2.putText(img, fps, (7,70), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
    dets = detector(rgbImage)
    for det in dets:
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), (255 ,0, 0), 2)
    cv2.resize(img, (100,100))    
    cv2.imshow("my webcam", img)
    if cv2.waitKey(1) == 27:
        break  
cv2.destroyAllWindows()