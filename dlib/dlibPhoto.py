import argparse
import cv2
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--upsample", type=int, default=1, help= "# of times to upsample")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
image = cv2.imread("images/eyedetect.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rects = detector(rgb, args["upsample"])

def convertAndTrimBb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    w = endX - startX
    h = endY - startY

    return (startX, startY, w, h)

boxes = [convertAndTrimBb(image, r) for r in rects]

for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)