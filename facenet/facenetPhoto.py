from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img1 = Image.open("/Users/emre/Desktop/staj/images/depp.jpg")
img2 = Image.open("/Users/emre/Desktop/staj/images/emre2.jpg")


faces1, _ = mtcnn.detect(img1)
faces2, _ = mtcnn.detect(img2)

if faces1 is not None and faces2 is not None:
    aligned1 = mtcnn(img1)
    aligned2 = mtcnn(img2)
    

    aligned1 = aligned1.unsqueeze(0) if aligned1 is not None else None
    aligned2 = aligned2.unsqueeze(0) if aligned2 is not None else None
    
    if aligned1 is not None and aligned2 is not None:
        embeddings1 = resnet(aligned1).detach()
        embeddings2 = resnet(aligned2).detach()

        
        distance = (embeddings1 - embeddings2).norm().item()
        if distance < 1.0:  
            print("Same person")
        else:
            print("Different persons")
    else:
        print("Face not detected in one or both images.")
else:
    print("No faces detected in one or both images.")
