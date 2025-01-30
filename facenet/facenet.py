import os 
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType

def encode(img):
    res = resnet(torch.Tensor(img))
    return res

def detect_box(self, img, save_path=None):
    
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

saved_pictures = "images"
all_people_faces = {}


for person_face in os.listdir(saved_pictures):
    if person_face.endswith('.jpg'):  
        filename, extension = os.path.splitext(person_face)
        
        img = cv2.imread(f'{saved_pictures}/{person_face}')
        cropped = mtcnn(img)
        
        if cropped is not None:
            all_people_faces[filename] = encode(cropped)[0, :]

def detect(capture_flag=0, euclidean_threshold=0.7):
    video = cv2.VideoCapture(capture_flag)
    while video.grab():
        _, img0 = video.retrieve()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(x) for x in box]
                img_embedding = encode(cropped.unsqueeze(0))
                detect_dict = {}
                
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                
                min_key = min(detect_dict, key=detect_dict.get)

                if detect_dict[min_key] >= euclidean_threshold:
                    min_key = 'unknown'
                
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  img0, min_key, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
        cv2.imshow("output", img0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect(0, 0.7)