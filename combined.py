import cv2
import numpy as np
from ultralytics import YOLO


word_model = YOLO('yolov8s.pt')
letter_model = YOLO('yolov8s.pt')


def load_image(img_path: str):
    frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return frame


word_model = YOLO("./word_model.pt")
def infer_words(img: np.ndarray):
    global word_model
    # returns (word_image, bb, class)
    results = word_model.predict(img)
    returnable = []
    while len(results[0])!=0:
      for r in results:
          boxes = r.boxes
          for box in boxes:
              x1, y1, x2, y2 = box.xyxy[0].cpu().data.numpy().astype(int).tolist()
              returnable.append((img[y1:y2, x1:x2].copy(), box,  word_model.names[int(box.cls)]))
              img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),-1)
      results = word_model.predict(img)
    return returnable

def infer_letters(img: np.ndarray):
    letter_model.predict(img)


def save_to_file():
    pass
