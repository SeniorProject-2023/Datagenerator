import os
import re
import sys
from functools import reduce
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_metric
from deskew import determine_skew
from jdeskew.utility import rotate
from more_itertools import split_when
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from synthetic_words_ramez import cleanup

word_model = YOLO("./word_model.pt")
letter_model = YOLO("./letter_model.pt")

name_to_unicode = {
    'baa': 1576,
    'hamza-sater': 1569,
    'ya': 1609,
    'sad': 1589,
    'kaf': 1603,
    'ha-last': 1607,
    'waw': 1608,
    'sen': 1587,
    'ta': 1578,
    'lam': 1604,
    'alef-hamza-under': 1573,
    'dal': 1583,
    'non': 1606,
    'zen': 1586,
    'ya-hamza': 1574,
    '3en': 1593,
    '8en': 1594,
    '5a': 1582,
    'fa-3dot': 1700,
    'mem': 1605,
    'fa': 1601,
    'alef-hamza-up': 1571,
    'ya-dot': 1610,
    'thal': 1584,
    'waw-hamza': 1572,
    'tha': 1579,
    'taa-': 1591,
    'thaa-': ord("ث"),
    'ha': 1726,
    '7a': 1581,
    'baa-3dot': 1662,
    'ra': 1585,
    'dad': 1590,
    'sheen': 1588,
    'gem': 1580,
    'qaf': 1602,
    'zaa': 1592,
    'gem-three-dot': 1670,
    'alef-mad': 1570,
    'taa-marpota': 1577,
    'alef-': 1575,
    'lam-alef': 65276,
    'lam-alef-la': 65275,
    'space': ord(" "),
}


def load_image(img_path: str):
    frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return frame


def infer_words(img: np.ndarray):
    global word_model
    # returns List[(word_image, bb, class)]
    results = word_model.predict(img, verbose=False)
    returnable = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu(
            ).data.numpy().astype(int).tolist()
            returnable.append(
                (img[y1:y2, x1:x2].copy(), box, word_model.names[int(box.cls)]))
            img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                (255, 255, 255), -1)
    return returnable


def infer_letters(img: np.ndarray, conf=0.5, model=letter_model, convert_names=False):
    results = model.predict(img, conf=conf, verbose=False)[0]
    boxes = [box.xyxy[0].tolist() for box in results.boxes]  # x1, y1, x2, y2
    xs = [box[2] for box in boxes]
    sorted_indices = np.argsort(xs)[::-1]  # sort by max x2

    letters = [int(results.boxes.cls[i]) for i in sorted_indices]
    letters = [model.names[class_no] for class_no in letters]

    if convert_names:
        letters = [name_to_unicode[name] for name in letters]
        letters = [chr(unc) for unc in letters]

    word = "".join(letters)
    return word


def groupbyrow(boxes):
    def get_y_center(b):
        x1, y1, x2, y2 = b
        return (y1 + y2) / 2

    def not_vertically_overlapping(b1, b2):
        _, up1, _, down1 = b1
        _, up2, _, down2 = b2
        return down1 < up2 or (down1 - up2) < (up2 - up1)

    sorted_boxes = sorted(boxes, key=get_y_center)
    return list(split_when(sorted_boxes, not_vertically_overlapping))


def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
        return False
    else:
        return True


def merge_boxes(boxes, iou_thresh=0.3):
    boxes = sorted(boxes, key=lambda b: b[0])

    if len(boxes) == 0:
        return []

    merged_boxes = [boxes[0]]

    for k in range(1, len(boxes)):
        prev_box = merged_boxes[-1]

        x1 = boxes[k][0]
        y1 = boxes[k][1]
        x2 = boxes[k][2]
        y2 = boxes[k][3]
        x1_other = prev_box[0]
        y1_other = prev_box[1]
        x2_other = prev_box[2]
        y2_other = prev_box[3]

        intersection_area = 0 if x2_other < x1 else (x2_other - x1) * (max(y2, y2_other) - max(y1, y1_other))
        area = (x2 - x1) * (y2 - y1)
        area_other = (x2_other - x1_other) * (y2_other - y1_other)
        union_area = area + area_other - intersection_area
        if intersection_area / union_area > iou_thresh or intersection_area / area > 0.7 or intersection_area / area_other > 0.7:
            x1_new = np.minimum(x1, x1_other)
            y1_new = np.minimum(y1, y1_other)
            x2_new = np.maximum(x2, x2_other)
            y2_new = np.maximum(y2, y2_other)

            merged_boxes[-1] = ([x1_new, y1_new, x2_new, y2_new])
        else:
            merged_boxes.append([x1, y1, x2, y2])

    return merged_boxes


def evaluate_yolo_model(test_dir, model):
    test_dir = Path(test_dir)
    assert test_dir.exists()
    assert "images" in os.listdir(test_dir)
    assert "labels" in os.listdir(test_dir)

    predicted_texts = []
    ground_truths = []
    for filename in os.listdir(Path(test_dir) / "images"):
        match = re.search("(.+)-\d+.png", filename)
        if not match:
            continue
        else:
            ground_truths.append(match.group(1))
            img = cv2.cvtColor(cv2.imread(str(Path(test_dir) / "images/" / filename)), cv2.COLOR_BGR2RGB)
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            predicted_texts.append(infer_letters(img))

    cer_metric = load_metric("cer")
    return cer_metric.compute(predictions=predicted_texts, references=ground_truths)


def map2d(func, grid):
    return [[func(value) for value in row] for row in grid]


# if __name__ == '__main__':
#     print(evaluate_yolo_model("./output/train", word_model))

if __name__ == '__main__':
    cleanup(Path("./output"))

    for iter, img_path in enumerate(sys.argv[1:]):
        img = cv2.imread(img_path)
        cv2.imwrite(f"./output/{iter}_orig_img.png", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = rotate(img, determine_skew(img), border_mode=cv2.BORDER_CONSTANT, border_value=255)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"./output/{iter}_filtered_img.png", img)

        img_before_inference = np.copy(img)
        pil_img_before_inference = Image.fromarray(img)

        word_boxes = infer_words(img)

        box_bounds = [box[1].xyxy[0].numpy() for box in word_boxes]
        box_bounds = [
            b + np.multiply(np.array([b[0] - b[2], b[1] - b[3], b[2] - b[0], b[3] - b[1]]), [0.05, 0.2, 0.05, 0.05]) for
            b in box_bounds]  # pad the boxes
        box_bounds = [np.array([max(0, b[0]), max(0, b[1]), min(img.shape[1] - 1, b[2]), min(img.shape[0] - 1, b[3])])
                      for b in box_bounds]  # make sure padding doesn't spill out of image
        box_bounds = reduce(lambda x, y: np.vstack((x, y)), box_bounds)

        rows_of_boxes = groupbyrow(box_bounds)
        rows_of_boxes = [merge_boxes(row) for row in rows_of_boxes]

        annotator = Annotator(img_before_inference, line_width=1)
        flattened_row_of_boxes = [i for row in rows_of_boxes for i in
                                  sorted(row, key=lambda b: b[2], reverse=True)]
        for i, b in enumerate(flattened_row_of_boxes):
            annotator.box_label(b, str(i))
        frame = annotator.result()
        cv2.imwrite(f"./output/{iter}_labeled_img.png", img_before_inference)

        rows_of_word_imgs = map2d(lambda box: np.array(pil_img_before_inference.crop(box)), rows_of_boxes)
        for i, row in enumerate(rows_of_word_imgs):
            for j, word_img in enumerate(row):
                cv2.imwrite(f"./output/{iter}_{i}_{j}_cropped_word.png", word_img)
        rows_of_word_texts = map2d(lambda x: infer_letters(x, conf=0.35), rows_of_word_imgs)
        final_rows = [" ".join(reversed(row)) for row in rows_of_word_texts]
        print("\n".join(final_rows))
        print("---------------------------------------------")

# if __name__ == '__main__':
#     for filename in sorted(os.listdir("./output.bak/words")):
#         img = cv2.cvtColor(cv2.imread(f"./output.bak/words/{filename}"), cv2.COLOR_BGR2RGB)
#         print(infer_letters(img, conf=0.3, convert_names=False))
