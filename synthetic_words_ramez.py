import argparse
import math
import pickle
import random
import shutil
import sys
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
import tqdm
from PIL import ImageFont, Image, ImageDraw
from fontTools.ttLib import TTFont
from skimage.util import random_noise

from probe_relighting.generate_images import generate_relighted_image
from punctuation import remove_punctuations, remove_diacritics

'''
split whole lines into a single word per line (requires moreutils)

cat words.txt | xargs -n1 | sponge words.txt


get a list of all installed fonts with Arabic support (requires fontconfig)

fc-list -f '"%{file}",\n' :lang=ar > fonts.txt
'''


def cleanup(out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)


def transform_box(boxes, M):
    boxes = map(lambda box: np.array(box).reshape(-1, 2).T, boxes)
    boxes = map(lambda box: np.vstack((box, np.ones(box.shape[1]))), boxes)
    boxes = map(lambda box: M @ box, boxes)
    boxes = map(lambda box: box.T.reshape(-1).tolist(), boxes)
    boxes = list(boxes)
    return boxes


def tighten_boxes(img, boxes):
    out = []
    for box in boxes:
        cpy = np.copy(img)
        contour_mask = np.zeros_like(cpy)
        contour_mask = cv2.fillPoly(contour_mask, [np.array(box).reshape(-1, 2).astype("int32")], 255)
        # mask = cv2.bitwise_not(contour_mask)
        result = cv2.bitwise_and(cpy, cpy, mask=contour_mask)
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_contour = np.vstack(contours)
        merged_contour = np.squeeze(merged_contour)
        hull = cv2.convexHull(merged_contour)
        hull = np.squeeze(hull)
        out.append(hull.reshape(-1).tolist())

    return out


def rotate(image, boxes, angle_in_degrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle_in_degrees, 1)

    rad = math.radians(angle_in_degrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = math.ceil((h * abs(sin)) + (w * abs(cos))) + 1
    b_h = math.ceil((h * abs(cos)) + (w * abs(sin))) + 1

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h))
    boxes = transform_box(boxes, rot)
    return outImg, boxes


def random_perspective(image, boxes, radius, rand_seed):
    r = random.Random(rand_seed)

    h, w = image.shape[:2]

    x1, y1, x2, y2 = h * 0.2, w * 0.2, h * 0.8, w * 0.2
    x3, y3, x4, y4 = h * 0.8, w * 0.8, h * 0.2, w * 0.8
    box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    new_box = [[x + r.randint(-radius, radius), y + r.randint(-radius, radius)] for x, y in box]
    src_points = np.float32(box)
    dst_points = np.float32(new_box)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    boxes_ = [cv2.perspectiveTransform(np.array(box).reshape((-1, 1, 2)), M).reshape(-1).tolist() for box in boxes]

    boxes_ = np.array([j for sub in boxes_ for j in sub])
    xs = boxes_[::2]
    ys = boxes_[1::2]
    top = np.min(ys)
    bot = np.max(ys)
    left = np.min(xs)
    right = np.max(xs)
    shape = (bot - top + 2, right - left + 2)

    trans = np.array([
        [1, 0, -left + 1],
        [0, 1, -top + 1],
        [0, 0, 1]
    ])

    img = cv2.warpPerspective(image, trans @ M, np.int32(np.ceil(shape[::-1])))
    boxes = [cv2.perspectiveTransform(np.array(box).reshape((-1, 1, 2)), trans @ M).reshape(-1).tolist() for box in
             boxes]

    return img, boxes


def pad(img, boxes, padding_left, padding_right, padding_top, padding_bot):
    trans = np.float32([
        [1, 0, padding_left],
        [0, 1, padding_top],
    ])

    new_height, new_width = img.shape
    new_height += padding_top + padding_bot
    new_width += padding_right + padding_left
    img = cv2.warpAffine(img, trans, (new_width, new_height))
    boxes = transform_box(boxes, trans)

    return img, boxes


def resize(img, boxes, width, height):
    trans = np.float32([
        [width / img.shape[1], 0, 0],
        [0, height / img.shape[0], 0],
    ])

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    boxes = transform_box(boxes, trans)

    return img, boxes


def generate_image_sentences(DEBUG, i, words, sizes, rotations, fonts, padding_left_range, padding_right_range,
                             padding_top_range,
                             padding_bot_range, perspective_radius, quality_coefficient_range, noise_amount_range,
                             no_mod_probability):
    word_not_reshaped = random.choice(words)
    size = random.choice(sizes)
    rotation = random.choice(rotations)
    font_filename = random.choice(fonts)
    padding_left = random.randint(*padding_left_range)
    padding_right = random.randint(*padding_right_range)
    padding_top = random.randint(*padding_top_range)
    padding_bot = random.randint(*padding_bot_range)
    quality_coefficient = random.randint(*quality_coefficient_range)
    noise_amount = random.uniform(*noise_amount_range)

    perspective_seed = random.random()
    lighting_seed = random.random()

    word = arabic_reshaper.reshape(word_not_reshaped)
    font = ImageFont.truetype(font_filename, size)
    word_box = font.getbbox(word)

    width = word_box[2] - word_box[0]
    height = word_box[3] - word_box[1]
    x1, y1 = (-word_box[0], -word_box[1])

    with TTFont(font_filename, 0, ignoreDecompileErrors=True) as ttf:
        if any([ord(c) not in ttf.getBestCmap() for c in word]):
            return generate_image_sentences(DEBUG, i, words, sizes, rotations, fonts, padding_left_range,
                                            padding_right_range, padding_top_range,
                                            padding_bot_range, perspective_radius, quality_coefficient_range,
                                            noise_amount_range, no_mod_probability)
        widths = [ttf.getGlyphSet().hMetrics[ttf.getBestCmap()[ord(c)]][0] for c in word]
    widths = [w * width / sum(widths) for w in widths][::-1]

    img = Image.new("L", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((x1, y1), word, "black", font=font)

    offset = 0
    boxes = []
    for w in widths:
        x1, y1, x2, y2 = (offset, 0, offset + w, height)
        x1, y1, x2, y2, x3, y3, x4, y4 = x1, y1, x1, y2, x2, y2, x2, y1  # 2 points to 4 points
        offset += w
        boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

    if DEBUG:
        img.save(f"./output/{i}.png")

    np_img = np.array(img)
    np_img ^= 255
    boxes = tighten_boxes(np_img, boxes)
    np_img ^= 255
    if np.random.choice([True, False], 1, p=[1 - no_mod_probability, no_mod_probability]):
        # quality reduction
        # img = img.resize((img.size[0] // 3, img.size[1] // quality_coefficient)).resize(img.size)

        np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
        _, np_img = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        np_img ^= 255
        np_img, boxes = rotate(np_img, boxes, rotation)
        np_img, boxes = resize(np_img, boxes, 128 // 2, 96 // 2)
        np_img, boxes = random_perspective(np_img, boxes, perspective_radius, perspective_seed)
        np_img, boxes = pad(np_img, boxes, padding_left, padding_right, padding_top, padding_bot)
        np_img = random_noise(np_img, mode='s&p', amount=noise_amount)
        np_img = np.uint8(255 * (np_img / np_img.max()))
        np_img ^= 255

        # np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
        _, np_img = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = Image.fromarray(np_img)
        img = generate_relighted_image(img, lighting_seed)
        np_img = np.array(img)

    np_img, boxes = resize(np_img, boxes, 128, 96)

    np_bytes = BytesIO()
    np.save(np_bytes, np_img)

    if DEBUG:
        img = Image.fromarray(np_img)
        draw = ImageDraw.Draw(img)
        for word_box in boxes:
            for even_idx in range(0, len(word_box), 2):
                draw.line((word_box[even_idx % len(word_box)], word_box[(even_idx + 1) % len(word_box)],
                           word_box[(even_idx + 2) % len(word_box)], word_box[(even_idx + 3) % len(word_box)]),
                          fill="black")
        img.save(f"./output/{i}_rotated.png")

    return {
        "image": np_bytes.getvalue(),
        "boxes": boxes,
        "ground_truth": word_not_reshaped,
    }


def pickle_to_yolo(pickle_path, unique_letters, letter_to_class, OUT_DIR, TRAIN, VAL):
    with open(pickle_path, "rb") as f:
        images = pickle.load(f)
        N = len(images)

        train_dir = OUT_DIR.joinpath("train")
        val_dir = OUT_DIR.joinpath("val")
        test_dir = OUT_DIR.joinpath("test")
        image_dir_name = "images"
        label_dir_name = "labels"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir()
            d.joinpath(label_dir_name).mkdir()
            d.joinpath(image_dir_name).mkdir()

        for i, img_dict in tqdm.tqdm(enumerate(images)):
            if i / N < TRAIN:
                dir_ = train_dir
            elif i / N < TRAIN + VAL:
                dir_ = val_dir
            else:
                dir_ = test_dir

            img = np.load(BytesIO(img_dict["image"]), allow_pickle=True)
            boxes = img_dict["boxes"]
            label = img_dict["ground_truth"]
            reshaped_label = arabic_reshaper.reshape(label)

            lam_alef_indices = [i for i, ltr in enumerate(reshaped_label) if ltr == 'ﻼ' or ltr == 'ﻻ']
            for idx in lam_alef_indices:
                label = label[:idx] + reshaped_label[idx] + label[idx + 2:]

            if len(label) != len(reshaped_label):
                continue

            # normalization
            for word_box in boxes:
                for k in range(len(word_box)):
                    word_box[k] /= img.shape[(k + 1) % 2]

            # add more points inbetween
            for iter in range(2):
                new_boxes = []
                for word_box in boxes:
                    new_box = []
                    for even_idx in range(0, len(word_box), 2):
                        new_box.append(word_box[even_idx % len(word_box)])
                        new_box.append(word_box[(even_idx + 1) % len(word_box)])
                        new_box.append(
                            (word_box[even_idx % len(word_box)] + word_box[(even_idx + 2) % len(word_box)]) / 2)
                        new_box.append(
                            (word_box[(even_idx + 1) % len(word_box)] + word_box[(even_idx + 3) % len(word_box)]) / 2)
                    new_boxes.append(new_box)
                boxes = new_boxes

            filename = f"{label}-{i}"
            Image.fromarray(img).save(dir_.joinpath(image_dir_name, filename + ".png"))
            box_to_string = lambda b: " ".join([str(b[i]) for i in range(len(b))])
            boxes_str = [
                f"{letter_to_class[c]} {box_to_string(box)}\n"
                for box, c in zip(boxes, label[::-1])]
            with open(dir_.joinpath(label_dir_name, filename + ".txt"), "w") as f:
                f.writelines(boxes_str)

        output_str = "names:\n"
        for l in unique_letters:
            output_str += f"  - {l}\n"
        output_str += f"nc: {len(unique_letters)}\n"

        with open("./output/data.yaml", "w") as f:
            f.write(output_str)


def main():
    OUT_DIR = Path("./output")
    FONTS_PATH = Path("fonts.txt")
    WORDS_PATH = Path("words.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--rebuild", action="store_true", default=True)
    parser.add_argument("--count", default=10, type=int)
    parser.add_argument("--start", default=0, type=int)
    args = parser.parse_args()

    DEBUG = args.debug
    REBUILD = args.rebuild
    N = args.count
    START = args.start
    TRAIN = 0.7
    VAL = 0.2
    TEST = 0.1

    if REBUILD:
        cleanup(OUT_DIR)

    no_mod_probability = 0.25

    sizes = list(range(40, 100))
    rotations = list(range(-2, 2))
    perspective_radius = 4

    padding_left_range = (0, 4)
    padding_right_range = (0, 4)
    padding_top_range = (0, 4)
    padding_bot_range = (0, 4)

    quality_coefficient_range = (3, 4)
    noise_amount_range = (0, 0.01)

    fonts = open(FONTS_PATH).read().splitlines()
    fonts = filter(lambda file: file.endswith(".ttf"), fonts)
    fonts = list(fonts)

    words = open(WORDS_PATH).read().split()
    words = map(remove_punctuations, words)
    words = map(remove_diacritics, words)
    words = list(words)

    unique_letters = list(OrderedDict.fromkeys("".join(words + ['ﻼ', 'ﻻ'])).keys())
    letter_to_class = {letter: i for i, letter in enumerate(unique_letters)}

    if REBUILD:
        images = [
            generate_image_sentences(DEBUG, i, words, sizes, rotations, fonts, padding_left_range, padding_right_range,
                                     padding_top_range, padding_bot_range, perspective_radius,
                                     quality_coefficient_range, noise_amount_range,
                                     no_mod_probability) for i in tqdm.tqdm(range(START, START + N))]
        images = list(filter(lambda x: x is not None, images))

        with open(OUT_DIR / "data.pickle", "wb") as f:
            pickle.dump(images, f)

    pickle_to_yolo(OUT_DIR / "data.pickle", unique_letters, letter_to_class, OUT_DIR, TRAIN, VAL)


if __name__ == "__main__":
    main()
