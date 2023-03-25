import os
import pickle
import random
import shutil
from io import BytesIO
from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
import tqdm
from PIL import ImageFont, Image, ImageDraw
from fontTools.ttLib import TTFont
from skimage.util import random_noise
from torch import multiprocessing

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


def rotate(image, boxes, angle, resize=True):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1.0)
    output_image = cv2.warpAffine(src=image, M=M, dsize=(w, h), flags=cv2.INTER_NEAREST)

    if resize is True:
        output_image = cv2.resize(output_image, (w, h))

    boxes = [np.matmul(M, np.array([
        [x1, x1, x2, x2],
        [y1, y2, y2, y1],
        [1, 1, 1, 1]
    ])).T.reshape(8).tolist() for x1, y1, x2, y2 in boxes]
    return output_image, boxes


def perspective(image, boxes, radius, rand_seed):
    r = random.Random(rand_seed)

    old_boxes = boxes

    x1 = boxes[0][0]
    y1 = boxes[0][1]
    x2 = boxes[0][2]
    y2 = boxes[0][3]
    x3 = boxes[-1][-4]
    y3 = boxes[-1][-3]
    x4 = boxes[-1][-2]
    y4 = boxes[-1][-1]
    top = np.min([y1, y2, y3, y4])
    bot = np.max([y1, y2, y3, y4])
    left = np.min([x1, x2, x3, x4])
    right = np.max([x1, x2, x3, x4])
    padding_top = top
    padding_bot = image.shape[0] - bot
    padding_left = left
    padding_right = image.shape[1] - right

    box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    new_box = [[x + r.randint(-radius, radius), y + r.randint(-radius, radius)] for x, y in box]
    src_points = np.float32(box)
    dst_points = np.float32(new_box)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    boxes = [cv2.perspectiveTransform(np.array(box).reshape((-1, 1, 2)), M).reshape(8).tolist() for box in boxes]

    x1 = boxes[0][0]
    y1 = boxes[0][1]
    x2 = boxes[0][2]
    y2 = boxes[0][3]
    x3 = boxes[-1][-4]
    y3 = boxes[-1][-3]
    x4 = boxes[-1][-2]
    y4 = boxes[-1][-1]
    top = np.min([y1, y2, y3, y4])
    bot = np.max([y1, y2, y3, y4])
    left = np.min([x1, x2, x3, x4])
    right = np.max([x1, x2, x3, x4])
    padding_top = np.max([padding_top, top])
    padding_bot = np.max([padding_bot, image.shape[0] - bot])
    padding_left = np.max([padding_left, left])
    padding_right = np.max([padding_right, image.shape[1] - right])
    shape = (bot - top + padding_top + padding_bot, right - left + padding_left + padding_right)

    # TODO
    if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0 or y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0 or x1 >= image.shape[1] or x2 >= \
            image.shape[1] or x3 >= image.shape[1] or x4 >= image.shape[1] or y1 >= image.shape[0] or y2 >= image.shape[0] or y3 >= image.shape[0] or y4 >= image.shape[0]:
        return image, old_boxes

    img = cv2.warpPerspective(image, M, np.int32(np.ceil(shape[::-1])))
    return img, boxes


def tmp(args):
    DEBUG, i, words, sizes, rotations, fonts, padding_left_range, padding_right_range, padding_top_range, padding_bot_range = args

    word_not_reshaped = random.choice(words)
    size = random.choice(sizes)
    rotation = random.choice(rotations)
    font_filename = random.choice(fonts)
    padding_left = random.randint(*padding_left_range)
    padding_right = random.randint(*padding_right_range)
    padding_top = random.randint(*padding_top_range)
    padding_bot = random.randint(*padding_bot_range)

    perspective_seed = random.random()
    lighting_seed = random.random()

    word = arabic_reshaper.reshape(word_not_reshaped)
    font = ImageFont.truetype(font_filename, size)
    word_box = font.getbbox(word)

    width = word_box[2] - word_box[0] + padding_left + padding_right
    height = word_box[3] - word_box[1] + padding_top + padding_bot
    x1, y1 = (-word_box[0] + padding_left, -word_box[1] + padding_top)

    with TTFont(font_filename, 0, ignoreDecompileErrors=True) as ttf:
        if any([ord(c) not in ttf.getBestCmap() for c in word]):
            return None
        widths = [ttf.getGlyphSet().hMetrics[ttf.getBestCmap()[ord(c)]][0] for c in word]
    widths = [w * (word_box[2] - word_box[0]) / sum(widths) for w in widths][::-1]

    img = Image.new("L", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((x1, y1), word, "black", font=font)

    offset = 0
    boxes = []
    for w in widths:
        word_box = (
            padding_left + offset, padding_top, padding_left + offset + w, padding_top + word_box[3] - word_box[1])
        offset += w
        boxes.append(word_box)

    if DEBUG:
        img.save(f"./output/{i}.png")

    np_img, boxes = rotate(np.array(img) ^ 255, boxes, rotation)
    # TODO
    x1 = boxes[0][0]
    y1 = boxes[0][1]
    x2 = boxes[0][2]
    y2 = boxes[0][3]
    x3 = boxes[-1][-4]
    y3 = boxes[-1][-3]
    x4 = boxes[-1][-2]
    y4 = boxes[-1][-1]
    if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0 or y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0 or x1 >= np_img.shape[1] or x2 >= \
            np_img.shape[1] or x3 >= np_img.shape[1] or x4 >= np_img.shape[1] or y1 >= np_img.shape[0] or y2 >= np_img.shape[
        0] or y3 >= np_img.shape[0] or y4 >= np_img.shape[0]:
        return None
    np_img, boxes = perspective(np_img, boxes, 4, perspective_seed)
    np_img = random_noise(np_img, mode='s&p', amount=0.3)
    np_img = 255 * np_img
    img = Image.fromarray(np_img)
    img = generate_relighted_image(img, lighting_seed)

    np_img = np.array(img)
    np_img ^= 255
    np_bytes = BytesIO()
    np.save(np_bytes, np_img)

    if DEBUG:
        img = Image.fromarray(np_img)
        draw = ImageDraw.Draw(img)
        for word_box in boxes:
            draw.line((word_box[0], word_box[1], word_box[2], word_box[3]), fill="black")
            draw.line((word_box[2], word_box[3], word_box[4], word_box[5]), fill="black")
            draw.line((word_box[4], word_box[5], word_box[6], word_box[7]), fill="black")
            draw.line((word_box[6], word_box[7], word_box[0], word_box[1]), fill="black")
        img.save(f"./output/{i}_rotated.png")

    return {
        "image": np_bytes.getvalue(),
        "boxes": boxes,
        "ground_truth": word_not_reshaped,
    }


def main():
    OUT_DIR = Path("./output")
    DEBUG = False
    REBUILD = True
    N = 10
    TRAIN = 0.7
    VAL = 0.2
    TEST = 0.1
    random.seed(0)

    if REBUILD:
        cleanup(OUT_DIR)

    sizes = list(range(40, 100))
    rotations = list(range(-10, 10))

    padding_left_range = (10, 30)
    padding_right_range = (10, 30)
    padding_top_range = (10, 30)
    padding_bot_range = (10, 30)

    fonts = open("fonts.txt").read().splitlines()
    fonts = filter(lambda file: file.endswith(".ttf"), fonts)
    fonts = list(fonts)

    words = open("words.txt").read().split()
    words = map(remove_punctuations, words)
    words = map(remove_diacritics, words)
    words = list(words)

    unique_letters = list(set("".join(words))) + ['ﻼ', 'ﻻ']
    letter_to_class = {letter: i for i, letter in enumerate(unique_letters)}

    if REBUILD:
        images = []

        with multiprocessing.Pool(os.cpu_count()) as pool:
            iter = [(DEBUG, i, words, sizes, rotations, fonts, padding_left_range, padding_right_range,
                     padding_top_range, padding_bot_range) for i in range(N)]
            tqd = tqdm.tqdm(total=N)
            images = []
            for image in map(tmp, iter):
                images.append(image)
                tqd.update(1)
            images = list(filter(lambda x: x is not None, images))

        with open("./output/data.pickle", "wb") as f:
            pickle.dump(images, f)

    with open("./output/data.pickle", "rb") as f:
        images = pickle.load(f)

        train_dir = OUT_DIR.joinpath("train")
        val_dir = OUT_DIR.joinpath("val")
        test_dir = OUT_DIR.joinpath("test")
        image_dir_name = "images"
        label_dir_name = "labels"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir()
            d.joinpath(label_dir_name).mkdir()
            d.joinpath(image_dir_name).mkdir()

        for i, img_dict in enumerate(images):
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

            filename = f"{label}-{i}"
            Image.fromarray(img).save(dir_.joinpath(image_dir_name, filename + ".png"))
            boxes_str = [
                f"{letter_to_class[c]} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]} {box[7]}\n"
                for box, c in zip(boxes, label[::-1])]
            with open(dir_.joinpath(label_dir_name, filename + ".txt"), "w") as f:
                f.writelines(boxes_str)

        output_str = "names:\n"
        for l in unique_letters:
            output_str += f"  - {l}\n"
        output_str += f"nc: {len(unique_letters)}\n"

        with open("./output/data.yaml", "w") as f:
            f.write(output_str)


if __name__ == "__main__":
    main()
