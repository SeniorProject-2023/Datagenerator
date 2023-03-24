import pickle
import random
import re
import shutil
import string
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

'''
split whole lines into a single word per line (requires moreutils)

cat words.txt | xargs -n1 | sponge words.txt


get a list of all installed fonts with Arabic support (requires fontconfig)

fc-list -f '"%{file}",\n' :lang=ar > fonts.txt
'''

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)


def cleanup(out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


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


def perspective(image, box, boxes, radius, rand_seed):
    r = random.Random(rand_seed)
    x0, x1, y0, y1 = box
    box = [[x0, x1], [x0, y1], [x1, y1], [x1, y0]]
    new_box = [[x + r.randint(-radius, radius), y + r.randint(-radius, radius)] for x, y in box]
    src_points = np.float32(box)
    dst_points = np.float32(new_box)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    boxes = [np.matmul(M, np.array([
        [x1, x2, x3, x4],
        [y1, y2, y3, y4],
        [1, 1, 1, 1]
    ]))[:2, :].T.reshape(8).tolist() for x1, y1, x2, y2, x3, y3, x4, y4 in boxes]
    img = cv2.warpPerspective(image, M, image.shape[::-1])
    return img, boxes, new_box


if __name__ == "__main__":
    OUT_DIR = Path("./output")
    DEBUG = True
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
    lighting_seed = random.random()
    perspective_seed = random.random()

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

    unique_letters = list(set("".join(words)))
    letter_to_class = {letter: i for i, letter in enumerate(unique_letters)}

    if REBUILD:
        images = []
        tqd = tqdm.tqdm(total=N)
        while len(images) < N:
            tqd.update()

            word_not_reshaped = random.choice(words)
            size = random.choice(sizes)
            rotation = random.choice(rotations)
            font_filename = random.choice(fonts)
            padding_left = random.randint(*padding_left_range)
            padding_right = random.randint(*padding_right_range)
            padding_top = random.randint(*padding_top_range)
            padding_bot = random.randint(*padding_bot_range)

            word = arabic_reshaper.reshape(word_not_reshaped)
            font = ImageFont.truetype(font_filename, size)
            box = font.getbbox(word)

            width = box[2] - box[0] + padding_left + padding_right
            height = box[3] - box[1] + padding_top + padding_bot
            x1, y1 = (-box[0] + padding_left, -box[1] + padding_top)

            with TTFont(font_filename, 0, ignoreDecompileErrors=True) as ttf:
                if any([ord(c) not in ttf.getBestCmap() for c in word]):
                    continue
                widths = [ttf.getGlyphSet().hMetrics[ttf.getBestCmap()[ord(c)]][0] for c in word]
            widths = [w * (box[2] - box[0]) / sum(widths) for w in widths][::-1]

            img = Image.new("L", (width, height), "white")
            draw = ImageDraw.Draw(img)
            draw.text((x1, y1), word, "black", font=font)

            offset = 0
            boxes = []
            for w in widths:
                box = (padding_left + offset, padding_top, padding_left + offset + w, padding_top + box[3] - box[1])
                offset += w
                boxes.append(box)

            if DEBUG:
                img.save(f"./output/{len(images)}.png")

            np_img, boxes = rotate(np.array(img) ^ 255, boxes, rotation)
            # np_img, boxes, box = perspective(np_img, box, boxes, 4, perspective_seed)
            np_img = random_noise(np_img, mode='s&p', amount=0.3)
            np_img = np.array(255 * np_img, dtype='uint8')
            img = Image.fromarray(np_img)
            img = generate_relighted_image(img, lighting_seed)

            np_img = np.array(img)
            np_img ^= 255
            np_bytes = BytesIO()
            np.save(np_bytes, np_img)
            images.append({
                "image": np_bytes.getvalue(),
                "boxes": boxes,
                "ground_truth": word_not_reshaped
            })

            if DEBUG:
                img = Image.fromarray(np_img)
                draw = ImageDraw.Draw(img)
                for box in boxes:
                    draw.line((box[0], box[1], box[2], box[3]), fill="black")
                    draw.line((box[2], box[3], box[4], box[5]), fill="black")
                    draw.line((box[4], box[5], box[6], box[7]), fill="black")
                    draw.line((box[6], box[7], box[0], box[1]), fill="black")
                img.save(f"./output/{len(images) - 1}_rotated.png")

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

            # normalization
            for box in boxes:
                for k in range(len(box)):
                    box[k] /= img.shape[(k + 1) % 2]

            filename = f"{i}"
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
