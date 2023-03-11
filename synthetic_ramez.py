import pickle
import random
import re
import shutil
import string
from pathlib import Path

from fontTools.ttLib import TTFont

from PIL import ImageFont, Image, ImageDraw
import arabic_reshaper
import tqdm

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


def cleanup():
    if Path("./output").exists():
        shutil.rmtree("./output")
    Path("./output").mkdir(exist_ok=True)


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


if __name__ == "__main__":
    random.seed(0)

    cleanup()

    sizes = list(range(40, 100))

    padding_left_range = (10, 100)
    padding_right_range = (10, 100)
    padding_top_range = (10, 100)
    padding_bot_range = (10, 100)

    fonts = open("fonts.txt").read().splitlines()
    fonts = filter(lambda file: file.endswith(".ttf"), fonts)
    fonts = list(fonts)

    words = open("words.txt").read().split()
    words = map(remove_punctuations, words)
    words = map(remove_diacritics, words)
    words = list(words)

    images = []
    tqd = tqdm.tqdm(total=5000)
    while len(images) < 5000:
        tqd.update()

        word_not_reshaped = random.choice(words)
        size = random.choice(sizes)
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
        sum_ = sum(widths)
        widths = [w * (box[2] - box[0]) / sum(widths) for w in widths][::-1]

        img = Image.new("L", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((x1, y1), word, "black", font=font)

        offset = 0
        boxes = []
        for w in widths:
            box = (padding_left + offset, padding_top, padding_left + offset + w, padding_top + box[3] - box[1])
            # draw.rectangle(box, outline="black")
            offset += w
            boxes.append(box)

        # img.save(f"./output/{len(images)}.png")
        images.append({
            "image": (img.mode, img.size, img.tobytes()),
            "boxes": boxes,
            "ground_truth": word_not_reshaped
        })

    with open("./output/data.pickle", "wb") as f:
        pickle.dump(images, f)

    # with open("./output/data.pickle", "rb") as f:
    #     images = pickle.load(f)
    #     Image.frombytes(*images[0]["image"]).save("./output/testtttttttttt.png")
    #     print(images[0]["boxes"])
    #     print(images[0]["ground_truth"][0])
