import random
from PIL import ImageFont, ImageDraw, Image
import enum
import numpy as np
from shapely.geometry import Polygon
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class Rectangle:
    def __init__(self, x, y, width, height) -> None:
        self.__coords = [(x, y), (x + width, y),
                              (x + width, y + height),
                              (x, y + height), (x, y)]
        self.__obj = Polygon(self.__coords)
        self.width = width
        self.height = height
    def intersects(self, rect):
        return self.__obj.intersects(rect.__obj)
    
    def GetRandomLocationForImageToPut(self, dest):
        return (random.randint(0, self.width- dest.width) , random.randint(0, self.height - dest.height) )
    
    def GetRandomLocationForSecondImageToPut(self, dest, old):
        while True:
            x,y  = self.GetRandomLocationForImageToPut(dest)
            newRect=  Rectangle(x, y, self.width, self.height)
            if not old.intersects(newRect):
                return (x, y)
    
    def points(self):
        return self.__coords



class ArabicWordGenerator:
    def __init__(self, fname) -> None:
        self.__words = []
        with open(fname, "r") as file:
            self.__words = file.read().splitlines()

    def generate(self):
        return self.__words[random.randint(0, len(self.__words))]


class FontTypes(enum.Enum):
    TrueFont = 1


class LineKind(enum.Enum):
    Text = 0
    Title = 1


def GetLineKind():
    if random.random() <= 0.1:
        return LineKind.Title
    else:
        return LineKind.Text


def GetNumberOfWords(lineKind):
    if lineKind == LineKind.Title:
        return random.randrange(3, 7)
    elif lineKind == LineKind.Text:
        return random.randrange(8, 10)
    else:
        raise Exception("Unkown type")

FONTFILES = ["~/Downloads/alfont_com_AlFont_com_29LTMakina-Regular.otf"]
NumberOfPages = 10
generator = ArabicWordGenerator(r"/home/astroc/Projects/C#/ImageGenerator/ImageGenerator/ar_reviews_100k.txt")




mmpi = 25.4
dpi = 150
image = Image.new('RGB', (int(210 / mmpi * dpi), int(297 / mmpi * dpi)), (255, 255, 255))
draw = ImageDraw.Draw(image)
pageRect = Rectangle(0,0,image.width, image.height)

imageToAdd = Image.open("/home/astroc/Projects/C#/ImageGenerator/ImageGenerator/w.jpeg", 'r')
imageToAddRect = Rectangle(0,0, imageToAdd.width, imageToAdd.height)

for font in FONTFILES:
    fontSize = random.randint(18, 30)
    regularFont = ImageFont.truetype(font,  fontSize)
    titleFont = ImageFont.truetype(font,  fontSize + 8)
    for pageIndex in tqdm(range(NumberOfPages)):
        model = {}
        draw.rectangle((0, 0, image.width, image.height), (255, 255, 255))
        numberOfImagesToAdd = random.randint(0, 3)
        imageRects = []
        if numberOfImagesToAdd == 1:
            xDestination, yDestination = pageRect.GetRandomLocationForImageToPut(imageToAddRect)
            image.paste(imageToAdd,(xDestination, yDestination))
            imageRects.append(Rectangle(xDestination, yDestination, imageToAdd.width, imageToAdd.height))
        elif numberOfImagesToAdd == 2:
            xDestination1, yDestination1 = pageRect.GetRandomLocationForImageToPut(imageToAddRect)
            image.paste(imageToAdd,(xDestination1, yDestination1))
            oldRect = Rectangle(xDestination1, yDestination1, imageToAdd.width, imageToAdd.height)
            xDestination2, yDestination2 = pageRect.GetRandomLocationForSecondImageToPut(imageToAddRect, oldRect)
            image.paste(imageToAdd,(xDestination2, yDestination2))
            imageRects.append(Rectangle(xDestination1, yDestination1, imageToAdd.width, imageToAdd.height))
            imageRects.append(Rectangle(xDestination2, yDestination2, imageToAdd.width, imageToAdd.height))

        currentY = 0
        for lineIndex in range(40):
            currentX = image.width
            kind = GetLineKind()
            font = regularFont if kind == LineKind.Text else titleFont
            numberOfWords = GetNumberOfWords(kind)
            words = list(map(lambda x: generator.generate(), range(numberOfWords)))
            lineText = ' '.join(words)
            lineTextMeasure = font.getsize(lineText, direction='rtl')
            if currentY + lineTextMeasure[1] > image.height:
                break
            spaceMeasure = font.getsize(" ",  direction='rtl')
            for word in words:
                wordMeasure = font.getsize(word, direction='rtl')
                leftOfWord = currentX - wordMeasure[0]
                currentX -= int(wordMeasure[0] + spaceMeasure[0])
                wordRect = Rectangle(leftOfWord, currentY,wordMeasure[0], wordMeasure[1])
                if leftOfWord < 0 or any(list(map(lambda img: wordRect.intersects(img), imageRects))):
                    continue
                draw.text((leftOfWord, currentY), word, font=font, fill='black')
                draw.line(wordRect.points(), fill='red', width=1 )
            currentY += int(lineTextMeasure[1] + 10)
        image.save(f'./data/{pageIndex}_whatever.png', dpi=(dpi, dpi))
