from __future__ import annotations
import string
from typing import Generic, TypeVar

import pdfplumber
from handlers import *

I = TypeVar('I')  # input type
O = TypeVar('O')  # output type
K = TypeVar('K')  # intermediate output type


class Pipeline(Generic[I, O]):
    currentHandler: Handler[I, O]

    def __init__(self, currentHandler: Handler[I, O]) -> None:
        super().__init__()
        self.currentHandler = currentHandler

    def addHandler(self, newHandler: Handler[O, K]) -> Pipeline[I, K]:
        return Pipeline(LazyHandler(lambda input: newHandler.process(self.currentHandler.process(input))))

    def execute(self, input: I) -> O:
        return self.currentHandler.process(input)


class PDFSegmentationGenerator:
    def __init__(self, pipline: Pipeline, parallel:bool=False) -> None:
        self.__pipeline = pipline

    def generate(self, fpath:str) -> None:
        with pdfplumber.open(fpath) as pdf:
            for page in pdf.pages[1:]:
                self.__pipeline.execute(page)
                input("continue?")




if __name__ == "__main__":
    viwer = ImageViewer()
    dataset = []
    pipline = Pipeline(ImageBoundingBoxExraction(dataset))\
                        .addHandler(TextBoundingBoxExraction(dataset))\
                        .addHandler(ImageViewerPageSetter(viwer))\
                        .addHandler(LazyHandler(lambda x: dataset)) \
                        .addHandler(MergeBoundingBoxes(1, ClusteringDirection.Horizontal)) \
                        .addHandler(viwer)\
                        .addHandler(LazyHandler(lambda x: dataset.clear()))
    generator = PDFSegmentationGenerator(pipline)
    generator.generate("/home/astroc/Projects/Python/AraGen/books/book4.pdf")
