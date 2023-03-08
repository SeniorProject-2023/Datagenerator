from __future__ import annotations

import shutil
import math

from pathlib import Path
from typing import Union, List, Dict, Any

from functools import partial, reduce
from itertools import product

import itertools
import string
import enum


import pdfplumber
import tqdm as tqdm


IGNORE_SET:str = string.punctuation+"*':،.؛!؟."
classes = ['image', 'word']



class ClusteringDirection(enum.Enum):
    Veritical  = 1
    Horizontal = 2

def cluster_bb(bb: List[Dict[str, Union[float, str]]],
               direction: ClusteringDirection,
               prox: float = 0,
               ) -> List[List[Dict[str, Union[float, str]]]]:
    """
    @bb: list of bounding boxes,
    @direction: direction to cluster along
    @prox: the proximity threshold value to cluster neighbouring bounding by
    """
    _key = None
    if direction == ClusteringDirection.Veritical:
        _key = VerticalClusteringKey(prox)
    else:
        _key = HorizontalClusteringKey(prox)
    return [list(g) for _, g in itertools.groupby(bb, key=_key)]


class HorizontalClusteringKey(object):
    def __init__(self, diff):
        self.diff, self.flag, self.prev = diff, [0, 1], None

    def __call__(self, elem):
        if self.prev and abs(self.prev['x1'] - elem['x0']) > self.diff:
            self.flag = self.flag[::-1]
        self.prev = elem
        return self.flag[0]


class VerticalClusteringKey(object):
    def __init__(self, diff):
        self.diff, self.flag, self.prev = diff, [0, 1], None

    def __call__(self, elem):
        if self.prev and abs(self.prev['bottom'] - elem['top']) > self.diff:
            self.flag = self.flag[::-1]
        self.prev = elem
        return self.flag[0]



def merge_bb(a, b):
    res = {}
    res['x0'] = min(a['x0'], b['x0'])
    res['x1'] = max(a['x1'], b['x1'])
    res['top'] = max(a['top'], b['top'])
    res['bottom'] = max(a['bottom'], b['bottom'])
    res['text'] = a['text'] + b['text']
    return res



def distance(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def lerp(p, q, t):
    return ((1 - t) * p[0] + t * q[0], (1 - t) * p[1] + t * q[1])


def resample_polygon(polygon, k):
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    arc_length = sum(distance(p, q) for p, q in edges) / k
    result = [polygon[0]]
    t = 0
    for p, q in edges:
        d_t = distance(p, q) / arc_length
        while t + d_t >= len(result) < k:
            v = lerp(p, q, (len(result) - t) / d_t)
            result.append(v)
        t += d_t
    return result


def id_generator():
    inital = 0
    while True:
        yield inital
        inital += 1


def to_coco(bbx):
    global classes
    poly = resample_polygon(
        list(product([bbx['x0'], bbx['x1']], [bbx['top'], bbx['bottom']])), 10)
    poly = [p for x in poly for p in x]
    return {
        "category_id": classes.index(bbx['label']),
        "segmentation": [poly],
        "area": (bbx['x1'] - bbx['x0']) * (bbx['bottom'] - bbx['top']),
        "bbox": [bbx['x0'],  bbx['top'], bbx['x1'] - bbx['x0'], bbx['bottom'] - bbx['top']],
        "iscrowd": 0
    }


def to_yolo():
    pass


def merge_close_boxes(grouped: List[List[Dict[str, Union[float, str]]]]) -> List[Dict[str, Any]]:
    merged = []
    for bbgrp in grouped:
        if len(bbgrp) > 1:
            partial = bbgrp[0]
            for prop in bbgrp[1:]:
                if 'text' not in prop.keys() or prop['text'] in IGNORE_SET:
                    merged.append(prop)
                    continue
                partial = merge_bb(partial, prop)
            merged.append(partial)
        else:
            merged.append(bbgrp[0])
    return merged


def process(fpath: str, extract_letters: bool = False, proximity: float = 1,
            direction: ClusteringDirection = ClusteringDirection.Horizontal) -> None:
    pdf = pdfplumber.open(fpath)
    pages = pdf.pages[7:]

    images_per_page = map(lambda page: page.images, pages)

    words_per_page = map(
        lambda page: page.extract_words(
            x_tolerance=(-1 if extract_letters else 3), split_at_punctuation=IGNORE_SET),
        pages
    )

    boxes_per_page = map(lambda x: x[0] + x[1], zip(images_per_page, words_per_page))
    if not extract_letters:
        grouped_boxes_per_page = map(lambda words_and_images: 
                                 cluster_bb(words_and_images, direction, proximity),boxes_per_page)
    else:
        grouped_boxes_per_page =  boxes_per_page

    label_bboxes = lambda _bbx:  {**_bbx, **{'label': 'word' if 'text' in _bbx.keys() else 'image'}}
    merged_boxes_per_page = map(merge_close_boxes, grouped_boxes_per_page)

    dataset = []
    for i, page, bboxes_per_page in tqdm.tqdm(zip(range(len(pages)), pages, merged_boxes_per_page)):
        fname  = fpath.split("/")[-1].split(".")[0]
        fname = f"./{fname}/{i}.png"
        record = {}
        record["file_name"] = fname
        record["height"] = page.height
        record["width"] = page.width
        labeled_boxes = map(label_bboxes, bboxes_per_page)
        converted = map(to_coco, labeled_boxes)
        record["annotations"] = list(converted)
        page.to_image().save(fname)
        dataset.append(record)
    return dataset


def cleanup():
    if Path("./output").exists():
        shutil.rmtree("./output")
    Path("./output").mkdir(exist_ok=True)


if __name__ == "__main__":
    cleanup()
    process("/home/astroc/Projects/Python/AraGen/books/book4.pdf")
