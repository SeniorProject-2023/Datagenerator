import enum
import pdfplumber
import string
import itertools
from typing import Dict, List, Union


IGNORE_SET:str = string.punctuation+"*':،.؛!؟."



class ClusteringDirection(enum.Enum):
    Veritical  = 1
    Horizontal = 2

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


def shift_bb(bb: Dict[str, Union[float, str]], h_shft: float = 0, v_shift: float = 0):
    """
    @bb : the bouding box to scale,
    @h_shft : the horizontal shift value in pixels
    @v_shft : the vertical shift value in pixels
    """
    bb['x0'] += h_shft
    bb['x1'] += h_shft
    bb['top'] += v_shift
    bb['bottom'] += v_shift
    return bb


def scale_bb(bb: Dict[str, Union[float, str]], v_factor: float = 1, h_factor: float = 1):
    """
    @bb : the bouding box to scale,
    @v_factor : the scaling factor along the vertical direction
    @h_factor : the scaling factor along the horizontal direction
    """
    if v_factor == 0 or h_factor == 0:
        raise Exception("Scale factor cannot be 0.")
    bb['x0'] *= h_factor
    bb['x1'] *= h_factor
    bb['top'] *= v_factor
    bb['bottom'] *= v_factor
    return bb


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


temp = None
image = None

with pdfplumber.open("/home/astroc/Projects/Python/AraGen/books/book4.pdf") as pdf:
    first_page = pdf.pages[42]
    im = first_page.to_image()
    temp = first_page.extract_words(
        x_tolerance=1, split_at_punctuation=IGNORE_SET)
    image = im
