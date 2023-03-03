from __future__ import annotations
from time import sleep
from typing import Any, Generic, TypeVar, Dict, List, Callable, Union
from abc import abstractmethod

import pdfplumber
from pdfplumber.page import Page
from utils import IGNORE_SET, ClusteringDirection, cluster_bb, merge_bb, scale_bb

I = TypeVar('I')  # input type
O = TypeVar('O')  # output type


class Handler(Generic[I, O]):
    @abstractmethod
    def process(self, input: I) -> O:
        pass

class LazyHandler(Handler[I, O]):
    def __init__(self, proc: Callable[[I], O]) -> None:
        super().__init__()
        self.__proc = proc

    def process(self, input: I) -> O:
        return self.__proc(input)




class TextBoundingBoxExraction(Handler[Page,Page ]):
    def __init__(self, dataset, extract_letters:bool= False) -> None:
        super().__init__()
        self.__dataset = dataset
        self.__tolerance:int = -1 if extract_letters else 3
    
    def process(self, input: Page) -> Page:
        self.__dataset.extend(input.extract_words(x_tolerance=self.__tolerance,split_at_punctuation=IGNORE_SET))
        return input


class ImageBoundingBoxExraction(Handler[Page,Page ]):
    def __init__(self, dataset: List[Dict[str,Any]]) -> None:
        super().__init__()
        self.__dataset = dataset
    
    def process(self, input: Page) ->Page: 

            for image in input.images:
                temp = {}
                temp['x0']        = image['x0']
                temp['x1']        = image['x1']
                temp['top']       = image['top']
                temp['bottom']    = image['bottom']
                self.__dataset.append(temp)
            return input

class TableBoundingBoxExraction(Handler[Page,List[Dict[str,Any]] ]):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplemented("Not implemented yet")
            
    def process(self, input: Page) -> List[Dict[str,Any]]:
        #https://medium.com/@karthickrajm/how-to-extract-table-from-pdf-using-python-pdfplumber-a2010b184431
        raise NotImplemented("Not implemented yet")


class MergeBoundingBoxes(Handler[List[Dict[str,Any]],
                                        List[Dict[str,Any]] ]):
    def __init__(self, proximity: float, direction: ClusteringDirection) -> None:
        super().__init__()
        self.__prox:float = proximity
        self.__clstr_dir:ClusteringDirection = direction
    
    def process(self, input: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        grouped = cluster_bb(input,self.__clstr_dir, self.__prox)
        merged  = []
        for bbgrp in grouped:
            if len(bbgrp) > 1:
                partial = bbgrp[0]
                for prop in bbgrp[1:]:
                    if prop['text'] in IGNORE_SET:
                        merged.append(prop)
                        continue
                    new_partial = merge_bb(partial.copy(), prop.copy())
                    partial = new_partial
                merged.append(partial)
            else:
                merged.append(bbgrp[0])
        return merged


class ImageSaver(Handler[Page, bool]):
    def __init__(self, location:str, format:str) -> None:
        super().__init__()
        self.__store_location:str =  location
        self.__im_format: str = format

    def process(self, input: Page) -> bool:
        return input.to_image().annotated.save(f'{self.__store_location}/{input.page_number}.{self.__im_format}')


class ImageViewer(Handler[List[Dict[str,Any]], bool]):
    def __init__(self) -> None:
        super().__init__()
        self._page: Page = None

    def process(self, input: List[Dict[str,Any]]) -> bool:
        im  = self._page.to_image()
        im.draw_rects(input)
        im.show()



class ImageViewerPageSetter(Handler[Page, Page]):
    def __init__(self, viewer) -> None:
        super().__init__()
        self.__viwer:ImageViewer = viewer

    def process(self, input: Page) -> Page:
        self.__viwer._page = input
        return input
         


class ShiftBoundingBoxHandler(Handler[List[Dict[str,Any]], 
                                      List[Dict[str,Any]]]):
    def __init__(self, h_shift:float=0, v_shift:float=0) -> None:
        super().__init__()
        self.__v_shift:float = v_shift
        self.__h_shift:float = h_shift

    def process(self, input: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        for idx,bb in enumerate(input):
            input[idx] = scale_bb(bb, self.__v_shift,  self.__h_shift)
        return input

class ScaleBoundingBoxHandler(Handler[List[Dict[str,Any]], 
                                      List[Dict[str,Any]]]):
    def __init__(self, h_factor:float=1, v_factor:float=1) -> None:
        super().__init__()
        self.__v_factor:float = v_factor
        self.__h_factor:float = h_factor

    def process(self, input: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        for idx,bb in enumerate(input):
            input[idx] = scale_bb(bb, self.__v_factor,  self.__h_factor)
        return input



