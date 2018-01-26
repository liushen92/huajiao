# coding: utf-8
import logging
from os import path
import numpy as np
from .constants import *
from .DataInterface import DataInterface

class TrrDataProvider(DataInterface):
    def __init__(self):
        super(TrrDataProvider, self).__init__()
        self.load_data(path.join(data_dir, "train_data"))

    def _load_tags(self, portrait_file):
        pass
