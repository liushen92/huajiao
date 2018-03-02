from .DataInterface import DataInterface
from os import path
from .constants import *
import logging


class MWDataProvider(DataInterface):
    def __init__(self):
        super(MWDataProvider, self).__init__()
        self.load_data(path.join(data_dir, "train_data"))


class MostWatched(object):
    def __init__(self):
        self.user_num = None
        self.item_num = None
        self.user_anchor_behavior = None

    def fit(self, input_data):
        self.user_num = input_data.user_num
        self.item_num = input_data.anchor_num
        self.user_anchor_behavior = input_data.user_anchor_behavior

    def recommend(self, max_size):
        rec_dict = dict()
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            rec_item_list = sorted(range(self.item_num), key=lambda x: self.user_anchor_behavior[user_id].get(x, [0])[0],
                                   reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(self.user_anchor_behavior[user_id].get(x, [0])[0])) for x in rec_item_list]
        return rec_dict
