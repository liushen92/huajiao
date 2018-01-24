# coding: utf-8
from model.utils.DataInterface import DataInterface
from model.utils.constants import *
import logging
from os import path
import numpy as np


class MFDataProvider(DataInterface):
    def __init__(self):
        super(MFDataProvider, self).__init__()
        self.load_data(path.join(data_dir, "train_data"))
        self.user_watch_time = self._parse_dict_to_nparray(self.user_anchor_behavior)

    def batch_generator(self, batch_size):
        """
        :param batch_size: size of mini-batch
        :return: batch_data: a generator for generate batch
        """
        # shuffle(self.user_watch_time)
        for i in range(0, len(self.user_watch_time), batch_size):
            batch_data = dict()
            start_idx = i
            end_idx = min(i + batch_size, len(self.user_watch_time))
            batch_data['user_idx'] = self.user_watch_time[start_idx: end_idx, 0].astype(np.int32)
            batch_data['item_idx'] = self.user_watch_time[start_idx: end_idx, 1].astype(np.int32)
            batch_data['user_item_score'] = self.user_watch_time[start_idx: end_idx, 2]
            yield batch_data

    def _parse_dict_to_nparray(self, user_anchor_behavior):
        user_watch_time_list = list()
        for user_id in user_anchor_behavior:
            for anchor_id in user_anchor_behavior[user_id]:
                user_watch_time_list.append([user_id, anchor_id,
                                             self._convert_watch_time_to_score(
                                                 user_anchor_behavior[user_id][anchor_id][0])])
        return np.array(user_watch_time_list)

    def _convert_watch_time_to_score(self, watch_time):
        return np.log10(watch_time + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S')
    mf_data_provider = MFDataProvider()
