# coding: utf-8
from model.utils.DataLoader import DataLoader
from random import shuffle
import logging
from os import path
import numpy as np


class MFDataProvider(DataLoader):
    def __init__(self, path_to_train, path_to_save):
        super(MFDataProvider, self).__init__()
        self.load_train(path_to_train, path_to_save)
        self.user_watch_time = self._parse_dict_to_list(self.user_watch_time)

    def batch_generator(self, batch_size):
        """
        :param batch_size: size of mini-batch
        :return: batch_data: a generator for generate batch
        """
        shuffle(self.user_watch_time)
        for i in range(0, len(self.user_watch_time), batch_size):
            batch_data = dict()
            start_idx = i
            end_idx = min(i + batch_size, len(self.user_watch_time))
            batch_data['user_idx'] = self.user_watch_time[start_idx: end_idx, 0]
            batch_data['item_idx'] = self.user_watch_time[start_idx: end_idx, 1]
            batch_data['user_item_score'] = self.user_watch_time[start_idx: end_idx, 2]
            yield batch_data

    def _parse_dict_to_list(self, user_watch_time_dict):
        user_watch_time_list = list()
        for user_id in user_watch_time_dict:
            for item_id in user_watch_time_dict[user_id]:
                if user_watch_time_dict[user_id][item_id] < 30:
                    continue
                user_watch_time_list.append([user_id, item_id, user_watch_time_dict[user_id][item_id]])
        return np.array(user_watch_time_list)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S')
    mf_data_provider = MFDataProvider(path.join("..", "..", "data", "train_data"), path.join("..", "..", "tmp"))
