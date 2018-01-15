# coding: utf-8
from model.utils.DataLoader import DataLoader
from random import shuffle
import logging
from functools import reduce


class MFDataProvider(DataLoader):
    def __init__(self, path_to_train):
        super(MFDataProvider, self).__init__()
        self.load_train(path_to_train)
        self.user_watch_time = reduce(lambda x, y: x + list(map(lambda z: (y[0], ) + z, list(y[1].items()))),
                                      self.user_watch_time.items(), [])

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
