# coding: utf-8
from utils import *
import os
import logging


class DataLoader(object):
    def __init__(self):
        self._user_watch_time = dict()
        self.user2id = None
        self.item2id = None
        self.id2item = None
        self.id2user = None

    @property
    def user_num(self):
        return len(self.user2id)

    @property
    def item_num(self):
        return len(self.item2id)

    @property
    def user_watch_time(self):
        return self._user_watch_time

    @user_watch_time.setter
    def user_watch_time(self, val):
        self._user_watch_time = val

    def load_train(self, path_to_data, path_to_save_dict=None):
        logging.info("Start reading data.")
        with open(path_to_data) as f:
            i = 0
            for l in f.readlines():
                user_md5, item_md5, watch_time = l.strip().split("\t")
                watch_time = float(watch_time)
                user_id = self._get_id(user_md5, self.user2id)
                item_id = self._get_id(item_md5, self.item2id)
                logging.debug("line {}: Parse triplet ({}, {}, {})".format(i + 1, user_id, item_id, watch_time))
                i += 1
                if self._user_watch_time.get(user_id) is None:
                    self._user_watch_time[user_id] = dict()
                self._user_watch_time[user_id][item_id] = watch_time
        if path_to_save_dict is not None:
            self.save_map(path_to_save_dict)
        logging.info("Data loaded.")

    def save_map(self, path_to_dir):
        save_dict(self.user2id, os.path.join(path_to_dir, "user_map.txt"))
        save_dict(self.item2id, os.path.join(path_to_dir, "item_map.txt"))

    def save_rec_dict(self, recommend_dict, path_to_rec_file):
        if self.id2item is None or self.id2user is None:
            self.id2item = reverse_dict(self.item2id)
            self.id2user = reverse_dict(self.user2id)

        with open(path_to_rec_file, 'w') as f:
            for user_id in recommend_dict:
                f.write(self.id2user[user_id] + "\t")
                for item_id, score in recommend_dict[user_id]:
                    f.write(self.id2item[item_id] + ":" + score + " ")
                f.write("\n")

    def generate_test_data(self, path_to_dict, path_to_raw_test, path_to_test_file):
        if self.item2id is None or self.user2id is None:
            self.item2id = load_dict(os.path.join(path_to_dict, "item_map.txt"))
            self.user2id = load_dict(os.path.join(path_to_dict, "user_map.txt"))

        logging.info("Start reading data.")
        user_watch_time = dict()
        with open(path_to_raw_test) as f:
            for l in f.readlines():
                user_md5, item_md5, watch_time = l.strip().split("\t")
                watch_time = float(watch_time)
                if self.user2id.get(user_md5) is None or self.item2id.get(item_md5) is None:
                    continue
                if user_watch_time.get(self.user2id.get(user_md5)) is None:
                    user_watch_time[self.user2id.get(user_md5)] = dict()
                user_watch_time[self.user2id.get(user_md5)][self.item2id.get(item_md5)] = watch_time

        if self.id2item is None or self.id2user is None:
            self.id2item = reverse_dict(self.item2id)
            self.id2user = reverse_dict(self.user2id)

        with open(path_to_test_file, 'w') as f:
            for user_id in user_watch_time:
                f.write(self.id2user[user_id] + "\t")
                test_list = sorted(list(user_watch_time[user_id].keys()),
                                   key=lambda x: user_watch_time[user_id][x],
                                   reverse=True)
                for item_id in test_list:
                    f.write(self.id2item[item_id] + ":" + str(user_watch_time[user_id][item_id]) + " ")
                f.write("\n")

        logging.info("Data loaded.")

    def _get_id(self, md5, map_dict):
        if map_dict is None:
            map_dict = dict()
        _id = map_dict.get(md5)
        if _id is None:
            _id = len(map_dict)
            map_dict[md5] = _id
        return _id


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.generate_test_data(".\\tmp\\", ".\\data\\test_data", ".\\data\\test")
