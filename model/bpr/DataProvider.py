from os import listdir, path
from utils import Saver
import numpy as np
from random import shuffle
import logging
from multiprocessing import Process, Queue


class DataProvider(object):
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.src_files = [path.join(self.data_path, f)
                          for f in listdir(self.data_path) if path.isfile(path.join(self.data_path, f))]
        self.queue = Queue(1)
        self._user_id_map, self._item_id_map = self._load_mappings()
        self._save_mappings()

    @property
    def user_num(self):
        return len(self._user_id_map)

    @property
    def item_num(self):
        return len(self._item_id_map)

    def batch_generator(self, batch_size):
        """
        :param batch_size: size of mini-batch
        :return: batch_data: a generator for generate batch
        """
        shuffle(self.src_files)
        # start a subprocess to read data, so that latency decreases by reading successive data while training.
        # TO DO: now we start/kill a subprocess in each epoch, actually only one subprocess is needed.
        subprocess = Process(target=self._readlines_from_file, args=(self._local_read,))
        subprocess.start()
        while True:
            paired_data = self.queue.get()
            if paired_data is None:
                break
            for i in range(0, len(paired_data), batch_size):
                batch_data = dict()
                start_idx = i
                end_idx = min(i + batch_size, len(paired_data))
                batch_data['user_idx'] = paired_data[start_idx: end_idx, 0]
                batch_data['pos_item_idx'] = paired_data[start_idx: end_idx, 1]
                batch_data['neg_item_idx'] = paired_data[start_idx: end_idx, 2]
                yield batch_data
        subprocess.terminate()

    def _readlines_from_file(self, func):
        """
        Read data from input files (to memory) sequentially, and put them into a subprocess queue.
        """
        for src_file in self.src_files:
            data = func(src_file)
            self.queue.put(data)
        self.queue.put(None)

    def _local_read_no_action(self, filename):
        with open(filename, 'r') as src:
            return src.readlines()

    def _local_read(self, filename):
        """
        Read data from a local file and parse them.
        :param filename: local file name
        :return: paired_data: a num_samples x 3 np.array that can be used as input for bpr.
        """
        paired_data = list()
        with open(filename, 'r') as src:
            for line in src.readlines():
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    raise ValueError('Encountered badly formatted line in {}'.format(filename))
                paired_data.append(
                    [self._user_id_map[parts[0]], self._item_id_map[parts[1]], self._item_id_map[parts[2]]]
                )
        paired_data = np.array(paired_data)
        return paired_data

    def _hdfs_read(self, filename):
        paired_data = list()
        pass

    def _load_mappings(self):
        """ traverse the whole dataset to get user_id mapping and item_id mapping
        :return:r
            user_id_map: user-id mapping
            item_id_map: item-id mapping
        """
        logging.info("Mapping generation task: Start. (It may take a long time as it may "
                     "need traverse the whole dataset)")
        user_id_map = dict()
        item_id_map = dict()

        if path.isfile(path.join(self.save_path, "user-id-map.txt")) and \
                path.isfile(path.join(self.save_path, "item-id-map.txt")):
            user_id_map = Saver.load_dict(path.join(self.save_path, "user-id-map.txt"))
            item_id_map = Saver.load_dict(path.join(self.save_path, "item-id-map.txt"))
            return user_id_map, item_id_map

        subprocess = Process(target=self._readlines_from_file, args=(self._local_read_no_action,))
        subprocess.start()
        while True:
            data = self.queue.get()
            if data is None:
                break
            for line in data:
                parts = line.strip().split("\t")
                user_name, pos_item_name, neg_item_name = parts

                if user_id_map.get(user_name) is None:
                    user_id_map[user_name] = len(user_id_map)
                if item_id_map.get(pos_item_name) is None:
                    item_id_map[pos_item_name] = len(item_id_map)
                if item_id_map.get(neg_item_name) is None:
                    item_id_map[neg_item_name] = len(item_id_map)
        subprocess.terminate()

        logging.info("Mapping generation task: Done.")
        return user_id_map, item_id_map

    def _save_mappings(self):
        Saver.save_dict(self._user_id_map, path.join(self.save_path, "user-id-map.txt"))
        Saver.save_dict(self._item_id_map, path.join(self.save_path, "item-id-map.txt"))


if __name__ == "__main__":
    pass
