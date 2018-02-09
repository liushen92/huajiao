# coding: utf-8
import numpy as np
import scipy.sparse as sp
import logging
from .DataInterface import DataInterface
from joblib import Parallel, delayed
import multiprocessing
from os import path
from .constants import *


class CFDataProvider(DataInterface):
    def __init__(self):
        super(CFDataProvider, self).__init__()
        self.item_num = self.anchor_num
        self.ui_matrix_csc = self._parse_dict_to_matrix(self.user_anchor_behavior)
        self.ui_matrix_csr = self.ui_matrix_csc.tocsr()

    def _parse_dict_to_matrix(self, user_anchor_behavior):
        ui_matrix = sp.dok_matrix((self.user_num, self.anchor_num), dtype=np.float64)
        for user_id in user_anchor_behavior:
            for item_id in user_anchor_behavior[user_id]:
                ui_matrix[user_id, item_id] = self._convert_watch_time_to_score(user_anchor_behavior[user_id][item_id][0])

        ui_matrix = ui_matrix.tocsc()
        return ui_matrix

    def _convert_watch_time_to_score(self, watch_time):
        if watch_time < 30:
            return 0
        return np.log10(watch_time + 1)


class CF(object):
    def __init__(self):
        self.user_num = None
        self.item_num = None
        self.ui_matrix_csc = None
        self.ui_matrix_csr = None

        self._item_sim_matrix = None

        self.sim_neighbor_size = None
        self.recommend_neighbor_num = None
        self.recommend_neighbor_size = None
        self.recommend_size = None
        self.user_liked_item = dict()

    @property
    def item_sim_matrix(self):
        if self._item_sim_matrix is None:
            row = np.array([np.ones(self.sim_neighbor_size, dtype=np.int32) * k for k in range(self.item_num)]).reshape(self.item_num * self.sim_neighbor_size)
            col = np.memmap(path.join(tmp_dir, "item_sim_matrix_id"), dtype=np.int32,
                            shape=(self.item_num, self.sim_neighbor_size), mode="r").reshape(self.item_num * self.sim_neighbor_size)
            data = np.memmap(path.join(tmp_dir, "item_sim_matrix_val"), dtype=np.float64,
                             shape=(self.item_num, self.sim_neighbor_size), mode="r").reshape(self.item_num * self.sim_neighbor_size)
            self._item_sim_matrix = sp.csr_matrix((data, (row, col)), shape=(self.item_num, self.item_num))

        return self._item_sim_matrix

    def _parse_config(self, configs):
        self.sim_neighbor_size = configs["sim_neighbor_size"]
        self.user_liked_item_size = configs["user_liked_item_size"]

    def fit(self, input_data, configs):
        self.item_num = input_data.item_num
        self.user_num = input_data.user_num
        self.ui_matrix_csc = input_data.ui_matrix_csc
        self.ui_matrix_csr = input_data.ui_matrix_csr
        self._parse_config(configs)

        user_based = configs["user_based"]
        if user_based:
            pass
        else:
            logging.info("Start calculating item-sim matrix.")
            item_sim_matrix_id = np.memmap(path.join(tmp_dir, "item_sim_matrix_id"),
                                           dtype=np.int32,
                                           shape=(self.item_num, self.sim_neighbor_size),
                                           mode="w+")
            item_sim_matrix_val = np.memmap(path.join(tmp_dir, "item_sim_matrix_val"),
                                            dtype=np.float64,
                                            shape=(self.item_num, self.sim_neighbor_size),
                                            mode="w+")

            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(self.sim_of_item)(item_sim_matrix_id,
                                                                 item_sim_matrix_val,
                                                                 item_id)
                                       for item_id in range(self.item_num))

            logging.info("Item-sim matrix done.")

    def sim_of_item(self, item_sim_matrix_id, item_sim_matrix_val, item_id):
        logging.info("item {} start".format(item_id))

        user_bought_item = sp.find(self.ui_matrix_csc.getcol(item_id))[0]
        paired_item_id_set = set()
        item_sim_array = np.zeros(shape=(self.item_num,), dtype=np.float64)
        for u in user_bought_item:
            for i in sp.find(self.ui_matrix_csr.getrow(u))[1]:
                    paired_item_id_set.add(i)

        for paired_item_id in paired_item_id_set:
            item_sim_array[paired_item_id] = self.cosine(self.ui_matrix_csc.getcol(item_id).todense(),
                                                         self.ui_matrix_csc.getcol(paired_item_id).todense())

        top_neighbors = sorted(range(self.item_num), key=lambda x: item_sim_array[x], reverse=True)[0:self.sim_neighbor_size]
        item_sim_matrix_id[item_id, :] = top_neighbors
        item_sim_matrix_val[item_id, :] = item_sim_array[top_neighbors]

        logging.info("item {} done.".format(item_id))

    def cosine(self, x, y):
        return x.T.dot(y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))

    def recommend(self, max_size):
        rec_dict = dict()

        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            item_liked = self._user_liked_item(user_id)
            item_liked_score = self.ui_matrix_csr[user_id, item_liked].todense()
            rec_score = dict()
            print(item_liked)
            for item_id in item_liked:
                for candidate_item_id in sp.find(self.item_sim_matrix.getrow(item_id))[1]:
                    if rec_score.get(candidate_item_id) is None:
                        sim_vec = self.item_sim_matrix[item_liked, candidate_item_id].todense()
                        rec_score[candidate_item_id] = item_liked_score.T.dot(sim_vec) / np.sum(sim_vec)

            print(rec_score.keys())
            rec_item_list = sorted(rec_score.keys(), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]

        return rec_dict

    def _user_liked_item(self, user_id):
        if self.user_liked_item.get(user_id) is None:
            self.user_liked_item[user_id] = sorted(range(self.item_num),
                                                   key=lambda x: self.ui_matrix_csr[user_id, x],
                                                   reverse=True)[0:self.user_liked_item_size]
        return self.user_liked_item[user_id]


