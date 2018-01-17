# coding: utf-8
import numpy as np
import scipy.sparse as sp
# from DataProvider import DataLoader
import logging
from joblib import Parallel, delayed
import multiprocessing
from os import path


class CF(object):
    def __init__(self, user_item_record, user_num, item_num):
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_matrix_csc = self._parse_dict_to_matrix(user_item_record) \
            if isinstance(user_item_record, dict) else user_item_record
        self.user_item_matrix_csr = self.user_item_matrix_csc.tocsr()

        self._item_similarity_matrix = None
        self._user_similarity_matrix = None
        self._item_mean = None
        self._item_std = None
        self._recommend_dict = None

        self.similarity_matrix_size = None
        self.recommend_neighbor_num = None
        self.recommend_neighbor_size = None
        self.recommend_size = None

    @property
    def item_mean(self):
        if self._item_mean is None:
            self._item_mean = np.mean(self.user_item_matrix_csc, axis=0)
            self._item_mean = self._item_mean.reshape((self.item_num, 1))
        return self._item_mean

    @property
    def item_std(self):
        if self._item_std is None:
            self._item_std = np.sqrt(np.sum(self.user_item_matrix_csc.power(2), axis=0).reshape((self.item_num, 1)) -
                                     self.user_num * np.power(self.item_mean, 2))
        return self._item_std

    @property
    def item_similarity_matrix(self):
        if self._item_similarity_matrix is None:
            row = np.array([np.ones(self.similarity_matrix_size, dtype=np.int32) * k for k in range(self.item_num)]) \
                .reshape(self.item_num * self.similarity_matrix_size)
            col = np.memmap(path.join("..", "tmp", "item_similarity_matrix_id"), dtype=np.int32,
                            shape=(self.item_num, self.similarity_matrix_size), mode="r") \
                .reshape(self.item_num * self.similarity_matrix_size)
            data = np.memmap(path.join("..", "tmp", "item_similarity_matrix_val"), dtype=np.float64,
                             shape=(self.item_num, self.similarity_matrix_size), mode="r") \
                .reshape(self.item_num * self.similarity_matrix_size)

            self._item_similarity_matrix = sp.csr_matrix((data, (row, col)), shape=(self.item_num, self.item_num))

        return self._item_similarity_matrix

    @property
    def recommend_dict(self):
        if self._recommend_dict is None:
            self._recommend_dict = dict()
            recommend_list_array = np.memmap(path.join("..", "tmp", "recommend_list_array"),
                                             dtype=np.int32, shape=(self.user_num, self.recommend_size), mode="w+")
            recommend_score_array = np.memmap(path.join("..", "tmp", "recommend_score_array"),
                                              dtype=np.float64, shape=(self.user_num, self.recommend_size), mode="w+")
            for user_id in range(self.user_num):
                self._recommend_dict[user_id] = list()
                for i in self.recommend_size:
                    if recommend_list_array[user_id, i] == -1:
                        break
                    self._recommend_dict[user_id].append(
                        (recommend_list_array[user_id, i], recommend_score_array[user_id, i])
                    )
        return self._recommend_dict

    def fit(self, user_based=False, similarity_matrix_size=100):
        """
        :param similarity_matrix_size: retain top-K simialr objects
        :param user_based: user-based CF or item-based CF
        :return:
        """
        if user_based:
            pass
        else:
            logging.info("Start calculating item-similarity matrix.")
            self.similarity_matrix_size = similarity_matrix_size
            item_similarity_matrix_id = np.memmap(path.join("..", "tmp", "item_similarity_matrix_id"),
                                                  dtype=np.int32, shape=(self.item_num, similarity_matrix_size),
                                                  mode="w+")
            item_similarity_matrix_val = np.memmap(path.join("..", "tmp", "item_similarity_matrix_val"),
                                                   dtype=np.float64, shape=(self.item_num, similarity_matrix_size),
                                                   mode="w+")

            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(self.pearson_of_item)
                                       (item_similarity_matrix_id, item_similarity_matrix_val, item_id,
                                        similarity_matrix_size)
                                       for item_id in range(self.item_num))

            logging.info("Item-similarity matrix done.")

    def pearson_of_item(self, item_similarity_matrix_id, item_similarity_matrix_val, item_id, K):
        logging.info("item {} start".format(item_id))

        user_bought_item1 = sp.find(self.user_item_matrix_csc.getcol(item_id))[0]
        _item_id_set = set()
        item_similarity_array = np.zeros(shape=(self.item_num,), dtype=np.float64)
        for u in user_bought_item1:
            for i in sp.find(self.user_item_matrix_csr.getrow(u))[1]:
                    _item_id_set.add(i)

        for _item_id in _item_id_set:
            item_similarity_array[_item_id] = (self.user_item_matrix_csc.getcol(item_id).T.dot
                                               (self.user_item_matrix_csc.getcol(_item_id)) -
                                               self.user_num * self.item_mean[item_id] *
                                               self.item_mean[_item_id]) / \
                                              (self.item_std[item_id] * self.item_std[_item_id])

        top_neighbors = sorted(range(self.item_num), key=lambda x: item_similarity_array[x], reverse=True)[0:K]
        item_similarity_matrix_id[item_id, :] = top_neighbors
        item_similarity_matrix_val[item_id, :] = item_similarity_array[top_neighbors]

        logging.info("item {} done.".format(item_id))

    def recommend(self, recommend_neighbor_num=20, recommend_neighbor_size=20, recommend_size=100):
        logging.info("Start item-based CF: no. of item {}, no. of user {}".format(self.item_num, self.user_num))
        self.recommend_neighbor_num = recommend_neighbor_num
        self.recommend_neighbor_size = recommend_neighbor_size
        self.recommend_size = recommend_size
        recommend_list_array = np.memmap(path.join("..", "tmp", "recommend_list_array"),
                                         dtype=np.int32, shape=(self.user_num, recommend_size), mode="w+")
        recommend_score_array = np.memmap(path.join("..", "tmp", "recommend_score_array"),
                                          dtype=np.float64, shape=(self.user_num, recommend_size), mode="w+")

        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(self.recommend_for_one_user)
                                   (recommend_list_array, recommend_score_array, user_id)
                                   for user_id in range(self.user_num))
        logging.info("Recommendation done.")
        return self.recommend_dict

    def recommend_for_one_user(self, recommend_list_array, recommend_score_array, user_id):
        logging.info("User {} start".format(user_id))
        score_list = np.zeros(shape=(self.item_num, 1))
        top_rated_items = sorted(range(1, self.item_num), key=lambda x: self.user_item_matrix_csr[user_id, x],
                                 reverse=True)[0:self.recommend_neighbor_num]
        top_rated_scores = self.user_item_matrix_csr[user_id, top_rated_items]
        candidate_set = set()
        for item_id in top_rated_items:
            for i in sorted(range(self.item_num), key=lambda x: self.item_similarity_matrix[item_id, x],
                            reverse=True)[0:self.recommend_neighbor_size]:
                candidate_set.add(i)
        logging.debug("candidate set size: {}".format(len(candidate_set)))

        for item_id in candidate_set:
            item_similarity_vec = self.item_similarity_matrix[item_id, top_rated_items]
            if np.sum(np.abs(item_similarity_vec)) < 10e-6:
                continue
            score_list[item_id] = \
                item_similarity_vec.dot(top_rated_scores.T)[0, 0] / np.sum(np.abs(item_similarity_vec))

        if len(candidate_set) >= self.recommend_size:
            recommend_list = sorted(range(self.item_num), key=lambda x: score_list[x],
                                    reverse=True)[0: self.recommend_size]
            recommend_score = [score_list[k] for k in recommend_list]
            recommend_list_array[user_id, :] = recommend_list
            recommend_score_array[user_id, :] = recommend_score
        else:
            recommend_list = sorted(range(self.item_num), key=lambda x: score_list[x], reverse=True)
            recommend_score = [score_list[k] for k in recommend_list]
            recommend_list_array[user_id, 0:len(candidate_set)] = recommend_list
            recommend_list_array[user_id, len(candidate_set):self.recommend_size] = \
                [-1 for _ in range(self.recommend_size - len(candidate_set))]
            recommend_score_array[user_id, 0:len(candidate_set)] = recommend_score
        logging.info("User {} done.".format(user_id))

    def _parse_dict_to_matrix(self, user_item_dict, score=True):
        """
        parse dict user-item record to sparse matrix.
        :param user_item_dict:
        :return:
        """
        logging.info("Start parse dict to matrix")
        user_item_matrix = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float64)

        if score:
            for user_id in user_item_dict:
                for item_id in user_item_dict[user_id]:
                    user_item_matrix[user_id, item_id] = \
                        self._convert_watch_time_to_score(user_item_dict[user_id][item_id])
        else:
            for user_id in user_item_dict:
                for item_id in user_item_dict[user_id]:
                    user_item_matrix[user_id, item_id] = user_item_dict[user_id][item_id]

        logging.info("Parse work done.")

        user_item_matrix = user_item_matrix.tocsc()

        logging.info("Save matrix..")
        sp.save_npz(path.join("..", "tmp", "user_item_matrix.npz"), user_item_matrix)
        return user_item_matrix

    def _convert_watch_time_to_score(self, watch_time):
        if watch_time < 30:
            return 0
        return np.log10(watch_time + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S', )

    # data_provider = DataLoader()
    # data_provider.load_train("..\\data\\train_data", "..\\tmp")
    # cf_model = CF(data_provider.user_watch_time, data_provider.user_num, data_provider.item_num)

    user_watch_time_matrix = sp.load_npz(path.join("..", "tmp", "user_item_matrix.npz"))
    user_num, item_num = user_watch_time_matrix.shape
    logging.info("Number of items consumed per user: {}".format(1.0 * user_watch_time_matrix.count_nonzero() / user_num))
    cf_model = CF(user_watch_time_matrix, user_num, item_num)

    # cf_model.fit()
    cf_model.similarity_matrix_size = 100
    recommend_dict = cf_model.recommend()
