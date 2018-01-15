# coding: utf-8
import numpy as np
import scipy.sparse as sp
from DataProvider import DataLoader
import logging
from joblib import Parallel, delayed
import multiprocessing


class CF(object):
    def __init__(self, user_item_record, user_num, item_num):
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_matrix_csc = self._parse_dict_to_matrix(user_item_record) \
            if isinstance(user_item_record, dict) else user_item_record
        self.user_item_matrix_csr = self.user_item_matrix_csc.tocsr()

        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self._item_mean = None
        self._item_std = None

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

    def fit(self, user_based=False, K=100):
        """
        :param user_based: user-based CF or item-based CF
        :param K: retain top-K simialr objects
        :return:
        """
        if user_based:
            pass
        else:
            logging.info("Start calculating item-similarity matrix.")
            self.item_similarity_matrix = sp.dok_matrix((self.item_num, self.item_num), dtype=np.float64)

            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(self.pearson_of_item)(item_id) for item_id in range(self.item_num))
            # for item_id in range(self.item_num):
            #     self.pearson_of_item(item_id)

            logging.info("Item-similarity matrix done.")
            self.item_similarity_matrix = self.item_similarity_matrix.tocsr()
            sp.save_npz("..\\tmp\\item_similarity_matrix", self.item_similarity_matrix)

    def pearson_of_item(self, item_id):
        print("item {} start".format(item_id))

        user_bought_item1 = sp.find(self.user_item_matrix_csc.getcol(item_id))[0]
        _item_id_set = set()
        for u in user_bought_item1:
            for i in sp.find(self.user_item_matrix_csr.getrow(u))[1]:
                if i > item_id:
                    _item_id_set.add(i)

        for _item_id in _item_id_set:
            self.item_similarity_matrix[item_id, _item_id] = (self.user_item_matrix_csc.getcol(item_id).T.dot
                                                              (self.user_item_matrix_csc.getcol(_item_id)) -
                                                              self.user_num * self.item_mean[item_id] *
                                                              self.item_mean[_item_id]) / \
                                                              (self.item_std[item_id] * self.item_std[_item_id])
            self.item_similarity_matrix[_item_id, item_id] = self.item_similarity_matrix[item_id, _item_id]

        print("item {} done.".format(item_id))

    def recommend(self, TopK=20, K=20, max_size=100):
        recommend_dict = dict()
        logging.info("Start item-based CF: no. of item {}, no. of user {}".format(self.item_num, self.user_num))
        score_list = np.zeros(shape=(self.item_num, 1))
        for user_id in range(self.user_num):
            logging.debug("User {} start".format(user_id))

            top_rated_items = sorted(range(1, self.item_num), key=lambda x: self.user_item_matrix_csr[user_id, x], reverse=True)[0:TopK]
            top_rated_scores = self.user_item_matrix_csr[user_id, top_rated_items]
            candidate_set = set()
            for item_id in top_rated_items:
                for i in sorted(range(self.item_num), key=lambda x: self.item_similarity_matrix[item_id, x], reverse=True)[0:K]:
                    candidate_set.add(i)

            for item_id in candidate_set:
                item_similarity_vec = self.item_similarity_matrix[np.ones(shape=[K, 1], dtype=np.int32) * item_id, top_rated_items]
                score_list[item_id] = top_rated_scores.dot(item_similarity_vec) / np.linalg.norm(item_similarity_vec, ord=1)

            recommend_dict[user_id] = [(k, score_list[k])
                                       for k in sorted(range(self.item_num), key=lambda x: score_list[x], reverse=True)[0:max_size]
                                       ]
            logging.debug("User {} done.".format(user_id))

        return recommend_dict

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
                    user_item_matrix[user_id, item_id] = self._convert_watch_time_to_score(user_item_dict[user_id][item_id])
        else:
            for user_id in user_item_dict:
                for item_id in user_item_dict[user_id]:
                    user_item_matrix[user_id, item_id] = user_item_dict[user_id][item_id]

        logging.info("Parse work done.")

        user_item_matrix = user_item_matrix.tocsc()

        logging.info("Save matrix..")
        sp.save_npz("..\\tmp\\user_item_matrix", user_item_matrix)
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

    user_watch_time_matrix = sp.load_npz("..\\tmp\\user_item_matrix.npz")
    user_num, item_num = user_watch_time_matrix.shape
    logging.info("Number of items consumed per user: {}".format(1.0 * user_watch_time_matrix.count_nonzero() / user_num))
    cf_model = CF(user_watch_time_matrix, user_num, item_num)

    cf_model.fit()
    # recommend_dict = cf_model.recommend()
