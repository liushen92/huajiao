# coding: utf-8
import numpy as np
from functools import reduce
from math import log2


class TopKTester(object):
    def __init__(self, path_to_truth, path_to_pred):
        """
        :param path_to_truth: path to ground truth, each line of ground truth is :
            userid [(anchorid, watching_length)]
        :param path_to_pred: path to pred, the file format is the same as ground truth file
        with watching_length being replaced by score.
        :param k: Top k recommendation.
        """
        self.path_to_truth = path_to_truth
        self.path_to_pred = path_to_pred
        self.truth_dict, self.truth = self._read_truth()
        self.pred_dict, self.pred = self._read_pred()
        self.user_set = set(self.truth.keys()).intersection(set(self.pred.keys()))

    def precision(self, max_k):
        precision_list = np.zeros(shape=(max_k, 1))
        for user in self.user_set:
            for k in range(max_k):
                precision_list[k] += len(set(self.pred[user][0:k]).intersection(set(self.truth[user]))) / len(self.truth[user])
        precision_list /= len(self.user_set)
        return precision_list

    def recall(self, max_k):
        recall_list = np.zeros(shape=(max_k, 1))
        for user in self.user_set:
            for k in range(max_k):
                recall_list[k] += len(set(self.pred[user][0:k]).intersection(set(self.truth[user]))) / k
        recall_list /= len(self.user_set)
        return recall_list

    def ndcg(self, max_k):
        ndcg_list = np.zeros(shape=(max_k, 1))
        for user in self.user_set:
            for k in range(max_k):
                n = k if k <= len(self.truth[user]) else len(self.truth[user])
                dcg = reduce(lambda x, y: x + y, map(lambda i: (2 ** self.truth_dict[user].get(self.pred[user][i], 0) - 1) / log2(i + 2), range(k)), 0)
                idcg = reduce(lambda x, y: x + y, map(lambda i: (2 ** self.truth_dict[user][self.truth[user][i]] - 1) / log2(i + 2), range(n)), 0)
                ndcg_list[k] += dcg / idcg
        ndcg_list /= len(self.user_set)
        return ndcg_list

    def _read_truth(self):
        return self._read_file(self.path_to_truth)

    def _read_pred(self):
        return self._read_file(self.path_to_pred)

    def _read_file(self, filename):
        _dict = dict()
        _list = dict()
        with open(filename) as f:
            for l in f.readlines():
                user_md5 = l.strip().split("\t")[0]
                _dict[user_md5] = dict()
                _list[user_md5] = list()
                total_count = 0
                for item_md5, item_count in l.strip().split("\t")[1].split(" "):
                    total_count += float(item_count)

                for item_md5, item_count in l.strip().split("\t")[1].split(" "):
                    _dict[user_md5][item_md5] = float(item_count) / total_count
                    _list[user_md5].append(item_md5)
        return _dict, _list
