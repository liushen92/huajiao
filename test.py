# coding: utf-8
import numpy as np
from functools import reduce
from math import log2
import argparse


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
        precision_list = np.zeros(shape=(max_k, ))
        for user in self.user_set:
            for k in range(max_k):
                precision_list[k] += len(set(self.pred[user][0:(k + 1)]).intersection(set(self.truth[user]))) / (k + 1)
        precision_list /= len(self.user_set)
        return list(precision_list)

    def recall(self, max_k):
        recall_list = np.zeros(shape=(max_k, ))
        for user in self.user_set:
            for k in range(max_k):
                recall_list[k] += len(set(self.pred[user][0:(k + 1)]).intersection(set(self.truth[user]))) / len(self.truth[user])
        recall_list /= len(self.user_set)
        return list(recall_list)

    def ndcg(self, max_k):
        ndcg_list = np.zeros(shape=(max_k, ))
        for user in self.user_set:
            for k in range(max_k):
                n = (k + 1) if (k + 1) <= len(self.truth[user]) else len(self.truth[user])
                dcg = 0
                for i in range(k + 1):
                    dcg += (2 ** self.truth_dict[user].get(self.pred[user][i], 0) - 1) / log2(i + 2)
                idcg = 0
                for i in range(n):
                    idcg += (2 ** self.truth_dict[user][self.truth[user][i]] - 1) / log2(i + 2)
                ndcg_list[k] += dcg / idcg
        ndcg_list /= len(self.user_set)
        return list(ndcg_list)

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
                for tmp in l.strip().split("\t")[1].split(" "):
                    item_md5, item_count = tmp.split(":")
                    total_count += float(item_count)

                for tmp in l.strip().split("\t")[1].split(" "):
                    item_md5, item_count = tmp.split(":")
                    _dict[user_md5][item_md5] = float(item_count) / total_count
                    _list[user_md5].append(item_md5)
        return _dict, _list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="huajiaoTester")
    parser.add_argument("--truth", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--k", type=int)
    args = parser.parse_args()
    tester = TopKTester(args.truth, args.pred)
    max_k = args.k

    print("precision@1-{}".format(max_k))
    print("\t".join(map(lambda s: "{0:.6f}".format(s), tester.precision(max_k))))

    print("recall@1-{}".format(max_k))
    print("\t".join(map(lambda s: "{0:.6f}".format(s), tester.recall(max_k))))

    print("ndcg@1-{}".format(max_k))
    print("\t".join(map(lambda s: "{0:.6f}".format(s), tester.ndcg(max_k))))

