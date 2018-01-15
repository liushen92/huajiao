# coding: utf-8


class Tester(object):
    def __init__(self, path_to_truth, path_to_pred):
        self.path_to_truth = path_to_truth
        self.path_to_pred = path_to_pred


class TopKTester(Tester):
    def __init__(self, path_to_truth, path_to_pred):
        """
        :param path_to_truth: path to ground truth, each line of ground truth is :
            userid [(anchorid, watching_length)]
        :param path_to_pred: path to pred, the file format is the same as ground truth file
        with watching_length being replaced by score.
        :param k: Top k recommendation.
        """
        super(TopKTester, self).__init__(path_to_truth, path_to_pred)
        self.truth = self._read_truth()
        self.pred = self._read_pred()

    def precision(self, ks, list_flag):
        start_k = 1 if list_flag else ks
        n = 0
        precision_list = list()
        for k in range(start_k, ks + 1):
            sum = 0.0
            for userid, pred_list in self.pred.items():
                truth_list = self.truth.get(userid)
                if truth_list is None:
                    continue
                pos_true = len(set(pred_list[0:k]).intersection(set(truth_list[0:k])))
                n += 1
                sum += pos_true / k
            precision_list.append(sum / n)
        return precision_list

    def ndcg(self, ks, list_flag):
        start_k = 1 if list_flag else ks
        n = 0
        ndcg_list = list()

    def _read_truth(self):
        return self._read_file(self.path_to_truth)

    def _read_pred(self):
        return self._read_file(self.path_to_pred)

    def _read_file(self, filename):
        _dict = dict()
        with open(filename) as f:
            for l in f.readlines():
                userid = int(l.strip()[0])
                _dict[userid] = map(lambda x: tuple(map(lambda y: int(y), x.split(","))),
                                    l.strip()[1][2:-2].split("),("))
        return _dict
