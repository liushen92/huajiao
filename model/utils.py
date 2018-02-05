# coding: utf - 8
import pickle
import numpy as np
import os


def mkdir(file_path):
    pardir = os.path.dirname(file_path)
    if not os.path.exists(pardir):
        os.mkdir(pardir)


def save_dict(dict_to_save, filename):
    with open(filename, 'w') as f:
        for key, value in dict_to_save.items():
            f.write("{}\t{}\n".format(str(key), str(value)))


def load_dict(filename):
    _dict = dict()
    with open(filename, 'r', encoding="utf-8") as f:
        for l in f.readlines():
            key, value = l.strip().split('\t')
            _dict[key] = int(value)
    return _dict


def reverse_dict(dict_to_reverse):
    reversed_dict = {v: k for (k, v) in dict_to_reverse.items()}
    return reversed_dict


def pickled_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_matrix(saved_matrix, file_path):
    mkdir(file_path)
    np.savetxt(file_path, saved_matrix, delimiter=",")
