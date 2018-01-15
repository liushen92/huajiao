import numpy as np
import os


def mkdir(file_path):
    pardir = os.path.dirname(file_path)
    if not os.path.exists(pardir):
        os.mkdir(pardir)


def save_dict(saved_dict, file_path):
    mkdir(file_path)
    with open(file_path, 'w') as f:
        for k, v in saved_dict.items():
            f.write("{0}\t{1}\n".format(str(k), str(v)))


def load_dict(file_path):
    loaded_dict = dict()
    with open(file_path, 'r') as f:
        for l in f.readlines():
            tmp = l.strip().split("\t")
            loaded_dict[tmp[0]] = tmp[1]
    return loaded_dict


def load_inverse_dict(file_path):
    loaded_dict = dict()
    with open(file_path, 'r') as f:
        for l in f.readlines():
            tmp = l.strip().split("\t")
            loaded_dict[tmp[1]] = tmp[0]
    return loaded_dict


def save_matrix(saved_matrix, file_path):
    mkdir(file_path)
    np.savetxt(file_path, saved_matrix, delimiter=",")
