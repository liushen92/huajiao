# coding: utf - 8


def save_dict(dict_to_save, filename):
    with open(filename, 'w') as f:
        for key, value in dict_to_save.items():
            f.write("{}\t{}\n".format(str(key), str(value)))


def load_dict(filename):
    _dict = dict()
    with open(filename, 'r') as f:
        for l in f.readlines():
            key, value = l.strip().split('\t')
            _dict[key] = int(value)
    return _dict


def reverse_dict(dict_to_reverse):
    reversed_dict = {v: k for (k, v) in dict_to_reverse.items()}
    return reversed_dict
