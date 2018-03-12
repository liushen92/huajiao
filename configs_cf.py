from model.constants import *
from os import path


configs = {
    "sim_neighbor_size": 100,
    "user_liked_item_size": 100,
    "max_size": 30,
    "rec_dict": path.join(data_dir, "cf_rec_list"),
}
