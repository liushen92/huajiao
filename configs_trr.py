import tensorflow as tf
from model.constants import *
from os import path


configs = {
    "user_embedding_size": 256,
    "item_embedding_size": 256,
    "user_common_layers": [],
    "item_common_layers": [],
    "user_label_layers": [128, 64],
    "item_label_layers": [128, 64],
    "score_layers": [256, 128, 64],
    "emb_init_value": 1,
    "display_step": 100,
    "training_epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.1,
    "label_lambda": 0.2,
    "emb_lambda": 0.001,
    "activation": tf.nn.relu,
    "model_name": "TagRateRegression",
    "max_size": 30,
    "save_path": path.join(tmp_dir, "TagRateRegression"),
    "rec_dict": path.join(data_dir, "trr_rec_list"),
}
