import tensorflow as tf
from model.constants import *
from os import path


configs = {
    "user_embedding_size": 256,
    "item_embedding_size": 256,
    "emb_init_value": 1,
    "display_step": 100,
    "training_epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.1,
    "lambda_value": 0.001,
    "keep_prob_value": 0.75,
    "layers": [256, 128, 64],
    "class_num": 3,
    "means": [0.75, 1.5, 2.5],
    "stds": [0.2, 0.2, 0.2],
    "prob": [0.333, 0.333, 0.333],
    "activation": tf.nn.relu,
    "model_name": "ProbRateRegression",
    "max_size": 30,
    "save_path": path.join(tmp_dir, "ProbRateRegression"),
    "rec_dict": path.join(data_dir, "prr_rec_list"),
}
