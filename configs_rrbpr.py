import tensorflow as tf
from model.constants import *
from os import path


configs = {
    "user_embedding_size": 256,
    "item_embedding_size": 256,
    "emb_init_value": 1,
    "display_step": 100,
    "training_epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.1,
    "lambda_value": 0.001,
    "keep_prob_value": 0.75,
    "layers": [256, 128, 64],
    "pos_item_threshold": 300,
    "neg_samples_num": 10,
    "activation": tf.nn.relu,
    "model_name": "RateRegressionBPR",
    "max_size": 30,
    "save_path": path.join(tmp_dir, "RateRegressionBPR"),
    "rec_dict": path.join(data_dir, "rrbpr_rec_list"),
}
