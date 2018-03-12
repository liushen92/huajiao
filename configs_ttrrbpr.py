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
    "emb_lambda": 0.001,
    "nn_lambda": 0.001,
    "label_lambda": 0.05,
    "keep_prob_value": 0.75,
    "bpr_layers": [256, 128, 64],
    "label_layers": [128, 64, 32],
    "pos_item_threshold": 300,
    "neg_samples_num": 10,
    "bpr_activation": tf.nn.relu,
    "label_activation": tf.nn.relu,
    "model_name": "TTRateRegressionBPR",
    "max_size": 30,
    "save_path": path.join(tmp_dir, "TTRateRegressionBPR"),
    "rec_dict": path.join(data_dir, "ttrrbpr_rec_list"),
}
