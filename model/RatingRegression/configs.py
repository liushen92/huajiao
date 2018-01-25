# coding: utf-8
import tensorflow as tf


configs = {
    "user_embedding_size": 256,
    "item_embedding_size": 256,
    "emb_init_value": 1,
    "display_step": 100,
    "training_epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.1,
    "lambda_value": 0.0001,
    "layers": [512, 256, 128],
    "activation": tf.nn.relu,
}