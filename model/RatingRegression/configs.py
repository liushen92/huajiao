# coding: utf-8
import tensorflow as tf


configs = {
    "user_embedding_size": 300,
    "item_embedding_size": 300,
    "emb_init_value": 2,
    "display_step": 100,
    "training_epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.1,
    "layers": [400, 400, 400],
    "activation": tf.nn.sigmoid,
}