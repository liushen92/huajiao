# coding: utf-8
import logging
import os
import tensorflow as tf
import model.mf as mf
from model.constants import *


mf_configs = {
    'embedding_size': 256,
    'learning_rate': 0.01,
    'training_epochs': 0,
    'self.batch_size': 256,
    'emb_init_value': 1,
    'lambda_value': 0.0001,
    'display_step': 100,
    'optimize_method': 'adam',
    'model_name': "mf_model"
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S', )

    input_data = mf.MFDataProvider()
    model = mf.MatrixFactorization()
    mf_configs['save_path'] = os.path.join(tmp_dir, "mf")
    with tf.Session() as sess:
        model.fit(sess=sess, input_data=input_data, configs=mf_configs)
        rec_dict = model.recommend(sess, 20)
        input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=os.path.join(data_dir, "mf_rec_list"))
