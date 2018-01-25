# coding: utf-8
import logging
import os
import tensorflow as tf
import model.mf as mf
import model.RateRegression as rr
from model.constants import *


mf_configs = {
    'embedding_size': 256,
    'learning_rate': 0.01,
    'training_epochs': 10,
    'self.batch_size': 256,
    'emb_init_value': 1,
    'lambda_value': 0.00001,
    'display_step': 100,
    'optimize_method': 'adam',
    'model_name': "mf_model"
}

rr_configs = {
    "user_embedding_size": 256,
    "item_embedding_size": 256,
    "emb_init_value": 1,
    "display_step": 100,
    "training_epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.1,
    "lambda_value": 0.00001,
    "layers": [512, 256, 128],
    "activation": tf.nn.relu,
    "model_name": "rr_model"
}

if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S', )

    input_data = mf.MFDataProvider()
#    model = mf.MatrixFactorization()
#    mf_configs['save_path'] = os.path.join(tmp_dir, "mf")
#    with tf.Session() as sess:
#        model.fit(sess=sess, input_data=input_data, configs=mf_configs)
#        rec_dict = model.recommend(sess, 30)
#        input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=os.path.join(data_dir, "mf_rec_list"))

    model = rr.RateRegression()
    rr_configs['save_path'] = os.path.join(tmp_dir, "RateRegression")
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.fit(sess=sess, input_data=input_data, configs=rr_configs)
        rec_dict = model.recommend(sess, 30)
        input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=os.path.join(data_dir, "rr_rec_list"))
