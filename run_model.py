# coding: utf-8
import logging
import os
import tensorflow as tf
import model.mf as mf
import model.RateRegression as rr
from model.constants import *
from model.DataInterface import DataInterface
import argparse

def parse_cmd():
    parser = argparse.ArgumentParser(description="huajiao")
    parser.add_argument("--model", type=str, metavar="model_type", help="specify model")
    parser.add_argument("--configs", type=str, metavar="config_file", help="specify file for specifying configs")
    # parser.add_argument("--model_name", str, metavar="model_name")
    # parser.add_argument("-save", str)
    # parser.add_argument("-rec_dict", str)
    # parser.add_argument("-max_size", type=int, default=30)

    args = parser.parse_args()
    model_type = args.model
    configs = __import__(args.configs).configs

    return model_type, configs


def run_mf(sess, configs):
    input_data = mf.MFDataProvider()
    model = mf.MatrixFactorization()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_rr(sess, configs):
    input_data = mf.MFDataProvider()
    model = rr.RateRegression()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def generate_test_data(filename):
    input_data = DataInterface()
    input_data.generate_test_data(os.path.join(data_dir, "test_data"),
                                  os.path.join(data_dir, filename))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_options = tf.GPUOptions(allow_growth=True)

    model_type, configs = parse_cmd()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if model_type == "mf":
            run_mf(sess, configs)
        elif model_type == "rr":
            run_rr(sess, configs)
        else:
            logging.error("Unknown model.")
