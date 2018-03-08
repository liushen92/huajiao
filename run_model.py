# coding: utf-8
import logging
import tensorflow as tf
import numpy as np
import model.mf as mf
import model.RateRegression as rr
import model.TagRateRegression as trr
import model.TaggingTagRateRegression as ttrr
import model.CF as cf
import model.ProbRatingRegression as prr
import model.MostWatched as mw
import model.bpr as bpr
import model.RateRegressionBPR as rrbpr
from model.constants import *
from model.DataInterface import DataInterface
import argparse


def parse_cmd():
    parser = argparse.ArgumentParser(description="huajiao")
    parser.add_argument("-model", type=str, metavar="model_type", help="specify model")
    parser.add_argument("-configs", type=str, metavar="config_file", help="specify file for specifying configs")
    parser.add_argument("-seed", type=int, metavar="seed", help="specify random seed")
    parser.add_argument("-model_name", type=str)
    parser.add_argument("-rec_dict", type=str)
    parser.add_argument("-max_size", type=int)

    args = parser.parse_args()
    model_type = args.model
    configs = __import__(args.configs).configs
    seed = args.seed
    if args.model_name is not None:
        configs["model_name"] = args.model_name
        configs["save_path"] = os.path.join(tmp_dir, configs["model_name"])
    if args.rec_dict is not None:
        configs["rec_dict"] = os.path.join(data_dir, args.rec_dict)
    if args.max_size is not None:
        configs["max_size"] = args.max_size

    return model_type, configs, seed


def run_mf(sess, configs):
    input_data = mf.MFDataProvider()
    model = mf.MatrixFactorization()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_rr(sess, configs):
    input_data = rr.RRDataProvider()
    model = rr.RateRegression()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_rrbpr(sess, configs):
    input_data = bpr.BPRDataProvider(configs["pos_item_threshold"])
    model = rrbpr.RateRegression()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_trr(sess, configs):
    input_data = trr.TrrDataProvider()
    model = trr.TagRateRegression()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_ttrr(sess, configs):
    input_data = ttrr.TtrrDataProvider()
    model = ttrr.TaggingTagRateRegression()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_cf(configs):
    configs["user_based"] = False
    input_data = cf.CFDataProvider()
    model = cf.CF()
    model.fit(input_data, configs=configs)
    rec_dict = model.recommend(configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_prr(sess, configs):
    input_data = prr.ProbRRDataProvider()
    model = prr.ProbRateRegression()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess, configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_mw(configs):
    input_data = mw.MWDataProvider()
    model = mw.MostWatched()
    model.fit(input_data=input_data)
    rec_dict = model.recommend(configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def run_bpr(sess, configs):
    input_data = bpr.BPRDataProvider(configs["pos_item_threshold"])
    model = bpr.BPR()
    model.fit(sess=sess, input_data=input_data, configs=configs)
    rec_dict = model.recommend(sess=sess, max_size=configs["max_size"])
    input_data.save_rec_dict(recommend_dict=rec_dict, path_to_rec_file=configs["rec_dict"])


def generate_test_data(filename):
    input_data = DataInterface()
    input_data.generate_test_data(os.path.join(data_dir, "test_data"),
                                  os.path.join(data_dir, filename))


def main():
    # parse command line
    model_type, configs, seed = parse_cmd()

    # set logging format
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S')

    if model_type == "cf":
        run_cf(configs)
        return
    elif model_type == "mw":
        run_mw(configs)
        return

    # GPU setting of tf
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_options = tf.GPUOptions(allow_growth=True)

    # add seed to get reproducible result.
    if seed is not None:
        tf.set_random_seed(seed)
        np.random.seed(seed)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if model_type == "mf":
            run_mf(sess, configs)
        elif model_type == "rr":
            run_rr(sess, configs)
        elif model_type == "trr":
            run_trr(sess, configs)
        elif model_type == "ttrr":
            run_ttrr(sess, configs)
        elif model_type == "prr":
            run_prr(sess, configs)
        elif model_type == "bpr":
            run_bpr(sess, configs)
        elif model_type == "rrbpr":
            run_rrbpr(sess, configs)
        else:
            logging.error("Unknown model.")


if __name__ == "__main__":
    main()
