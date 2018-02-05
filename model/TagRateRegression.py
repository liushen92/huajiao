# coding: utf-8
import logging
from os import path
import numpy as np
import scipy.sparse as sp
from .constants import *
from .mf import MFDataProvider
import tensorflow as tf


class TrrDataProvider(MFDataProvider):
    def __init__(self):
        super(TrrDataProvider, self).__init__()
        self.user_label_matrix = self._load_labels(path.join(data_dir, "user_portrait"),
                                                   self.user_num,
                                                   self.user2id)
        self.anchor_label_matrix = self._load_labels(path.join(data_dir, "anchor_portrait"),
                                                     self.anchor_num,
                                                     self.anchor2id)

    def _load_labels(self, portrait_file, item_num, item_dict):
        label_matrix = sp.dok_matrix((item_num, self.label_num), dtype=np.float32)
        with open(portrait_file, encoding="utf-8") as f:
            for l in f.readlines():
                tmp = l.strip().split('\t')
                if len(tmp) == 1:
                    continue
                item_md5, item_labels = tmp
                item_id = item_dict[item_md5]
                total_count = 0.0
                for label_and_count in item_labels.split(","):
                    _, label_count = label_and_count.split(":")
                    total_count += int(label_count)

                for label_and_count in item_labels.split(','):
                    label, label_count = label_and_count.split(":")
                    label_id = self.label2id[label]
                    label_count = int(label_count)
                    label_matrix[item_id, label_id] = label_count / total_count
        return label_matrix

    def batch_generator(self, batch_size):
        for i in range(0, len(self.user_watch_time), batch_size):
            batch_data = dict()
            start_idx = i
            end_idx = min(i + batch_size, len(self.user_watch_time))
            batch_data['user_idx'] = self.user_watch_time[start_idx: end_idx, 0].astype(np.int32)
            batch_data['item_idx'] = self.user_watch_time[start_idx: end_idx, 1].astype(np.int32)
            batch_data['user_item_score'] = self.user_watch_time[start_idx: end_idx, 2]
            batch_data['user_labels'] = self.user_label_matrix[batch_data["user_idx"], :].todense()
            batch_data['item_labels'] = self.anchor_label_matrix[batch_data["item_idx"], :].todense()
            yield batch_data


class TagRateRegression(object):
    def __init__(self):
        # model configuration parameters
        self.user_num = None
        self.item_num = None
        self.label_num = None
        self.user_embedding_size = None
        self.item_embedding_size = None
        self.emb_init_value = None
        self.display_step = None
        self.training_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.optimize_method = None
        self.emb_lambda = None
        self.label_lambda = None

        # model variables
        self.user_idx = None
        self.item_idx = None
        self.user_item_score = None
        self.user_labels = None
        self.item_labels = None
        self.loss = None
        self.score_pred = None
        self.user_label_pred = None
        self.item_label_pred = None
        self.score_loss = None
        self.user_label_loss = None
        self.item_label_loss = None
        self.optimizer = None

    def _parse_config(self, configs):
        self.user_embedding_size = configs["user_embedding_size"]
        self.item_embedding_size = configs["item_embedding_size"]
        self.learning_rate = configs.get('learning_rate', 0.01)
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('batch_size', 2)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'adam')
        self.emb_lambda = configs.get("emb_lambda", 0.0001)
        self.label_lambda = configs.get("label_lambda", 1)

    def define_model(self, configs):
        self._parse_config(configs)

        # placeholder layers and embedding matrices
        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="item_idx")
            self.user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name="user_item_score")
            self.user_labels = tf.placeholder(dtype=tf.float32, shape=[None, self.label_num], name="user_labels")
            self.item_labels = tf.placeholder(dtype=tf.float32, shape=[None, self.label_num], name="item_labels")

        with tf.name_scope("embedding"):
            user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.user_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb")
            item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.item_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb")
            item_emb = tf.nn.embedding_lookup(item_emb_matrix, self.item_idx)
            user_emb = tf.nn.embedding_lookup(user_emb_matrix, self.user_idx)

        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.emb_lambda)
        parameters = dict()

        # common layers
        with tf.name_scope("common_layers"):
            with tf.name_scope("user_common_layers"):
                parameters["user_common_layer_0"] = user_emb
                for i in range(len(configs["user_common_layers"])):
                    parameters["user_common_layer_" + str(i + 1)] = tf.layers.dense(parameters["user_common_layer_" +
                                                                                               str(i)],
                                                                                    configs["user_common_layers"][i],
                                                                                    activation=configs["activation"],
                                                                                    kernel_regularizer=regularizer,
                                                                                    bias_regularizer=regularizer,
                                                                                    kernel_initializer=initializer)
            with tf.name_scope("item_common_layers"):
                parameters["item_common_layer_0"] = item_emb
                for i in range(len(configs["item_common_layers"])):
                    parameters["item_common_layer_" + str(i + 1)] = tf.layers.dense(parameters["item_common_layer_" +
                                                                                               str(i)],
                                                                                    configs["item_common_layers"][i],
                                                                                    activation=configs["activation"],
                                                                                    kernel_regularizer=regularizer,
                                                                                    bias_regularizer=regularizer,
                                                                                    kernel_initializer=initializer)
        # score layers
        with tf.name_scope("score_layers"):
            parameters["score_layer_0"] = tf.concat(
                [parameters["user_common_layer_" + str(len(configs["user_common_layers"]))],
                 parameters["item_common_layer_" + str(len(configs["item_common_layers"]))]], 1)
            for i in range(len(configs["score_layers"])):
                parameters["score_layer_" + str(i + 1)] = tf.layers.dense(parameters["score_layer_" + str(i)],
                                                                          configs["score_layers"][i],
                                                                          activation=configs["activation"],
                                                                          kernel_regularizer=regularizer,
                                                                          bias_regularizer=regularizer,
                                                                          kernel_initializer=initializer)
            self.score_pred = tf.layers.dense(parameters["score_layer_" + str(len(configs["score_layers"]))], 1,
                                              activation=None,
                                              name="score_prediction")

        # label layers
        with tf.name_scope("label_layers"):
            with tf.name_scope("user_label_layers"):
                parameters["user_label_layer_0"] = parameters["user_common_layer_" + str(len(configs["user_common_layers"]))]
                for i in range(len(configs["user_label_layers"])):
                    parameters["user_label_layer_" + str(i + 1)] = tf.layers.dense(parameters["user_label_layer_" + str(i)],
                                                                                   configs["user_label_layers"][i],
                                                                                   activation=configs["activation"],
                                                                                   kernel_regularizer=regularizer,
                                                                                   bias_regularizer=regularizer,
                                                                                   kernel_initializer=initializer)
                self.user_label_pred = tf.layers.dense(parameters["user_label_layer_" + str(len(configs["user_label_layers"]))],
                                                       self.label_num, activation=tf.nn.relu, name="user_label_prediction")

            with tf.name_scope("item_label_layers"):
                parameters["item_label_layer_0"] = parameters["item_common_layer_" + str(len(configs["item_common_layers"]))]
                for i in range(len(configs["item_label_layers"])):
                    parameters["item_label_layer_" + str(i + 1)] = tf.layers.dense(parameters["item_label_layer_" + str(i)],
                                                                                   configs["item_label_layers"][i],
                                                                                   activation=configs["activation"],
                                                                                   kernel_regularizer=regularizer,
                                                                                   bias_regularizer=regularizer,
                                                                                   kernel_initializer=initializer)
                self.item_label_pred = tf.layers.dense(parameters["item_label_layer_" + str(len(configs["item_label_layers"]))],
                                                       self.label_num, activation=tf.nn.relu, name="item_label_prediction")

        # loss layers
        with tf.name_scope("loss"):
            self.score_loss = tf.losses.mean_squared_error(tf.reshape(self.user_item_score, [-1, 1]), self.score_pred)
            self.user_label_loss = tf.reduce_mean(self.label_loss(self.user_labels, tf.nn.softmax(self.user_label_pred)))
            self.item_label_loss = tf.reduce_mean(self.label_loss(self.item_labels, tf.nn.softmax(self.item_label_pred)))

            self.loss = tf.add_n([self.score_loss, self.label_lambda * self.user_label_loss, self.label_lambda * self.item_label_loss])
            self.loss += self.emb_lambda * (tf.add_n([tf.reduce_mean(tf.square(user_emb)),
                                                      tf.reduce_mean(tf.square(item_emb))]))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def label_loss(self, truth, pred):
        return -tf.reduce_sum(tf.multiply(truth, pred), keep_dims=True, axis=1)

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"],
            self.user_labels: input_batch["user_labels"],
            self.item_labels: input_batch["item_labels"],
        }

    def run_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, _ = sess.run([self.loss, self.optimizer], feed_dict=self.create_feed_dict(input_batch))
            total_loss += loss_batch
            if i % self.display_step == 0:
                logging.info('Average loss at epoch {} step {}: {:5.6f}'
                             .format(epoch_idx, i, total_loss / self.display_step))
                total_loss = 0.0

    def fit(self, sess, input_data, configs):
        self.user_num = input_data.user_num
        self.item_num = input_data.anchor_num
        self.label_num = input_data.label_num
        self.define_model(configs)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        logging.info("Start training")
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)
        logging.info("Training complete and saving...")
        saver.save(sess, os.path.join(configs["save_path"],
                                      configs["model_name"]))

    def recommend(self, sess, max_size, model_path=None, model_name=None):
        if self.loss is None:
            if model_path is None:
                logging.error("Saving path of rate regression model should be given")
            saver = tf.train.import_meta_graph(os.path.join(model_path, model_name))
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            graph = tf.get_default_graph()
            self.user_idx = graph.get_tensor_by_name("placeholders/user_idx")
            self.item_idx = graph.get_tensor_by_name("placeholders/item_idx")
            self.user_num = tf.shape(graph.get_tensor_by_name("embedding/user_emb"))[0]
            self.item_num = tf.shape(graph.get_tensor_by_name("embedding/item_emb"))[0]
            self.score_pred = graph.get_tensor_by_name("score_layers/score_prediction")

        rec_dict = dict()
        item_list = np.array(range(self.item_num))
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            feed_dict = {self.user_idx: np.array([user_id] * self.item_num), self.item_idx: item_list}
            rec_score = sess.run(self.score_pred, feed_dict=feed_dict)
            rec_item_list = sorted(range(self.item_num), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]
        return rec_dict
