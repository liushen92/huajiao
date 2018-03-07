# coding: utf-8
import tensorflow as tf
import numpy as np
import logging
from os import path

from .constants import *
from .DataInterface import DataInterface


class MFDataProvider(DataInterface):
    def __init__(self):
        super(MFDataProvider, self).__init__()
        self.load_data(path.join(data_dir, "train_data"))
        self._parse_dict_to_nparray(self.user_anchor_behavior)

    def batch_generator(self, batch_size):
        """
        :param batch_size: size of mini-batch
        :return: batch_data: a generator for generate batch
        """
        np.random.shuffle(self.user_watch_time)
        for i in range(0, len(self.user_watch_time), batch_size):
            batch_data = dict()
            start_idx = i
            end_idx = min(i + batch_size, len(self.user_watch_time))
            batch_data['user_idx'] = self.user_watch_time[start_idx: end_idx, 0].astype(np.int32)
            batch_data['item_idx'] = self.user_watch_time[start_idx: end_idx, 1].astype(np.int32)
            batch_data['user_item_score'] = self.user_watch_time[start_idx: end_idx, 2]
            yield batch_data

    def _parse_dict_to_nparray(self, user_anchor_behavior):
        user_watch_time_list = list()
        for user_id in user_anchor_behavior:
            for anchor_id in user_anchor_behavior[user_id]:
                user_watch_time_list.append([user_id, anchor_id,
                                             self._convert_watch_time_to_score(
                                                 user_anchor_behavior[user_id][anchor_id][0])])
        self.user_watch_time = np.array(user_watch_time_list)

    def _convert_watch_time_to_score(self, watch_time):
        return np.log(np.log(watch_time + 1) + 1)


class ConstrainedPMF(object):
    def __init__(self):
        # model configuration parameters
        self.user_num = None
        self.item_num = None
        self.embedding_size = None
        self.embedding_size = None
        self.learning_rate = None
        self.training_epochs = None
        self.batch_size = None
        self.emb_init_value = None
        self.lambda_value = None
        self.display_step = None
        self.optimize_method = None

        # model variables
        self.user_item_score = None
        self.user_idx = None
        self.item_idx = None
        self.item_emb_matrix = None
        self.user_emb_matrix = None
        self.loss = None
        self.optimizer = None
        self.pred = None

    def _parse_config(self, configs):
        self.embedding_size = configs['embedding_size']
        self.learning_rate = configs.get('learning_rate', 0.01)
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('emb_init_value', 2)
        self.lambda_value = configs.get('lambda_value', 0.0001)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'sgd')

    def define_model(self, configs):
        self._parse_config(configs)

        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='user_idx')
            self.item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='item_idx')
            self.user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name='user_item_score')

        with tf.name_scope("embedding"):
            self.user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb_w"
            )
            self.item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb_w"
            )
            self.item_con_matrix = tf.get_variable("item_con_matrix",
                initializer=tf.random_uniform([self.item_num, self.embedding_size],
                                              -self.emb_init_value, self.emb_init_value))
        with tf.name_scope("loss"):
            user_emb = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_idx, name="user_emb")
            item_emb = tf.nn.embedding_lookup(self.item_emb_matrix, self.item_idx, name="item_emb")
            self.pred = tf.add_n([tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=1, keep_dims=True), name="prediction")
            self.loss = tf.reduce_mean(tf.square(self.user_item_score - self.pred)
                                       + self.lambda_value * (tf.add_n([tf.reduce_mean(tf.square(user_emb)),
                                                                        tf.reduce_mean(tf.square(item_emb)),
                                                                        tf.reduce_mean(tf.square(user_bias)),
                                                                        tf.reduce_mean(tf.square(item_bias))])))

        with tf.name_scope("optimizer"):
            if self.optimize_method == "adam":
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            elif self.optimize_method == "sgd":
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def run_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, _ = sess.run([self.loss, self.optimizer],
                                     feed_dict=self.create_feed_dict(input_batch))
            total_loss += loss_batch
            if i % self.display_step == 0:
                logging.info('Average loss at epoch {} step {}: {:5.6f}'
                             .format(epoch_idx, i, total_loss / self.display_step))
                total_loss = 0.0

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"]
        }

    def fit(self, sess, input_data, configs):
        self.user_num = input_data.user_num
        self.item_num = input_data.anchor_num
        self.define_model(configs)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        logging.info("Start training")
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)
        logging.info("Training complete and saving...")
        saver.save(sess, path.join(configs["save_path"], configs["model_name"]))

    def recommend(self, sess, max_size, model_path=None, model_name=None):
        if self.loss is None:
            if model_path is None:
                logging.error("Saving path of matrix factorization model should be given.")
                return
            saver = tf.train.import_meta_graph(path.join(model_path, model_name))
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            graph = tf.get_default_graph()
            self.user_idx = graph.get_tensor_by_name("placeholders/user_idx")
            self.item_idx = graph.get_tensor_by_name("placeholders/item_idx")
            self.pred = graph.get_tensor_by_name("loss/prediction")
            self.user_num = tf.shape(graph.get_tensor_by_name("embedding/user_bias"))[0]
            self.item_num = tf.shape(graph.get_tensor_by_name("embedding/item_bias"))[0]

        rec_dict = dict()
        item_list = np.array(range(self.item_num))
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            feed_dict = {self.user_idx: np.array([user_id] * self.item_num), self.item_idx: item_list}
            rec_score = sess.run(self.pred, feed_dict=feed_dict)
            rec_item_list = sorted(range(self.item_num), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]
        return rec_dict
