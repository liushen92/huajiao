# coding: utf-8
import tensorflow as tf
from .DataInterface import DataInterface
from .constants import *
from os import path
import numpy as np
import logging


class BPRDataProvider(DataInterface):
    def __init__(self):
        super(BPRDataProvider, self).__init__()
        self.pos_samples_size = None
        self.user_pos_item_set = None
        self.load_data(path.join(data_dir, "train_data"))
        self.find_pos_items(self.user_anchor_behavior)

    def batch_generator(self, batch_size, neg_samples_num=5):
        self.pairs_array = np.zeros(shape=(self.pos_samples_size * neg_samples_num, 3), dtype=np.int32)
        start_line = 0
        for user_idx in np.random.permutation(range(self.user_num)):
            pairs_list, pairs_list_length = self.sample_neg_items(user_idx=user_idx, neg_samples_num=neg_samples_num)
            self.pairs_array[start_line : start_line + pairs_list_length, :] = pairs_list
            start_line += pairs_list_length

        for i in range(0, len(self.pairs_array), batch_size):
            batch_data = dict()
            start_idx = i
            end_idx = min(i + batch_size, len(self.pairs_array))
            batch_data['user_idx'] = self.pairs_array[start_idx: end_idx, 0]
            batch_data['pos_item_idx'] = self.pairs_array[start_idx: end_idx, 1]
            batch_data['neg_item_idx'] = self.pairs_array[start_idx: end_idx, 2]
            yield batch_data

    def find_pos_items(self, user_anchor_behavior):
        self.user_pos_item_set = dict()
        self.pos_samples_size = 0
        for user_id in user_anchor_behavior:
            self.user_pos_item_set[user_id] = set()
            for anchor_id in user_anchor_behavior[user_id]:
                if user_anchor_behavior[user_id][anchor_id][0] >= 60:
                    self.user_pos_item_set[user_id].add(anchor_id)
            self.pos_samples_size += len(self.user_pos_item_set[user_id])

    def sample_neg_items(self, user_idx, neg_samples_num):
        pairs_list_length = len(self.user_pos_item_set[user_idx]) * neg_samples_num
        pairs_list = np.zeros(shape=(pairs_list_length, 3))
        candidate_items = list(set(range(self.anchor_num)) - self.user_pos_item_set[user_idx])
        pairs_list[:, 0] = user_idx
        pairs_list[:, 1] = list(self.user_pos_item_set[user_idx]) * neg_samples_num
        pairs_list[:, 2] = np.random.choice(candidate_items, pairs_list_length)

        return pairs_list, pairs_list_length


class BPR(object):
    def __init__(self):
        # model configuration parameters
        self.user_num = None
        self.item_num = None
        self.embedding_size = None
        self.lambda_value = None
        self.batch_size = None
        self.training_epochs = None
        self.display_step = None
        self.emb_init_value = None

        # model variables
        self.user_idx = None
        self.pos_item_idx = None
        self.neg_item_idx = None
        self.user_emb_matrix = None
        self.item_emb_matrix = None
        self.loss = None
        self.optimizer = None
        self.score = None

    def _parse_config(self, configs):
        self.embedding_size = configs['embedding_size']
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('emb_init_value', 1)
        self.display_step = configs.get('display_step', 100)
        self.lambda_value = configs.get("lambda_value", 0.001)

    def define_model(self, configs):
        self._parse_config(configs)
        with tf.name_scope("placeholder"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="pos_item_idx")
            self.neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="neg_item_idx")

        with tf.name_scope("embedding_matrix"):
            self.user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb"
            )
            self.item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb"
            )

        with tf.name_scope("bpr"):
            u = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_idx)
            i = tf.nn.embedding_lookup(self.item_emb_matrix, self.pos_item_idx)
            j = tf.nn.embedding_lookup(self.item_emb_matrix, self.neg_item_idx)

        with tf.name_scope("loss"):
            self.loss = - tf.reduce_mean(tf.log(tf.sigmoid(
                tf.reduce_sum(tf.multiply(u, i), axis=1) - tf.reduce_sum(tf.multiply(u, j), axis=1))))
            self.loss = self.loss + self.lambda_value * tf.reduce_mean(
                tf.add_n([tf.square(u), tf.square(i), tf.square(j)]))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        with tf.name_scope("prediction"):
            self.score = tf.reduce_sum(tf.multiply(u, i), axis=1)

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.pos_item_idx: input_batch["pos_item_idx"],
            self.neg_item_idx: input_batch["neg_item_idx"],
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
        rec_dict = dict()
        item_list = np.array(range(self.item_num))
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            feed_dict = {self.user_idx: np.array([user_id] * self.item_num), self.pos_item_idx: item_list}
            rec_score = sess.run(self.score, feed_dict=feed_dict)
            rec_item_list = sorted(range(self.item_num), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]
        return rec_dict
