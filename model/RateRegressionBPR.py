# coding: utf-8
import tensorflow as tf
import numpy as np
import logging
from .DataInterface import DataInterface
from .constants import *
from os import path


class RateRegression(object):
    def __init__(self):
        # model configuration parameters
        self.user_num = None
        self.item_num = None
        self.user_embedding_size = None
        self.item_embedding_size = None
        self.emb_init_value = None
        self.display_step = None
        self.training_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.optimize_method = None
        self.lambda_value = None
        self.keep_prob_value = None

        # model variables
        self.user_idx = None
        self.pos_item_idx = None
        self.neg_item_idx = None
        self.user_emb_matrix = None
        self.item_emb_matrix = None
        self.keep_prob = None
        self.loss = None
        self.pos_pred = None
        self.neg_pred = None
        self.optimizer = None

    def _parse_config(self, configs):
        self.user_embedding_size = configs['user_embedding_size']
        self.item_embedding_size = configs['item_embedding_size']
        self.learning_rate = configs.get('learning_rate', 0.01)
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('emb_init_value', 1)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'sgd')
        self.lambda_value = configs.get("lambda_value", 0.001)
        self.keep_prob_value = configs.get("keep_prob_value", 1.0)

    def forward_nn(self, user_idx, item_idx, configs, reuse=False):
        parameters = dict()
        with tf.name_scope("forward_NN"):
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.lambda_value)
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            user_emb = tf.nn.embedding_lookup(self.user_emb_matrix, user_idx)
            item_emb = tf.nn.embedding_lookup(self.item_emb_matrix, item_idx)
            parameters["h0"] = tf.concat([user_emb, item_emb], 1)
            for i in range(len(configs["layers"])):
                parameters["h" + str(i + 1)] = tf.layers.dense(parameters["h" + str(i)],
                                                               configs["layers"][i],
                                                               activation=configs["activation"],
                                                               kernel_regularizer=regularizer,
                                                               kernel_initializer=initializer,
                                                               name="h" + str(i + 1),
                                                               reuse=reuse)
            pred = tf.layers.dense(parameters["h" + str(len(configs["layers"]))], 1,
                                   activation=None,
                                   kernel_regularizer=regularizer,
                                   kernel_initializer=initializer,
                                   name="prediction",
                                   reuse=reuse)

            return user_emb, item_emb, pred

    def define_model(self, configs):
        self._parse_config(configs)
        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="pos_item_idx")
            self.neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="neg_item_idx")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

        with tf.name_scope("embedding"):
            self.user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.user_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb"
            )
            self.item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.item_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb"
            )

        user_emb, pos_item_emb, self.pos_pred = self.forward_nn(user_idx=self.user_idx, item_idx=self.pos_item_idx, configs=configs)
        user_emb, neg_item_emb, self.neg_pred = self.forward_nn(user_idx=self.user_idx, item_idx=self.neg_item_idx, configs=configs, reuse=True)

        with tf.name_scope("loss"):
            self.loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.pos_pred - self.neg_pred))) \
                        + tf.losses.get_regularization_loss()
            self.loss += self.lambda_value * tf.reduce_mean(tf.add_n([tf.square(user_emb),
                                                                      tf.square(pos_item_emb),
                                                                      tf.square(neg_item_emb)]))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def run_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, _ = sess.run([self.loss, self.optimizer],
                                     feed_dict={self.user_idx: input_batch["user_idx"],
                                                self.pos_item_idx: input_batch["pos_item_idx"],
                                                self.neg_item_idx: input_batch["neg_item_idx"]
                                                })
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
        # print(self.pos_pred.eval(feed_dict={self.user_idx: np.array([0]), self.pos_item_idx: np.array([0])}))
        # print(self.neg_pred.eval(feed_dict={self.user_idx: np.array([0]), self.neg_item_idx: np.array([0])}))
        saver = tf.train.Saver()
        logging.info("Start training")
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)
        logging.info("Training complete and saving...")
        saver.save(sess, os.path.join(configs["save_path"], configs["model_name"]))

    def recommend(self, sess, max_size):
        rec_dict = dict()
        item_list = np.array(range(self.item_num))
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            rec_score = sess.run(self.pos_pred, feed_dict={self.user_idx: np.array([user_id] * self.item_num),
                                                       self.pos_item_idx: item_list})
            rec_item_list = sorted(range(self.item_num), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]
        return rec_dict
