# coding: utf-8
import tensorflow as tf
import numpy as np
import logging
from .mf import MFDataProvider
from .constants import *


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
        self.item_idx = None
        self.user_item_score = None
        self.keep_prob = None
        self.loss = None
        self.pred = None
        self.optimizer = None

    def _parse_config(self, configs):
        self.user_embedding_size = configs['user_embedding_size']
        self.item_embedding_size = configs['item_embedding_size']
        self.learning_rate = configs.get('learning_rate', 0.01)
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('batch_size', 2)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'sgd')
        self.lambda_value = configs.get("lambda_value", 0.0001)
        self.keep_prob_value = configs.get("keep_prob_value", 1.0)

    def define_model(self, configs):
        self._parse_config(configs)
        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="item_idx")
            self.user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name="user_item_score")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

        with tf.name_scope("embedding"):
            user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.user_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb"
            )
            item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.item_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb"
            )

        parameters = dict()
        with tf.name_scope("forward_NN"):
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.lambda_value)
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)

            user_emb = tf.nn.embedding_lookup(user_emb_matrix, self.user_idx)
            item_emb = tf.nn.embedding_lookup(item_emb_matrix, self.item_idx)
            parameters["h0"] = tf.concat([user_emb, item_emb], 1)

            for i in range(len(configs["layers"])):
                parameters["h" + str(i + 1)] = tf.layers.dense(parameters["h" + str(i)],
                                                               configs["layers"][i],
                                                               activation=configs["activation"],
                                                               kernel_regularizer=regularizer,
                                                               bias_regularizer=regularizer,
                                                               kernel_initializer=initializer)
                parameters["h" + str(i + 1)] = tf.nn.dropout(parameters["h" + str(i + 1)], keep_prob=self.keep_prob)

        with tf.name_scope("loss"):
            self.pred = tf.layers.dense(parameters["h" + str(len(configs["layers"]))], 1,
                                        activation=None,
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        kernel_initializer=initializer,
                                        name="prediction")
            self.loss = tf.losses.mean_squared_error(tf.reshape(self.user_item_score, [-1, 1]), self.pred) + tf.losses.get_regularization_loss()
            self.loss += self.lambda_value * (tf.add_n([tf.reduce_mean(tf.square(user_emb)), tf.reduce_mean(tf.square(item_emb))]))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"],
            self.keep_prob: self.keep_prob_value,
        }

    def run_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, pred, _ = sess.run([self.loss, self.pred, self.optimizer],
                                           feed_dict=self.create_feed_dict(input_batch))
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
            self.pred = graph.get_tensor_by_name("loss/prediction")
            self.user_num = tf.shape(graph.get_tensor_by_name("embedding/user_emb"))[0]
            self.item_num = tf.shape(graph.get_tensor_by_name("embedding/item_emb"))[0]

        rec_dict = dict()
        item_list = np.array(range(self.item_num))
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            feed_dict = {self.user_idx: np.array([user_id] * self.item_num), self.item_idx: item_list, self.keep_prob: 1.0}
            rec_score = sess.run(self.pred, feed_dict=feed_dict)
            rec_item_list = sorted(range(self.item_num), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]
        return rec_dict
