# coding: utf-8
import logging
from os import path
import numpy as np
import scipy.sparse as sp
from .constants import *
import tensorflow as tf


class TaggingTagRateRegression(object):
    def __init__(self):
        # model configuration parameters
        self.user_num = None
        self.item_num = None
        self.label_num = None
        self.embedding_size = None
        self.emb_init_value = None
        self.display_step = None
        self.training_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.optimize_method = None

        ## regularization parameters
        self.nn_lambda = None
        self.emb_lambda = None
        self.label_lambda = None
        self.tagging_lambda = None

        # model variables
        self.user_idx = None
        self.pos_item_idx = None
        self.neg_item_idx = None
        self.item_emb_matrix = None
        self.user_emb_matrix = None

        self.user_label = None
        self.pos_item_label = None
        self.neg_item_label = None
        self.user_label_mask = None
        self.pos_item_label_mask = None
        self.neg_item_label_mask = None

        self.loss = None
        self.tagging_pred = None
        self.tagging_loss = None
        self.optimizer = None
        self.tagging_optimizer = None

    def _parse_config(self, configs):
        self.user_embedding_size = configs["user_embedding_size"]
        self.item_embedding_size = configs["item_embedding_size"]
        self.learning_rate = configs.get('learning_rate', 0.01)
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('emb_init_value', 1)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'adam')
        self.emb_lambda = configs.get("emb_lambda", 0.0001)
        self.label_lambda = configs.get("label_lambda", 1)
        self.tagging_lambda = configs.get("tagging_lambda", 1)

    def bpr_nn(self, user_idx, item_idx, configs, reuse=False):
        parameters = dict()
        with tf.name_scope("bpr_NN"):
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.nn_lambda)
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            user_emb = tf.nn.embedding_lookup(self.user_emb_matrix, user_idx)
            item_emb = tf.nn.embedding_lookup(self.item_emb_matrix, item_idx)
            parameters["h0"] = tf.concat([user_emb, item_emb], 1)
            for i in range(len(configs["bpr_layers"])):
                parameters["h" + str(i + 1)] = tf.layers.dense(parameters["h" + str(i)],
                                                               configs["bpr_layers"][i],
                                                               activation=configs["bpr_activation"],
                                                               kernel_regularizer=regularizer,
                                                               kernel_initializer=initializer,
                                                               name="h" + str(i + 1),
                                                               reuse=reuse)
            pred = tf.layers.dense(parameters["h" + str(len(configs["bpr_layers"]))], 1,
                                   activation=None,
                                   kernel_regularizer=regularizer,
                                   kernel_initializer=initializer,
                                   name="prediction",
                                   reuse=reuse)


            return user_emb, item_emb, pred

    def label_nn(self, embedding_vec, configs, reuse=False):
        parameters = dict()
        with tf.name_scope("label_NN"):
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.nn_lambda)
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            parameters["l0"] = embedding_vec
            for i in range(len(configs["label_layers"])):
                parameters["l" + str(i + 1)] = tf.layers.dense(parameters["l" + str(i)],
                                                               configs["label_layers"][i],
                                                               activation=configs["label_activation"],
                                                               kernel_regularizer=regularizer,
                                                               kernel_initializer=initializer,
                                                               name="l" + str(i + 1),
                                                               reuse=reuse)
            label_logit = tf.layers.dense(parameters["l" + str(len(configs["label_layers"]))],
                                          self.label_num,
                                          activation=tf.nn.tanh,
                                          name="label_logits",
                                          reuse=reuse)
            return label_logit

    def define_model(self, configs):
        self._parse_config(configs)

        # placeholder layers and embedding matrices
        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="pos_item_idx")
            self.neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="neg_item_idx")
            self.user_label = tf.placeholder(dtype=tf.float32, shape=[None, self.label_num], name="user_labels")
            self.pos_item_label = tf.placeholder(dtype=tf.float32, shape=[None, self.label_num], name="pos_item_label")
            self.neg_item_label = tf.placeholder(dtype=tf.float32, shape=[None, self.label_num], name="neg_item_label")
            self.user_label_mask = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="user_labels_mask")
            self.item_label_mask = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="item_labels_mask")

        with tf.name_scope("embedding_matrix"):
            self.user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb")

            self.item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb")

        user_emb, pos_item_emb, self.pos_pred = self.bpr_nn(user_idx=self.user_idx, item_idx=self.pos_item_idx, configs=configs)
        user_emb, neg_item_emb, self.neg_pred = self.bpr_nn(user_idx=self.user_idx, item_idx=self.neg_item_idx, configs=configs, reuse=True)

        user_label_logit = self.label_nn(embedding_vec=user_emb, configs=configs)
        pos_item_label_logit = self.label_nn(embedding_vec=pos_item_emb, configs=configs, reuse=True)
        neg_item_label_logit = self.label_nn(embedding_vec=neg_item_emb, configs=configs, reuse=True)

        def label_loss(truth, pred, mask):
            return tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=pred), mask)

        with tf.name_scope("loss"):
            user_label_loss = tf.reduce_mean(label_loss(truth=self.user_label,
                                                        pred=user_label_logit,
                                                        mask=self.user_label_mask))
            pos_item_label_loss = tf.reduce_mean(label_loss(truth=self.pos_item_label,
                                                            pred=pos_item_label_logit,
                                                            mask=self.pos_item_label_mask))
            neg_item_label_loss = tf.reduce_mean(label_loss(truth=self.neg_item_label,
                                                            pred=neg_item_label_logit,
                                                            mask=self.neg_item_label_mask))

            self.loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.pos_pred - self.neg_pred))) \
                        + user_label_loss + pos_item_label_loss + neg_item_label_loss \
                        + tf.losses.get_regularization_loss()
            self.loss += self.emb_lambda * tf.reduce_mean(tf.add_n([tf.square(user_emb),
                                                                    tf.square(pos_item_emb),
                                                                    tf.square(neg_item_emb)]))

            # self.tagging_loss = self.tagging_lambda * tf.reduce_mean(tf.reduce_sum(
            #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tagging, logits=self.tagging_pred), axis=1))
            # self.tagging_loss += embedding_reg

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            # self.tagging_optimizer = tf.train.AdamOptimizer().minimize(self.tagging_loss)

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"],
            self.user_labels: input_batch["user_labels"],
            self.item_labels: input_batch["item_labels"],
            self.item_labels_mask: input_batch["item_labels_mask"],
            self.user_labels_mask: input_batch["user_labels_mask"],
        }

    def create_tagging_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"],
            self.tagging: input_batch["tagging"],
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

    def run_tagging_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, _ = sess.run([self.tagging_loss, self.tagging_optimizer],
                                     feed_dict=self.create_tagging_feed_dict(input_batch))
            total_loss += loss_batch
            if i % self.display_step == 0:
                logging.info('Average loss at epoch {} step {}: {:5.6f}'
                             .format(epoch_idx, i, total_loss / self.display_step))
                total_loss = 0.0

    def fit(self, sess, input_data, configs):
        self.user_num = input_data.user_num
        self.item_num = input_data.anchor_num
        self.label_num = input_data.label_num
        parameters = self.define_model(configs)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        logging.info("Start training")
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            tagging_batch_gen = input_data.tagging_batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)
            self.run_tagging_epoch(sess, i, tagging_batch_gen)
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
