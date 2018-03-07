# coding: utf-8
import tensorflow as tf
import numpy as np
import logging
from .DataInterface import DataInterface
from .constants import *
from os import path


class ProbRRDataProvider(DataInterface):
    def __init__(self):
        super(ProbRRDataProvider, self).__init__()
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
        return 1 / (1 + np.exp(- 0.01 * watch_time))


class ProbRateRegression(object):
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
        self.class_num = None
        self.means = None
        self.stds = None

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
        self.emb_init_value = configs.get('emb_init_value', 1)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'sgd')
        self.lambda_value = configs.get("lambda_value", 0.0001)
        self.keep_prob_value = configs.get("keep_prob_value", 1.0)
        self.class_num = configs.get('class_num', 5)
        self.means_init = np.array(configs.get('means', [0, 0, 0, 0, 0]), dtype=np.float32)
        self.stds_init = np.array(configs.get('stds', [1, 1, 1, 1, 1]), dtype=np.float32)
        self.prob_init = np.array(configs.get("prob", [0.2, 0.2, 0.2, 0.2, 0.2]), dtype=np.float32)
        assert(self.means_init.shape[0] == self.stds_init.shape[0] == self.class_num)

    def define_model(self, configs):

        self._parse_config(configs)

        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="item_idx")
            self.user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name="user_item_score")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
            self.means = tf.get_variable("means", initializer=self.means_init, dtype=tf.float32, trainable=False)
            self.stds = tf.get_variable("stds", initializer=self.stds_init, dtype=tf.float32, trainable=False)
            self.prob = tf.get_variable("prob", initializer=self.prob_init, dtype=tf.float32, trainable=False)

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
                                                               kernel_initializer=initializer)

            logits = tf.layers.dense(parameters["h" + str(len(configs["layers"]))], self.class_num, activation=None, name="logits")
            self.softmax = tf.nn.softmax(logits=logits, name="softmax")

        with tf.name_scope("loss"):
            # self.cond_prob = tf.square(tf.reshape(self.user_item_score, (-1, 1)) - self.means) / (2 * tf.square(self.stds)) + 2 * tf.log(self.stds)
            # min_label = tf.argmin(self.cond_prob, axis=1)
            # self.loss = tf.losses.sparse_softmax_cross_entropy(labels=min_label, logits=self.softmax)
            self.cond_prob = tf.exp(-tf.square(tf.reshape(self.user_item_score, (-1, 1)) - self.means) / (2 * tf.square(self.stds))) / self.stds * self.prob
            self.cond_prob = self.cond_prob / tf.reduce_sum(self.cond_prob, axis=1, keep_dims=True)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.cond_prob, logits=self.softmax))
            self.loss += self.lambda_value * (tf.add_n([tf.reduce_mean(tf.square(user_emb)), tf.reduce_mean(tf.square(item_emb))])) + tf.losses.get_regularization_loss()

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
        prob_mean = np.zeros(shape=(self.class_num, ))
        prob_y_2 = np.zeros(shape=(self.class_num, ))
        prob_sum = np.zeros(shape=(self.class_num, ))
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, softmax, cond_prob, _ = sess.run([self.loss, self.softmax, self.cond_prob, self.optimizer],
                                                         feed_dict=self.create_feed_dict(input_batch))
            total_loss += loss_batch
            prob_mean += input_batch["user_item_score"].T.dot(softmax)
            prob_y_2 += np.sum(np.multiply(np.square(input_batch["user_item_score"].reshape((-1, 1))), softmax), axis=0)
            prob_sum += np.sum(softmax, axis=0)
            if i % self.display_step == 0:
                logging.info('Average loss at epoch {} step {}: {:5.6f}'
                             .format(epoch_idx, i, total_loss / self.display_step))
                logging.info("softmax = {}".format(np.sum(softmax, axis=0)))
                total_loss = 0.0

        self.means = tf.convert_to_tensor(prob_mean / (prob_sum + 0.000001))
        self.stds = tf.convert_to_tensor(np.sqrt(prob_y_2 / (prob_sum + 0.000001) - np.square(self.means.eval())))
        self.prob = tf.convert_to_tensor(prob_sum / np.sum(prob_sum))
        logging.info("means = {}, stds = {}, prob = {}".format(self.means.eval(), self.stds.eval(), self.prob.eval()))

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
            self.softmax = graph.get_tensor_by_name("forward_NN/softmax")
            self.user_num = tf.shape(graph.get_tensor_by_name("embedding/user_emb"))[0]
            self.item_num = tf.shape(graph.get_tensor_by_name("embedding/item_emb"))[0]

        rec_dict = dict()
        item_list = np.array(range(self.item_num))
        for user_id in range(self.user_num):
            logging.info("recommend for user {}".format(user_id))
            feed_dict = {self.user_idx: np.array([user_id] * self.item_num), self.item_idx: item_list, self.keep_prob: 1.0}
            softmax = sess.run(self.softmax, feed_dict=feed_dict)
            rec_score = np.sum(np.multiply(softmax, self.means.eval()), axis=1)
            rec_item_list = sorted(range(self.item_num), key=lambda x: rec_score[x], reverse=True)[0:max_size]
            rec_dict[user_id] = [(x, float(rec_score[x])) for x in rec_item_list]
        return rec_dict
