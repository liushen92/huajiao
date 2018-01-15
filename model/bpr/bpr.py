import tensorflow as tf
import logging
from os import path
from utils import Saver


class BPRmodel(object):
    def __init__(self, params):
        self.learning_rate = params["learning_rate"]
        self.batch_size = params["batch_size"]
        self.embedding_size = params["embedding_size"]
        self.emb_init_value = params["emb_init_value"]
        self.reg_params = params["reg_params"]
        self.training_epochs = params["training_epochs"]
        self.display_step = params["display_step"]
        self.log_dir = params["log_dir"]
        self.save_path = params["save_path"]
        self.optimizer_type = params["optimizer"]

        self.input_data = None

        self._loss = None
        self._optimizer = None
        self._user_emb_matrix = None
        self._item_emb_matrix = None
        self._item_bias_matrix = None
        self._user_idx = None
        self._pos_item_idx = None
        self._neg_item_idx = None

    @property
    def loss(self):
        if self._loss is None:
            self._loss = self.add_loss_op()
        return self._loss

    @property
    def optimizer(self):
        if self._optimizer is None:
            if self.optimizer_type == "sgd":
                self._optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "adam":
                self._optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self._optimizer = tf.train.MomentumOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "RMS":
                self._optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        return self._optimizer

    @property
    def user_emb_matrix(self):
        if self._user_emb_matrix is None:
            self._user_emb_matrix, self._item_emb_matrix, self._item_bias_matrix = self.add_embedding()
        return self._user_emb_matrix

    @property
    def item_emb_matrix(self):
        if self._item_emb_matrix is None:
            self._user_emb_matrix, self._item_emb_matrix, self._item_bias_matrix = self.add_embedding()
        return self._item_emb_matrix

    @property
    def item_bias_matrix(self):
        if self._item_bias_matrix is None:
            self._user_emb_matrix, self._item_emb_matrix, self._item_bias_matrix = self.add_embedding()
        return self._item_bias_matrix

    @property
    def user_idx(self):
        if self._user_idx is None:
            self._user_idx, self._pos_item_idx, self._neg_item_idx = self.add_placeholders()
        return self._user_idx

    @property
    def pos_item_idx(self):
        if self._pos_item_idx is None:
            self._user_idx, self._pos_item_idx, self._neg_item_idx = self.add_placeholders()
        return self._pos_item_idx

    @property
    def neg_item_idx(self):
        if self._neg_item_idx is None:
            self._user_idx, self._pos_item_idx, self._neg_item_idx = self.add_placeholders()
        return self._neg_item_idx

    def add_placeholders(self):
        """ Add Placeholders.
        Create placeholders for holding a batch of (user_idx, pos_item_idx, neg_item_idx).

        :return: user_idx, pos_item_idx, neg_item_idx
        """
        with tf.name_scope("placeholders"):
            user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='user_idx')
            pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='pos_item_idx')
            neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_idx')
        logging.debug("Placeholder added.")
        return user_idx, pos_item_idx, neg_item_idx

    def add_embedding(self):
        """ Add embedding Layers.
        With given input_data (which has user and item mappings), create embedding layers.

        :return: user_emb_matrix, item_emb_matrix
        """
        if self.input_data is None:
            raise ValueError("Must provide input_data before creating embedding layers.")

        with tf.name_scope("embedding_matrix"):
            user_emb_matrix = tf.Variable(
                tf.random_uniform([self.input_data.user_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb_w"
            )
            item_emb_matrix = tf.Variable(
                tf.random_uniform([self.input_data.item_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb_w"
            )
            item_bias_matrix = tf.Variable(
                tf.random_uniform([self.input_data.item_num, 1],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb_b"
            )
        logging.debug("Embedding layers added.")
        return user_emb_matrix, item_emb_matrix, item_bias_matrix

    def add_loss_op(self):
        """ Create loss op.
        loss = - \sum{log(sig(x_uij))} + regularization, where x_uij stands for the difference between user u's utility
        of positive item i and negative item j. Thus the larger the difference is, the smaller the loss will be.

        :return: loss
        """
        with tf.name_scope("difference"):
            user_emb = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_idx,
                                              name="user_emb")
            pos_item_emb = tf.nn.embedding_lookup(self.item_emb_matrix, self.pos_item_idx,
                                                  name="pos_item_emb")
            neg_item_emb = tf.nn.embedding_lookup(self.item_emb_matrix, self.neg_item_idx,
                                                  name="neg_item_emb")
            pos_item_bias = tf.nn.embedding_lookup(self.item_bias_matrix, self.pos_item_idx,
                                                   name="pos_item_bias")
            neg_item_bias = tf.nn.embedding_lookup(self.item_bias_matrix, self.neg_item_idx,
                                                   name="neg_item_bias")
            x_uij = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb - neg_item_emb), 1, keep_dims=True) \
                    + pos_item_bias - neg_item_bias

        reg1, reg2, reg3 = self.reg_params

        with tf.name_scope("loss"):
            loss = - tf.reduce_mean(tf.log(tf.sigmoid(x_uij))) \
                   + tf.reduce_sum(tf.add_n([reg1 * tf.square(user_emb),
                                   reg2 * tf.square(pos_item_emb),
                                   reg3 * tf.square(neg_item_emb)]))
        logging.debug("Loss ops added.")
        return loss

    def create_feed_dict(self, input_batch):
        """
        :param input_batch:
        :return:
        """
        return {
            self.user_idx: input_batch['user_idx'],
            self.pos_item_idx: input_batch['pos_item_idx'],
            self.neg_item_idx: input_batch['neg_item_idx']
        }

    def run_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, _ = sess.run([self.loss, self.optimizer],
                                     feed_dict=self.create_feed_dict(input_batch))
            total_loss += loss_batch
            if i % self.display_step == 0:
                print('Average loss at epoch {} step {}: {:5.6f}'.format(epoch_idx, i, total_loss / self.display_step))
                total_loss = 0.0

    def fit(self, sess, input_data):
        # load input_data & initialize embedding matrix
        self.input_data = input_data
        init_emb = tf.variables_initializer([self.user_emb_matrix, self.item_emb_matrix, self.item_bias_matrix],
                                            name='init_emb')
        sess.run(init_emb)
        logging.debug("Embedding parameters initialized.")
        saver = tf.train.Saver()
        # run epochs_num epochs and save it.
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {0}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)
            saver.save(sess, path.join(self.log_dir, "bpr_model"), global_step=i)

        user_emb_matrix, item_emb_matrix, item_bias_matrix = sess.run([self.user_emb_matrix,
                                                                       self.item_emb_matrix,
                                                                       self.item_bias_matrix])
        Saver.save_matrix(user_emb_matrix, path.join(self.save_path, "user-emb-matrix"))
        Saver.save_matrix(item_emb_matrix, path.join(self.save_path, "item-emb-matrix"))
        Saver.save_matrix(item_bias_matrix, path.join(self.save_path, "item-bias-matrix"))