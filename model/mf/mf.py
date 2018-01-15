# coding: utf-8
import tensorflow as tf
import logging
from . import MFDataProvider
from os import path


class MatrixFactorization(object):
    def __init__(self, user_num, item_num, embedding_size):
        self.embedding_size = embedding_size
        self.user_num = user_num
        self.item_num = item_num
        self.emb_init_value = 2
        self.lambda_value = 0.0001
        self.learning_rate = 0.01
        self.display_step = 100
        self.training_epochs = 10
        self.batch_size = 64

        # model parameters
        self.user_item_score = None
        self.user_idx = None
        self.item_idx = None
        self.loss = None
        self.optimizer = None

    def define_model(self):
        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='user_idx')
            self.item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='item_idx')
            self.user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name='user_item_score')

        with tf.name_scope("embedding_matrix"):
            user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb_w"
            )
            user_bias_matrix = tf.Variable(
                tf.random_uniform([self.user_num, 1],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb_b"
            )
            item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb_w"
            )
            item_bias_matrix = tf.Variable(
                tf.random_uniform([self.item_num, 1],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb_b"
            )

        with tf.name_scope("loss"):
            user_emb = tf.nn.embedding_lookup(user_emb_matrix, self.user_idx, name="user_emb")
            user_bias = tf.nn.embedding_lookup(user_bias_matrix, self.user_idx, name="user_bias")
            item_emb = tf.nn.embedding_lookup(item_emb_matrix, self.item_idx, name="item_emb")
            item_bias = tf.nn.embedding_lookup(item_bias_matrix, self.item_idx, name="item_bias")
            self.loss = tf.reduce_mean(tf.square(self.user_item_score
                                                 - tf.reduce_sum(tf.multiply(user_emb, item_emb))
                                                 - user_bias
                                                 - item_bias)
                                       + self.lambda_value * (tf.add_n(tf.square(user_emb),
                                                                       tf.square(item_emb),
                                                                       tf.square(user_bias),
                                                                       tf.square(item_bias))))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        return user_emb_matrix, user_bias_matrix, item_emb_matrix, item_bias_matrix

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

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"]
        }

    def fit(self, sess, input_data):
        user_emb_matrix, user_bias_matrix, item_emb_matrix, item_bias_matrix = self.define_model()
        sess.run(tf.global_variables_initializer())
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {0}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)


if __name__ == "__main__":
    input_data = MFDataProvider.MFDataProvider(path.join("..", "..", "data", "train_data"))
    model = MatrixFactorization(input_data.user_num, input_data.item_num, 100)
    with tf.Session() as sess:
        model.fit(sess=sess, input_data=input_data)
