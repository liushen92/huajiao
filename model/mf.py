# coding: utf-8
import tensorflow as tf


class MatrixFactorization(object):
    def __init__(self, user_num, item_num, embedding_size):
        self.embedding_size = embedding_size
        self.user_num = user_num
        self.item_num = item_num
        self.emb_init_value = 2
        self.lambda_value = 0.0001
        self.learning_rate = 0.01

    def model(self):
        with tf.name_scope("placeholders"):
            user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='user_idx')
            item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='item_idx')
            user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name='user_item_score')

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

        with tf.name_scope("difference"):
            user_emb = tf.nn.embedding_lookup(user_emb_matrix, user_idx, name="user_emb")
            user_bias = tf.nn.embedding_lookup(user_bias_matrix, user_idx, name="user_bias")
            item_emb = tf.nn.embedding_lookup(item_emb_matrix, item_idx, name="item_emb")
            item_bias = tf.nn.embedding_lookup(item_bias_matrix, item_idx, name="item_bias")

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(user_item_score
                                            - tf.reduce_sum(tf.multiply(user_emb, item_emb))
                                            - user_bias
                                            - item_bias)
                                  + self.lambda_value * (tf.add_n(tf.square(user_emb), tf.square(item_emb), tf.square(user_bias), tf.square(item_bias))))

        with tf.name_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        return user_emb_matrix, user_bias_matrix, item_emb_matrix, item_bias_matrix, loss, optimizer

    def fit(self, input_data):
        user_emb_matrix, user_bias_matrix, item_emb_matrix, item_bias_matrix, loss, optimizer = self.model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(5000):
                loss_batch, _ = tf.run([loss, optimizer], feed_dict={})