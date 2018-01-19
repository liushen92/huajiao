# coding: utf-8
import tensorflow as tf
import logging
import model.mf.MFDataProvider as MFDataProvider
from os import path
from model.utils import utils


class MatrixFactorization(object):
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
        self.loss = None
        self.optimizer = None

    def _parse_config(self, configs):
        self.embedding_size = configs['embedding_size']
        self.learning_rate = configs.get('learning_rate', 0.01)
        self.training_epochs = configs.get('training_epochs', 10)
        self.batch_size = configs.get('batch_size', 128)
        self.emb_init_value = configs.get('batch_size', 2)
        self.lambda_value = configs.get('lambda_value', 0.0001)
        self.display_step = configs.get('display_step', 100)
        self.optimize_method = configs.get('optimize_method', 'sgd')

    def define_model(self, configs):
        self._parse_config(configs)

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
                                                 - tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=1)
                                                 - user_bias
                                                 - item_bias)
                                       + self.lambda_value * (tf.add_n([tf.reduce_mean(tf.square(user_emb)),
                                                                        tf.reduce_mean(tf.square(item_emb)),
                                                                        tf.reduce_mean(tf.square(user_bias)),
                                                                        tf.reduce_mean(tf.square(item_bias))])))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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

    def fit(self, sess, input_data, configs):
        self.user_num = input_data.user_num
        self.item_num = input_data.item_num
        user_emb_matrix, user_bias_matrix, item_emb_matrix, item_bias_matrix = self.define_model(configs)
        sess.run(tf.global_variables_initializer())
        logging.info("Start training")
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)

        logging.info("Training complete and saving...")
        user_emb_matrix, user_bias_matrix, item_emb_matrix, item_bias_matrix = sess.run([user_emb_matrix,
                                                                                         user_bias_matrix,
                                                                                         item_emb_matrix,
                                                                                         item_bias_matrix])
        utils.save_matrix(user_emb_matrix, path.join(configs['save_path'], "user-emb-matrix"))
        utils.save_matrix(item_emb_matrix, path.join(configs['save_path'], "item-emb-matrix"))
        utils.save_matrix(user_bias_matrix, path.join(configs['save_path'], "user-bias-matrix"))
        utils.save_matrix(item_bias_matrix, path.join(configs['save_path'], "item-bias-matrix"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S', )
    input_data = MFDataProvider.MFDataProvider(path.join("..", "..", "data", "train_data"),
                                               path.join("..", "..", "tmp"))
    model = MatrixFactorization()

    from . import configs
    configs['save_path'] = path.join("..", "..", "tmp")
    with tf.Session() as sess:
        model.fit(sess=sess, input_data=input_data, configs=configs.configs)
