# coding: utf-8
import tensorflow as tf
import logging


class RateRegression(object):
    def __init__(self):
        # model configuration parameters
        self.user_num = None
        self.item_num = None
        self.user_embedding_size = None
        self.item_embedding_size = None
        self.emb_init_value = None
        self.display_steps = None
        self.training_epochs = None
        self.batch_size = None

        # model variables
        self.user_idx = None
        self.item_idx = None
        self.user_item_score = None
        self.loss = None
        self.pred = None
        self.optimizer = None

    def define_model(self, configs):
        with tf.name_scope("placeholders"):
            self.user_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="user_idx")
            self.item_idx = tf.placeholder(dtype=tf.int32, shape=[None], name="item_idx")
            self.user_item_score = tf.placeholder(dtype=tf.float32, shape=[None], name="user_item_score")

        with tf.name_scope("embedding_layer"):
            user_emb_matrix = tf.Variable(
                tf.random_uniform([self.user_num, self.user_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="user_emb_w"
            )
            item_emb_matrix = tf.Variable(
                tf.random_uniform([self.item_num, self.item_embedding_size],
                                  -self.emb_init_value, self.emb_init_value),
                name="item_emb_w"
            )

        parameters = dict()
        with tf.name_scope("forward_NN"):
            parameters["h0"] = tf.layers.dense(tf.concat([tf.nn.embedding_lookup(user_emb_matrix, self.user_idx),
                                                          tf.nn.embedding_lookup(item_emb_matrix, self.item_idx)], 1),
                                               configs["layers"][0],
                                               activation=configs["activation"])

            for i in range(len(configs["layers"]) - 1):
                parameters["h" + str(i + 1)] = tf.layers.dense(parameters["h" + str(i)],
                                                               configs["layers"][i + 1],
                                                               activation=configs["activation"])

        with tf.name_scope("prediction"):
            self.pred = tf.layers.dense(parameters["h" + str(configs["layers"][-1])], 1, activation=None, name="prediction")
            self.loss = tf.losses.mean_squared_error(self.user_item_score, self.pred)

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=configs.learning_rate).minimize(self.loss)

    def create_feed_dict(self, input_batch):
        return {
            self.user_idx: input_batch["user_idx"],
            self.item_idx: input_batch["item_idx"],
            self.user_item_score: input_batch["user_item_score"]
        }

    def run_epoch(self, sess, epoch_idx, batch_gen):
        total_loss = 0.0
        i = 0
        for input_batch in batch_gen:
            i += 1
            loss_batch, _ = sess.run([self.loss, self.optimizer],
                                     feed_dict=self.create_feed_dict(input_batch))
            total_loss += loss_batch
            if i % self.display_steps == 0:
                print('Average loss at epoch {} step {}: {:5.6f}'.format(epoch_idx, i, total_loss / self.display_steps))
                total_loss = 0.0

    def fit(self, sess, input_data, configs):
        self.user_num = input_data.user_num
        self.item_num = input_data.item_num
        self.define_model(configs)
        sess.run(tf.global_variables_initializer())
        logging.info("Start training")
        for i in range(1, self.training_epochs + 1):
            logging.info("training epochs {}".format(i))
            batch_gen = input_data.batch_generator(self.batch_size)
            self.run_epoch(sess, i, batch_gen)
        logging.info("Training complete and saving...")

    def recommend_per_user(self, sess, userid):
        itemid_list = list()
        score_list = [(itemid, sess.run(self.pred, feed_dict={self.item_idx: itemid, self.user_idx: userid}))
                      for itemid in itemid_list]
        return sorted(score_list, key=lambda x: x[1], reverse=True)

    def recommend(self, sess):
        rec_dict = dict()
        for userid in range(self.user_num):
            rec_dict[userid] = self.recommend_per_user(sess, userid)
        return rec_dict
