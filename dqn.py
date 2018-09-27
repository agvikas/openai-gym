import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
    def __init__(self, env, parameters):
        self.env = env
        self.memory = deque(maxlen=1500)
        self.epsilon_decay = parameters['epsilon_decay']
        self.epsilon = parameters['epsilon']
        self.gamma = parameters['gamma']
        self.learning_rate = parameters['learning_rate']
        self.batch_size = parameters['batch_size']

    def build_model(self):

        init = tf.truncated_normal_initializer(mean=0, stddev=2)
        self.input = tf.placeholder(dtype=tf.float32, shape=(1, self.env.observation_space.shape[0]), name='input')

        layer1 = tf.layers.dense(inputs=self.input, units=50, kernel_initializer=init, activation=tf.nn.relu,
                                 name='function_layer1')

        layer2 = tf.layers.dense(inputs=layer1, units=50, kernel_initializer=init, activation=tf.nn.relu,
                                 name='function_layer2')

        self.output = tf.layers.dense(inputs=layer2, units=self.env.action_space.n,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                                      name='output')

        self.target_value = tf.placeholder(dtype=tf.float32, shape=(1, self.env.action_space.n), name='target_value')
        self.loss = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.output)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        self.train_op = optimizer.minimize(self.loss)

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, sess, state):

        q_values, = sess.run([self.output], feed_dict={self.input: state})
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(np.squeeze(q_values))
        return q_values, action

    #on-policy learning
    def replay(self, sess):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            q_future, next_action = self.act(sess, next_state)
            if done:
                q_future[0][next_action] = reward
            else:
                q_future[0][next_action] = reward + max(max(q_future))*self.gamma
            _, = sess.run([self.train_op], feed_dict={self.input: state, self.target_value:q_future})

    #off-policy learning
    def replay_off(self, sess):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            q_future, next_action = self.act(sess, next_state)
            if done:
                q_future[0][action] = reward
            else:
                q_future[0][action] = reward + max(max(q_future))*self.gamma
            _, = sess.run([self.train_op], feed_dict={self.input: state, self.target_value:q_future})


