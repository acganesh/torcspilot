""" Model architectures for Actor and Critic networks
"""

import tensorflow as tf
import keras.backend as K

from keras.initializations import normal
from keras.layers import Dense, Input, merge
from keras.optimizers import Adam
from keras.models import Model

class ActorFCNet():
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        # Perhaps make this into a dict -- would be nicer
        self.HIDDEN1_UNITS = 300
        self.HIDDEN2_UNITS = 600

        K.set_session(sess)

        # Create the model
        self.model, self.weights, self.state = self.build_model(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.build_model(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def train_target_net(self):
        """
        Train a target network that gradually incorporates
        changes in weights, parametrized by a smoothing factor TAU.
        (TODO: cite paper).
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in xrange(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)


    def build_model(self, state_size, action_size):
        S = Input(shape=[state_size])
        h0 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Acceleration = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Brake = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        V = merge([Steering, Acceleration, Brake], mode='concat')
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S 


class CriticFCNet():
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        self.HIDDEN1_UNITS = 300
        self.HIDDEN2_UNITS = 600

        K.set_session(sess)

        # Create the model
        self.model, self.actions, self.state = self.build_model(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.build_model(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.actions)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.actions: actions
        })[0]

    def train_target_net(self):
        """
        Train a target network that gradually incorporates
        changes in weights, parametrized by a smoothing factor TAU.
        (TODO: cite paper).
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in xrange(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def build_model(self, state_size, action_dim):
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(self.HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(self.HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1, a1], mode='sum')
        h3 = Dense(self.HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim, activation='linear')(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr = self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
