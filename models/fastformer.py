from tensorflow.keras.layers import Dense, Layer

import tensorflow as tf
import numpy as np

class Fastformer(Layer):
    def __init__(self, dim, **kwargs):
        super(Fastformer, self).__init__()
        self.dim = dim
        #self.decode_dim = decode_dim
        
    def build(self, input_shape):
        self.weight_query = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.weight_key = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.weight_value = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.weight_r = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.scale_factor = self.dim ** -0.5
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        query = self.weight_query(x)
        key = self.weight_key(x)
        value = self.weight_value(x)
        b, s, d = query.shape

        alpha_weight = self.softmax(query * self.scale_factor)
        global_query = query * alpha_weight
        global_query = tf.einsum('bsd->bd', global_query)

        repeat_global_query = tf.tile(tf.expand_dims(global_query, axis=1), [1, s, 1])
        p = repeat_global_query * key
        beta_weight = self.softmax(p * self.scale_factor)
        global_key = p * beta_weight
        global_key = tf.einsum('bsd->bd', global_key)

        repeat_global_key = tf.tile(tf.expand_dims(global_key, axis=1), [1, s, 1])
        u = repeat_global_key * value
        r = self.weight_r(u)        

        result = query + r
        return result


