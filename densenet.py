


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, TimeDistributed, InputLayer, Bidirectional, Softmax, BatchNormalization, Concatenate, LayerNormalization, AlphaDropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow_addons.layers import InstanceNormalization
from ctc_classifier import *
tf.compat.v1.enable_eager_execution()


class DenseNetBlock(keras.layers.Layer):

    def __init__(self, n_layers=5, n_filters=20):
        super().__init__()
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []
        self.concatenate_layers = []
        for i in range(n_layers):
            self.layers.append(TimeDistributed(Conv2D(filters=self.n_filters, kernel_size=3, strides=(1,1),
                                                      padding='same', activation=CTCClassifier.lrelu)))
            #self.batch_norm_layers.append(BatchNormalization())
            if i == n_layers-1:
                self.batch_norm_layers.append(InstanceNormalization()) # allow scaling in final layer only
            else:
                self.batch_norm_layers.append(InstanceNormalization(center=False, scale=False))
            self.dropout_layers.append(TimeDistributed(Dropout(0.05)))
            self.concatenate_layers.append(TimeDistributed(Concatenate()))
            

    def call(self, x, training=False):
        layer_outs = [x]
        for i in range(self.n_layers):
            y = self.concatenate_layers[i](layer_outs, training=training)
            y = self.layers[i](y, training=training)
            y = self.batch_norm_layers[i](y, training=training)
            y = self.dropout_layers[i](y, training=training)
            layer_outs.append(y)
        return y


        
    def get_config(self):
        return {'n_layers': self.n_layers, 'n_filters': self.n_filters}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
