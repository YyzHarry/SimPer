"""
Example network architectures:
- Featurizer (for representation learning)
- Classifier (for downstream tasks)
"""
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, Flatten, BatchNormalization,
                                     TimeDistributed, MaxPool2D, GlobalAveragePooling2D)


class Featurizer(tf.keras.Model):

    def __init__(self, n_outputs):
        super(Featurizer, self).__init__()
        self.conv0 = Conv3D(64, (5, 3, 3), padding='same')
        self.conv1 = Conv3D(128, (5, 3, 3), padding='same')
        self.conv2 = Conv3D(128, (5, 3, 3), padding='same')
        self.conv3 = Conv3D(1, (1, 1, 1))

        self.bn0 = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

        self.pool0 = TimeDistributed(MaxPool2D((2, 2)))
        self.pool1 = TimeDistributed(MaxPool2D((2, 2)))
        self.pool2 = TimeDistributed(MaxPool2D((2, 2)))
        self.pool3 = TimeDistributed(GlobalAveragePooling2D())

        self.flatten = Flatten()

        self.n_outputs = n_outputs

    def call(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = tf.nn.relu(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = tf.nn.relu(x)
        # x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        return x


class MLP(tf.keras.Model):

    def __init__(self, n_outputs):
        super(MLP, self).__init__()
        self.inputs = Dense(n_outputs)
        self.hidden = Dense(n_outputs)
        self.outputs = Dense(n_outputs)

    def call(self, x):
        x = self.inputs(x)
        x = tf.nn.relu(x)
        # x = self.hidden(x)
        # x = tf.nn.relu(x)
        x = self.outputs(x)
        return x


def Classifier(in_features, out_features, nonlinear=False):
    if nonlinear:
        return tf.keras.Sequential(
            [Dense(in_features // 2, activation=tf.nn.relu),
             Dense(in_features // 4, activation=tf.nn.relu),
             Dense(out_features)])
    else:
        return Dense(out_features)
