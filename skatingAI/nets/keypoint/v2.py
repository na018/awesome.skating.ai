import tensorflow as tf
import tensorflow.keras.backend as K

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class KPDetector(object):
    def __init__(self, input_shape, hrnet_input: tf.keras.Model, output_channels=15):
        self.inputs = tf.keras.Input(shape=input_shape, name='input_hrnet')
        self.output_channels = output_channels
        self.outputs = None
        self.hrnet_input: tf.keras.Model = hrnet_input(self.inputs)
        self.hrnet_input_layer = self.hrnet_input
        self.model = self._build_model()


    def _build_model(self):
        mask = tf.reduce_max(self.hrnet_input, axis=-1, keepdims=True)
        pool = layers.MaxPool2D(pool_size=[2, 2])(mask)
        pool = layers.BatchNormalization(momentum=BN_MOMENTUM)(pool)
        pool = layers.AlphaDropout(0.1)(pool)
        conv = layers.Conv2D(32, kernel_size=5, padding="valid")(pool)  # 116x156x32

        pool = layers.MaxPool2D(pool_size=[2, 2])(conv)
        pool = layers.BatchNormalization(momentum=BN_MOMENTUM)(pool)
        pool = layers.AlphaDropout(0.2)(pool)
        conv = layers.Conv2D(64, kernel_size=3, padding="valid")(pool)  # 56x76x64

        pool = layers.MaxPool2D(pool_size=[2, 2])(conv)
        pool = layers.BatchNormalization(momentum=BN_MOMENTUM)(pool)
        pool = layers.AlphaDropout(0.2)(pool)
        conv = layers.Conv2D(128, kernel_size=3, padding="valid")(pool)  # 26x36x64

        pool = layers.MaxPool2D(pool_size=[2, 2])(conv)
        pool = layers.BatchNormalization(momentum=BN_MOMENTUM)(pool)
        pool = layers.AlphaDropout(0.3)(pool)
        conv = layers.Conv2D(512, kernel_size=3, padding="valid")(pool)  # 11x16x64

        flatten = layers.Flatten()(conv)
        dense = layers.Dense(1024)(flatten)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.2)(dense)
        dense = layers.Dense(512, )(dense)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.1)(dense)
        self.outputs = layers.Dense(self.output_channels)(dense)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
