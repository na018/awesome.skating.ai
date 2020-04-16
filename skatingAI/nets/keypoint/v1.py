import tensorflow as tf
import tensorflow.keras.backend as K

from skatingAI.nets.keypoint.KPDetectorBase import KPDetectorBase

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class KPDetector(KPDetectorBase):
    def __init__(self, input_shape, hrnet_input: tf.keras.Model, output_channels=15):
        super().__init__(input_shape, hrnet_input, output_channels)

    def _build_model(self):
        pool = layers.MaxPool2D(pool_size=[6, 6])(self.hrnet_input)
        pool = layers.BatchNormalization(momentum=BN_MOMENTUM)(pool)
        pool = layers.AlphaDropout(0.2)(pool)
        flatten = layers.Flatten()(pool)
        dense = layers.Dense(256, 'relu')(flatten)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.2)(dense)
        dense = layers.Dense(90, 'sigmoid')(dense)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.1)(dense)
        self.outputs = layers.Dense(self.output_channels)(dense)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
