import tensorflow as tf
import tensorflow.keras.backend as K

from skatingAI.nets.keypoint.KPDetectorBase import KPDetectorBase

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class KPDetector(KPDetectorBase):
    def __init__(self, input_shape, hrnet_input: tf.keras.Model, output_channels=15):
        super().__init__(input_shape, hrnet_input, output_channels)

    def _build_model(self):
        block_l = self.conv3x3_block(self.hrnet_input_layer, 4, filter_counts=[49, 25], layer_type=['c', 'c', 'c'],
                                     name="bl")

        block_m = self.stride_down(self.hrnet_input_layer, 4, name="bm")
        block_s = self.stride_down(self.hrnet_input_layer, 4, k=8, name="bs")
        block_xs = self.stride_down(self.hrnet_input_layer, 4, k=20, name="bxs")

        block_m = self.conv3x3_block(block_m, 4, filter_counts=[64], name="bm")
        block_m = self.stride_up(block_m, 4, name="bm")

        block_s = self.conv3x3_block(block_s, 4, filter_counts=[64], name="bs")
        block_s = self.stride_up(block_s, 4, k=8, name="bs1")

        block_xs = self.conv3x3_block(block_xs, 4, filter_counts=[64], name="bxs")
        block_xs = self.stride_up(block_xs, 4, k=20, name="bxs")

        concat = layers.concatenate([block_l, block_m, block_s, block_xs, ])

        pool = layers.MaxPool2D(pool_size=[6, 6])(concat)
        pool = layers.BatchNormalization(momentum=BN_MOMENTUM)(pool)
        pool = layers.AlphaDropout(0.2)(pool)

        flatten = layers.Flatten()(pool)
        dense = layers.Dense(1024, 'relu')(flatten)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.2)(dense)
        dense = layers.Dense(512, 'sigmoid')(dense)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.1)(dense)
        dense = layers.Dense(self.output_channels)(dense)

        model = tf.keras.Model(inputs=self.inputs, outputs=dense)

        return model
