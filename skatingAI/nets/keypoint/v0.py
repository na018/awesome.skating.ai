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

    def conv3x3_block(self, inputs: tf.Tensor,
                      block_nr=1,
                      filter_counts=[36, 64, 121, 256],
                      layer_type=["c", "c", "c", "c"],
                      activation=["selu", "selu", "selu", "selu"],
                      name="block") -> tf.Tensor:
        conv = tf.identity(inputs)
        for i, filter in enumerate(filter_counts):
            if layer_type[i] not in ["c", "dw"]:
                return AssertionError(f"layer_type '{layer_type[i]}' must be 'c' or 'dw'")

            if layer_type[i] == 'c':
                conv = layers.Conv2D(filter,
                                     kernel_size=3,
                                     padding="same",
                                     name=f"{name}_{block_nr}_conv_{i}")(conv)
            else:
                conv = layers.DepthwiseConv2D(filter,
                                              padding="same",
                                              name=f"{name}_{block_nr}_conv_{i}")(conv)

            conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{i}")(conv)

            if activation[i]:
                conv = layers.Activation(activation[i], name=f"{name}_{block_nr}_{activation[i]}_{i}")(conv)

        return conv

    def stride_down(self, inputs: tf.Tensor, block_nr=1, k=5, f=49, activation='selu', name="block") -> tf.Tensor:

        conv = layers.Conv2D(f, k, strides=k, padding="same", name=f"{name}_{block_nr}_conv_stride_down")(inputs)

        conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{block_nr}_stride_down")(
            conv)

        return layers.Activation(activation, name=f"{name}_{block_nr}_{activation}_{block_nr}_stride_down")(conv)

    def stride_up(self, inputs: tf.Tensor, block_nr=1, k=5, f=25, activation='selu', name="block") -> tf.Tensor:

        conv = layers.Conv2DTranspose(f, k, strides=k, padding="same",
                                      name=f"{name}_{block_nr}_conv_stride_up")(inputs)
        conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{block_nr}_stride_up")(conv)

        return layers.Activation(activation, name=f"{name}_{block_nr}_{activation}_{block_nr}_stride_up")(conv)

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
        dense = layers.Dense(256, 'relu')(flatten)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.2)(dense)
        dense = layers.Dense(90, 'sigmoid')(dense)
        dense = layers.BatchNormalization(momentum=BN_MOMENTUM)(dense)
        dense = layers.AlphaDropout(0.1)(dense)
        dense = layers.Dense(self.output_channels)(dense)

        model = tf.keras.Model(inputs=self.inputs, outputs=dense)

        return model
