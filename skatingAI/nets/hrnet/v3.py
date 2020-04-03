import tensorflow as tf
import tensorflow.keras.backend as K

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class HRNet(object):

    def __init__(self, input_shape, output_channels=15, block_amount=3):
        self.inputs = tf.keras.Input(shape=input_shape, name='images')
        self.output_channels = output_channels
        self.model = self._build_model(block_amount)
        self.outputs = None

    # noinspection PyDefaultArgument
    def conv3x3_block(self, inputs: tf.Tensor,
                      block_nr=1,
                      filter_counts=[32, 64, 128, 256],
                      name="block") -> tf.Tensor:
        bn = tf.identity(inputs)
        for i, filter in enumerate(filter_counts):
            conv = layers.Conv2D(filter,
                                 kernel_size=3,
                                 activation='relu',
                                 padding="same",
                                 name=f"{name}_{block_nr}_conv_{i}")(bn)
            bn = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{i}")(conv)
            # drop = layers.Dropout(0.2)(bn)

        return bn

    def stride_down(self, inputs: tf.Tensor, block_nr=1, k=5, f=32, name="block") -> tf.Tensor:
        return layers.Conv2D(f, k, strides=k, activation='relu', padding="same",
                             name=f"{name}_{block_nr}_conv_stride_down")(
            inputs)

    def stride_up(self, inputs: tf.Tensor, block_nr=1, k=5, f=32, name="block") -> tf.Tensor:
        return layers.Conv2DTranspose(f, k, strides=k, activation='relu', padding="same",
                                      name=f"{name}_{block_nr}_conv_stride_up")(inputs)

    def _build_model(self, block_amount=3) -> tf.keras.Model:

        # --------------first-block-------------------#
        input = self.stride_down(self.inputs, k=2, name="input")
        block_l = self.conv3x3_block(input, filter_counts=[16], name="bl")

        # --------------second-block-------------------#
        block_m = self.stride_down(block_l, name="bm")
        block_l = self.conv3x3_block(block_l, 2, filter_counts=[16, 16, 16, 16], name="bl")
        block_m = self.conv3x3_block(block_m, 2, filter_counts=[16, 32, 32, 64], name="bm")
        block_m = self.stride_up(block_m, 2, name="bm")

        concat = layers.concatenate([block_l, block_m])

        # --------------third-block-------------------#
        if block_amount > 2:
            block_l = self.conv3x3_block(concat, 3, filter_counts=[16, 16, 16, 16], name="bl")

            block_m = self.stride_down(concat, 3, name="bm")
            block_s = self.stride_down(concat, 3, k=8, name="bs")

            block_m = self.conv3x3_block(block_m, 3, filter_counts=[16, 32, 32, 64], name="bm")
            block_m = self.stride_up(block_m, 3, name="bm")

            block_s = self.conv3x3_block(block_s, 3, filter_counts=[32, 64, 64, 128], name="bs")
            block_s = self.stride_up(block_s, 3, k=8, name="bs1")

            concat = layers.concatenate([block_l, block_m, block_s])

        # --------------fourth-block-------------------#
        if block_amount > 3:
            block_l = self.conv3x3_block(concat, 4, filter_counts=[16, 16, 16, 16], name="bl")

            block_m = self.stride_down(concat, 4, name="bm")
            block_s = self.stride_down(concat, 4, k=8, name="bs")
            block_xs = self.stride_down(concat, 4, k=20, name="bxs")

            block_m = self.conv3x3_block(block_m, 4, filter_counts=[16, 32, 32, 64], name="bm")
            block_m = self.stride_up(block_m, 4, name="bm")

            block_s = self.conv3x3_block(block_s, 4, filter_counts=[32, 64, 64, 128], name="bs")
            block_s = self.stride_up(block_s, 4, k=8, name="bs1")

            block_xs = self.conv3x3_block(block_xs, 4, filter_counts=[32, 64, 64, 128], name="bxs")
            block_xs = self.stride_up(block_xs, 4, k=20, name="bxs")

            concat = layers.concatenate([block_l, block_m, block_s, block_xs])

        concat = self.stride_up(concat, 5, k=2, name="bl")
        self.outputs = layers.Conv2D(filters=self.output_channels, kernel_size=3,
                                     activation='softmax',
                                     padding="same",
                                     name=f"conv_output")(concat)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
