import tensorflow as tf
import tensorflow.keras.backend as K

from skatingAI.nets.hrnet import HRNetBase

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class HRNet(HRNetBase):

    def conv3x3_block(self, inputs: tf.Tensor,
                      block_nr=1,
                      filter_counts=[36, 64, 121, 256],
                      layer_type=["c", "dw", "dw", "c"],
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

    def stride_down(self, inputs: tf.Tensor, block_nr=1, k=5, f=36,activation='selu',  name="block") -> tf.Tensor:

        conv=layers.Conv2D(f, k, strides=k, padding="same", name=f"{name}_{block_nr}_conv_stride_down")(inputs)

        conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{block_nr}_stride_down")(conv)

        return layers.Activation(activation, name=f"{name}_{block_nr}_{activation}_{block_nr}_stride_down")(conv)

    def stride_up(self, inputs: tf.Tensor, block_nr=1, k=5, f=36, activation='selu', name="block") -> tf.Tensor:

        conv = layers.Conv2DTranspose(f, k, strides=k, padding="same",
                                      name=f"{name}_{block_nr}_conv_stride_up")(inputs)
        conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{block_nr}_stride_up")(conv)

        return layers.Activation(activation, name=f"{name}_{block_nr}_{activation}_{block_nr}_stride_up")(conv)

    def _build_model(self, block_amount=3) -> tf.keras.Model:
        # --------------first-block-------------------#
        input = self.stride_down(self.inputs, k=2, f=36, name="input")
        # input2 = self.stride_down(self.inputs, k=2, f=1, name="input2")
        # block_l = self.conv3x3_block(input, filter_counts=[9], name="bl")

        # --------------second-block-------------------#

        block_l = self.conv3x3_block(input, 2, filter_counts=[16, 16, 36], layer_type=['c', 'dw', 'c'], name="bl")
        block_m = self.stride_down(input, name="bm")
        block_m = self.conv3x3_block(block_m, 2, filter_counts=[64, 36, 36], layer_type=['c', 'dw', 'c'], name="bm")
        block_m = self.stride_up(block_m, 2, name="bm")

        concat = layers.add([block_l, block_m, input])

        # --------------third-block-------------------#
        block_l = self.conv3x3_block(concat, 3, filter_counts=[16, 16, 36], layer_type=['c', 'dw', 'c'], name="bl")

        block_m = self.stride_down(concat, 3, name="bm")
        block_s = self.stride_down(concat, 3, k=8, name="bs")

        block_m = self.conv3x3_block(block_m, 3, filter_counts=[64, 36, 36, 36], name="bm")
        block_m = self.stride_up(block_m, 3, name="bm")

        block_s = self.conv3x3_block(block_s, 3, filter_counts=[121, 64, 64, 36], name="bs")
        block_s = self.stride_up(block_s, 3, k=8, name="bs1")

        concat = layers.add([block_l, block_m, block_s, input])

        # --------------fourth-block-------------------#
        block_l = self.conv3x3_block(concat, 4, filter_counts=[16, 16, 36], layer_type=['c', 'dw', 'c'], name="bl")

        block_m = self.stride_down(concat, 4, name="bm")
        block_s = self.stride_down(concat, 4, k=8, name="bs")
        block_xs = self.stride_down(concat, 4, k=20, name="bxs")

        block_m = self.conv3x3_block(block_m, 4, filter_counts=[64, 36, 36, 36], name="bm")
        block_m = self.stride_up(block_m, 4, name="bm")

        block_s = self.conv3x3_block(block_s, 4, filter_counts=[121, 121, 64, 36], name="bs")
        block_s = self.stride_up(block_s, 4, k=8, name="bs1")

        block_xs = self.conv3x3_block(block_xs, 4, filter_counts=[256, 121, 64, 64], name="bxs")
        block_xs = self.stride_up(block_xs, 4, k=20, name="bxs")

        concat = layers.add([block_l, block_m, block_s, block_xs, input])

        self.outputs = self.stride_up(concat, 5, k=2, f=9, activation="softmax", name="conv_output")
        # self.outputs = layers.Conv2D(filters=self.output_channels, kernel_size=3,
        #                              activation='relu',
        #                              padding="same",
        #                              name=f"conv_output")(concat)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
