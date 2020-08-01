import tensorflow as tf
import tensorflow.keras.backend as K

from skatingAI.nets.hrnet.HPNetBase import HPNetBase

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class HPNet(HPNetBase):

    def _build_model(self) -> tf.keras.Model:
        # --------------first-block-------------------#
        input = self.stride_down(self.bgnet_input, k=2, name="input")
        input = layers.concatenate([input, self.inputs])
        block_l = self.conv3x3_block(input, filter_counts=[16, 16, 16, 16], name="bl")

        # --------------second-block-------------------#
        block_m = self.stride_down(block_l, name="bm")
        block_l = self.conv3x3_block(block_l, 2, filter_counts=[16, 16, 16, 16], name="bl")
        block_m = self.conv3x3_block(block_m, 2, filter_counts=[16, 32, 32, 64], name="bm")
        block_m = self.stride_up(block_m, 2, name="bm")

        concat = layers.concatenate([block_l, block_m])

        block_l = self.conv3x3_block(concat, 3, filter_counts=[16, 16, 16, 16], name="bl")

        block_m = self.stride_down(concat, 3, name="bm")
        block_s = self.stride_down(concat, 3, k=8, name="bs")

        block_m = self.conv3x3_block(block_m, 3, filter_counts=[16, 32, 32, 64], name="bm")
        block_m = self.stride_up(block_m, 3, name="bm")

        block_s = self.conv3x3_block(block_s, 3, filter_counts=[32, 64, 64, 128], name="bs")
        block_s = self.stride_up(block_s, 3, k=8, name="bs1")

        concat = layers.concatenate([block_l, block_m, block_s])

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
                                     name=f"output")(concat)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
