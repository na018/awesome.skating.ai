import tensorflow as tf
import tensorflow.keras.backend as K

from skatingAI.nets.hrnet.HPNetBase import HPNetBase

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class HPNet(HPNetBase):

    def _build_model(self) -> tf.keras.Model:
        # --------------first-block-------------------#
        # input = self.stride_down(self.inputs, name="input")
        block_l_1 = self.conv3x3_block(self.inputs, filter_counts=[16, 16, 16, 16], name="1bl")
        block_m_1 = self.stride_down(block_l_1, name="1bm")
        block_s_1 = self.stride_down(block_m_1, 1, k=4, name="1bs")
        block_xs_1 = self.stride_down(block_s_1, 1, k=4, name="1bxs")

        block_l_1 = self.conv3x3_block(block_l_1, 2, filter_counts=[16, 16, 16, 16], name="2bl")
        block_m_1 = self.conv3x3_block(block_m_1, 2, filter_counts=[16, 32, 32, 64], name="2bm")
        block_s_1 = self.conv3x3_block(block_s_1, 2, filter_counts=[32, 64, 64, 128], name="2bs")
        block_xs_1 = self.conv3x3_block(block_xs_1, 2, filter_counts=[32, 64, 64, 128], name="2bxs")

        block_s_2 = self.stride_up(block_xs_1, 3, k=4, name="3bs")
        block_s_2 = layers.concatenate([block_s_1, block_s_2])
        block_m_2 = self.stride_up(block_s_2, 3, k=4, name="3bm")
        block_m_2 = layers.concatenate([block_m_1, block_m_2])
        block_l_2 = self.stride_up(block_m_2, 3, name="3bl")

        concat = layers.concatenate([block_l_1, block_l_2])

        self.outputs = layers.Conv2D(filters=self.output_channels, kernel_size=3,
                                     activation='softmax',
                                     padding="same",
                                     name=f"output")(concat)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
