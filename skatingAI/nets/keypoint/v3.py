import tensorflow as tf
import tensorflow.keras.backend as K

from skatingAI.nets.keypoint.KPDetectorBase import KPDetectorBase

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class KPDetector(KPDetectorBase):

    def _build_model(self):
        img_input = self.conv3x3_block(self.inputs, filter_counts=[16, 16, 33], name="img_input")
        hrnet_input = self.conv3x3_block(self.hrnet_input, filter_counts=[16, 16, 33], name="hrnet_input")
        input = layers.concatenate([hrnet_input, img_input])

        block_l = self.conv3x3_block(input, 4, filter_counts=[49, 25], layer_type=['c', 'c', 'c'],
                                     name="bl")

        block_m = self.stride_down(self.hrnet_input, 4, name="bm")
        block_s = self.stride_down(self.hrnet_input, 4, k=8, name="bs")
        block_xs = self.stride_down(self.hrnet_input, 4, k=20, name="bxs")

        block_m = self.conv3x3_block(block_m, 4, filter_counts=[64], name="bm")
        block_m = self.stride_up(block_m, 4, name="bm")

        block_s = self.conv3x3_block(block_s, 4, filter_counts=[64], name="bs")
        block_s = self.stride_up(block_s, 4, k=8, name="bs1")

        block_xs = self.conv3x3_block(block_xs, 4, filter_counts=[64], name="bxs")
        block_xs = self.stride_up(block_xs, 4, k=20, name="bxs")

        concat = layers.concatenate([block_l, block_m, block_s, block_xs, ])

        self.outputs = layers.Conv2D(filters=self.output_channels, kernel_size=3,
                                     activation='softmax',
                                     padding="same",
                                     name=f"output")(concat)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model
