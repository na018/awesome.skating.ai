import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


class UNet(object):
    def __init__(self, input_shape=[240, 320, 3], output_channels=9):
        self.input_shape = list(input_shape)
        self.inputs = layers.Input(input_shape)
        self.output_channels = output_channels
        self.model = self._build_model()

    def _encoder(self):
        mobile_net_base_model = MobileNetV2(self.input_shape, include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 120x160x96
            'block_3_expand_relu',  # 60x80x144
            'block_6_expand_relu',  # 30x40x192
            'block_13_expand_relu',  # 15x20x576
            'block_16_project',  # 8x10x320
        ]

        layers = [mobile_net_base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = Model(inputs=mobile_net_base_model.input, outputs=layers)

        down_stack.trainable = False

        return down_stack

    def stride_up(self, inputs: tf.Tensor, filters, kernel):
        initializer = tf.random_normal_initializer(0., 0.02)
        conv = layers.Conv2DTranspose(filters, kernel, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False)(inputs)
        conv = layers.BatchNormalization()(conv)

        return layers.Activation(tf.keras.activations.selu)(conv)

    def _decoder(self):
        up_stack = [
            [576, 3],  # 8,10,320-> 15x20x576
            [192, 3],  # 8x8 -> 16x16
            [144, 3],  # 16x16 -> 32x32
            [96, 3]  # 32x32 -> 64x64
        ]
        return up_stack

    def _build_model(self):
        decoder = self._decoder()
        encoder = self._encoder()

        skips = encoder(self.inputs)
        mobilenet_b16 = skips[-1]
        skips = reversed(skips[:-1])  # all but smallest layer b16

        conv = mobilenet_b16  # (8,10,320)

        for up, skip in zip(decoder, skips):
            conv = self.stride_up(conv, up[0], up[1])  # (16,20,512)
            if conv.shape[1] == 16:
                conv = layers.Cropping2D(cropping=((1, 0), (0, 0)))(conv)
            conv = layers.concatenate([conv, skip])

        self.output = layers.Conv2DTranspose(
            self.output_channels, 3, strides=2,
            padding='same', activation='softmax')(conv)  # 64x64 -> 128x128

        return Model(inputs=self.inputs, outputs=self.output)
