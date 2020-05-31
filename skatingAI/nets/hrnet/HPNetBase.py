import tensorflow as tf
import tensorflow.keras.backend as K

layers = tf.keras.layers

BN_MOMENTUM = 0.01


class HPNetBase(object):

    def __init__(self, input_shape, output_channels=15):
        self.inputs = tf.keras.Input(shape=input_shape, name='images')
        self.output_channels = output_channels
        self.model = self._build_model()
        self.outputs = None

    # noinspection PyDefaultArgument
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

    def stride_down(self, inputs: tf.Tensor, block_nr=1, k=5, f=36, activation='selu', name="block") -> tf.Tensor:

        conv = layers.Conv2D(f, k, strides=k, padding="same", name=f"{name}_{block_nr}_conv_stride_down")(inputs)

        conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{block_nr}_stride_down")(
            conv)

        return layers.Activation(activation, name=f"{name}_{block_nr}_{activation}_{block_nr}_stride_down")(conv)

    def stride_up(self, inputs: tf.Tensor, block_nr=1, k=5, f=36, activation='selu', name="block") -> tf.Tensor:

        conv = layers.Conv2DTranspose(f, k, strides=k, padding="same",
                                      name=f"{name}_{block_nr}_conv_stride_up")(inputs)
        conv = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{block_nr}_stride_up")(conv)

        return layers.Activation(activation, name=f"{name}_{block_nr}_{activation}_{block_nr}_stride_up")(conv)

    def _build_model(self) -> tf.keras.Model:
        raise NotImplementedError