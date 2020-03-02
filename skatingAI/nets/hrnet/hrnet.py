import tensorflow as tf

layers = tf.keras.layers

BN_MOMENTUM = 0.01

class HRNet(object):

    def __init__(self, input_shape, output_channels=15, block_amount=3):
        self.inputs = tf.keras.Input(shape=input_shape, name='images')
        self.output_channels = output_channels
        self.model = self._build_model(block_amount)
        self.outputs = None

    # noinspection PyDefaultArgument
    def conv3x3_block(self, inputs: tf.Tensor, block_nr=1, filter_counts=[32, 64, 128, 258], name="block") -> tf.Tensor:
        bn = tf.identity(inputs)
        for i, filter in enumerate(filter_counts):
            conv = layers.Conv2D(64, kernel_size=3, activation='relu', padding="same",
                                 name=f"{name}_{block_nr}_conv_{i}")(
                bn)
            bn = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{i}")(conv)

        return bn

    def stride_down(self, inputs: tf.Tensor, block_nr=1, k=4, f=258, name="block") -> tf.Tensor:
        return layers.Conv2D(f, k, strides=k, activation='relu', padding="same", name=f"{name}_{block_nr}_stride_down")(
            inputs)

    def stride_up(self, inputs: tf.Tensor, block_nr=1, k=4, f=64, name="block") -> tf.Tensor:
        return layers.Conv2DTranspose(f, k, strides=k, activation='relu', padding="same",
                                      name=f"{name}_{block_nr}_stride_up")(inputs)

    def _build_model(self, block_amount=3) -> tf.keras.Model:

        # --------------first-block-------------------#
        block_l = self.conv3x3_block(self.inputs, name="bl")
        block_m = self.stride_down(block_l, name="bm")

        # --------------second-block-------------------#
        block_l = self.conv3x3_block(block_l, 2, name="bl")
        block_m = self.conv3x3_block(block_m, 2, name="bm")
        block_m = self.stride_up(block_m, 2, name="bm")
        # block_m = tf.keras.backend.resize_volumes(he)(block_m)

        # block_m = layers.Cropping2D(cropping=((0,1), (0, 0)),
        #                         input_shape=(427, 640, 64))(block_m)

        concat = layers.concatenate([block_l, block_m])

        # --------------third-block-------------------#
        if block_amount > 2:
            block_l = self.conv3x3_block(concat, 3, name="bl")

            block_m = self.stride_down(concat, 3, name="bm")
            block_s = self.stride_down(concat, 3, k=8, name="bs")

            block_m = self.conv3x3_block(block_m, 3, name="bm")
            block_m = self.stride_up(block_m, 3, name="bm")

            # block_m = layers.Cropping2D(cropping=((1, 0), (0, 0)),
            #                             input_shape=(427, 640, 64))(block_m)

            block_s = self.conv3x3_block(block_s, 3, name="bs")
            block_s = self.stride_up(block_s, 3, k=8, name="bs1")

            # block_s = layers.Cropping2D(cropping=((2, 3), (0, 0)),
            #                              input_shape=(427, 640, 64))(block_s)

            concat = layers.concatenate([block_l, block_m, block_s])

        # --------------fourth-block-------------------#
        if block_amount > 3:
            block_l = self.conv3x3_block(concat, 4, name="bl")

            block_m = self.stride_down(concat, 4, name="bm")
            block_s = self.stride_down(concat, 4, k=8, name="bs")
            block_xs = self.stride_down(concat, 4, k=20, name="bxs")

            block_m = self.conv3x3_block(block_m, 4, name="bm")
            block_m = self.stride_up(block_m, 4, name="bm")

            block_s = self.conv3x3_block(block_s, 4, name="bs")
            block_s = self.stride_up(block_s, 4, k=8, name="bs1")

            block_xs = self.conv3x3_block(block_xs, 4, name="bxs")
            block_xs = self.stride_up(block_xs, 4, k=20, name="bxs")

            concat = layers.concatenate([block_l, block_m, block_s, block_xs])

        # output 427x640x2x34

        self.outputs = layers.Conv2D(filters=self.output_channels, kernel_size=3, activation='relu', padding="same",
                                name=f"output")(concat)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        return model

    # # Define custom loss
    # def custom_loss(self, x, y, training):
    #
    #     y_pred = self.model(x, training=training)
    #
    #     # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    #     def loss(y_true, y_pred):
    #         return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)
    #
    #     # Return a function
    #     return loss
