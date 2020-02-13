import tensorflow as tf

layers = tf.keras.layers


BN_MOMENTUM = 0.01


def conv3x3_block(inputs: tf.Tensor, block_nr=1, filter_counts=[32, 64, 128, 258], name="block") -> tf.Tensor:
    bn = tf.identity(inputs)
    for i, filter in enumerate(filter_counts):
        conv = layers.Conv2D(64, kernel_size=3, activation='relu', padding="same", name=f"{name}_{block_nr}_conv_{i}")(
            bn)
        bn = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_{block_nr}_bn_{i}")(conv)

    return bn


def stride_down(inputs: tf.Tensor, block_nr=1, k=4, f=258, name="block") -> tf.Tensor:
    return layers.Conv2D(f, k, strides=k, activation='relu', padding="same", name=f"{name}_{block_nr}_stride_down")(
        inputs)


def stride_up(inputs: tf.Tensor, block_nr=1, k=4, f=64, name="block") -> tf.Tensor:
    return layers.Conv2DTranspose(f, k, strides=k, activation='relu', padding="same",
                                  name=f"{name}_{block_nr}_stride_up")(inputs)


def create_hrnet_large(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, block_amount=3, output_channels=18) -> tf.Tensor:

    if batch_shape:
        img_input = tf.keras.Input(batch_shape=batch_shape)
        #image_size = batch_shape[1:3]
    else:
        img_input = tf.keras.Input(shape=(input_shape))
        #image_size = input_shape[0:2]


    # --------------first-block-------------------#
    block_l = conv3x3_block(img_input, name="bl")
    block_m = stride_down(block_l, name="bm")

    # --------------second-block-------------------#
    block_l = conv3x3_block(block_l, 2, name="bl")
    block_m = conv3x3_block(block_m, 2, name="bm")
    block_m = stride_up(block_m, 2, name="bm")
    #block_m = tf.keras.backend.resize_volumes(he)(block_m)

    # block_m = layers.Cropping2D(cropping=((0,1), (0, 0)),
    #                         input_shape=(427, 640, 64))(block_m)

    concat = layers.concatenate([block_l, block_m])

#ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs shapes: [(None, 427, 640, 64), (None, 428, 640, 258)]
    # --------------third-block-------------------#
    if block_amount > 2:
        block_l = conv3x3_block(concat, 3, name="bl")

        block_m = stride_down(concat, 3, name="bm")
        block_s = stride_down(concat, 3, k=8, name="bs")

        block_m = conv3x3_block(block_m, 3, name="bm")
        block_m = stride_up(block_m, 3, name="bm")

        # block_m = layers.Cropping2D(cropping=((1, 0), (0, 0)),
        #                             input_shape=(427, 640, 64))(block_m)

        block_s = conv3x3_block(block_s, 3, name="bs")
        block_s = stride_up(block_s, 3, k=8, name="bs1")

        # block_s = layers.Cropping2D(cropping=((2, 3), (0, 0)),
        #                              input_shape=(427, 640, 64))(block_s)

        concat = layers.concatenate([block_l, block_m, block_s])


    # --------------fourth-block-------------------#
    if block_amount > 3:
        block_l = conv3x3_block(concat, 4, name="bl")

        block_m = stride_down(concat, 4, name="bm")
        block_s = stride_down(concat, 4, k=8, name="bs")
        block_xs = stride_down(concat, 4, k=20, name="bxs")

        block_m = conv3x3_block(block_m, 4, name="bm")
        block_m = stride_up(block_m, 4, name="bm")

        block_s = conv3x3_block(block_s, 4, name="bs")
        block_s = stride_up(block_s, 4, k=8, name="bs1")

        block_xs = conv3x3_block(block_xs, 4, name="bxs")
        block_xs = stride_up(block_xs, 4, k=20, name="bxs")

        concat = layers.concatenate([block_l, block_m, block_s, block_xs])

    # output 427x640x2x34

    outputs = layers.Conv2D(filters=output_channels, kernel_size=3, activation='relu', padding="same", name=f"output")(concat)

    model = tf.keras.Model(inputs=[img_input], outputs=[outputs])

    return model
