import tensorflow as tf
import numpy as np
from pathlib import Path
layers = tf.keras.layers

from skatingAI.nets.hrnet.hrnet import create_hrnet_large
from skatingAI.utils.utils import get_random_train_image, DisplayCallback, calculate_loss
tf.compat.v1.enable_eager_execution()
#from keras.utils.vis_utils import plot_model



if __name__ == "__main__":
    keypoint_count = 17
    dimensions = 2
    batch_size=64
    buffer_size=128
    epoch_steps=5000
    epochs=50
    width=640
    height=427
    #inputs = tf.keras.Input(shape=(4,None,None,3))
    ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=(), )


    ds_series = tf.data.Dataset.from_generator(get_random_train_image, args=[25],output_types=(tf.float32, tf.float32),
                                             output_shapes = ((height,width,3),(height,width,3)))


    train_dataset = ds_series.cache().shuffle(buffer_size).batch(batch_size).repeat()
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = ds_series.batch(batch_size)


    for image, mask in ds_series.take(1):
        sample_image, sample_mask = image, mask

    print('# create hrnet')
    model = create_hrnet_large(input_shape=(height,width, 3,))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='nadins_hrnet_1.png', show_shapes=True, expand_nested=False)

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics")
    file_writer.set_as_default()

    print('# Fit model on training data')

    history = model.fit(train_dataset,
                        epochs=epochs,
                        steps_per_epoch=epoch_steps,
                        validation_data=test_dataset,
                        callbacks=[
                            DisplayCallback(
                                model, sample_image, sample_mask, file_writer, epochs=epochs),
                            tf.keras.callbacks.TensorBoard(
                                log_dir=f"{Path.cwd()}/logs")
                        ])

    calculate_loss(history, epochs)
