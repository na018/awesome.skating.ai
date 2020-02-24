import tensorflow as tf

from pathlib import Path
layers = tf.keras.layers

from skatingAI.nets.hrnet.hrnet import create_hrnet_large
from skatingAI.utils.utils import get_video_batches, DisplayCallback
#tf.compat.v1.enable_eager_execution()




if __name__ == "__main__":
    keypoint_count = 17
    dimensions = 2
    batch_size=3
    buffer_size=1000
    epoch_steps=12
    epochs=128
    # width=640
    # height=427
    width=640
    height=480


    # download datasets


    train_dataset = tf.data.Dataset.from_generator(get_video_batches,output_types=(tf.float32, tf.float32),
                                             output_shapes = ((batch_size, 480, 640, 3),(batch_size, 480, 640, 1)), args=[batch_size])
    test_dataset = tf.data.Dataset.from_generator(get_video_batches,output_types=(tf.float32, tf.float32),
                                             output_shapes = ((batch_size, 480, 640, 3),(batch_size, 480, 640, 1)), args=[batch_size])

    for image, mask in train_dataset.take(1):
        sample_image, sample_mask = image[0], mask[0]


    #
    #
    # train_dataset = ds_series.batch(batch_size).repeat()
    # # train_dataset = train_dataset.prefetch(
    # #     buffer_size=tf.data.experimental.AUTOTUNE)
    #
    # test_dataset = ds_series.batch(batch_size)
    #
    #
    # for image, mask in ds_series.take(1):
    #     sample_image, sample_mask = image, mask

    print('# create hrnet')
    model = create_hrnet_large(input_shape=(height,width, 3,), output_channels=15)

    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, amsgrad=True)
    model.compile(optimizer='sgd', loss=tf.keras.losses.MSE, metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='nadins_hrnet_1.png', show_shapes=True, expand_nested=False)

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics")
    file_writer.set_as_default()

    print('# Fit model on training data')

    # session = tf.compat.v1.keras.backend.get_session()
    # tf.compat.v1.keras.backend.set_session(
    #     tf_debug.TensorBoardDebugWrapperSession(session, "nadinkatrin-HP-ENVY-17-Notebook-PC:8000"))

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

    #calculate_loss(history, epochs)
# tensorboard --logdir /home/nadin-katrin/CodeProjects/00_skating_ai/awesome.skating.ai/skatingAI/logs
