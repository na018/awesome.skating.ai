import tensorflow as tf
import numpy as np
from pathlib import Path
layers = tf.keras.layers

from skatingAI.nets.hrnet.hrnet import create_hrnet_large
from skatingAI.utils.utils import get_random_images, DisplayCallback

#from keras.utils.vis_utils import plot_model



if __name__ == "__main__":
    keypoint_count = 17
    dimensions = 2
    batch_size=64
    width=640
    height=427
    #inputs = tf.keras.Input(shape=(4,None,None,3))



    train_x, train_y= get_random_images()

    val_x = np.array(train_x[-100:])
    val_y = np.array(train_y[-100:])
    train_x = np.array(train_x[:-100])
    train_y = np.array(train_y[:-100])
    sample_image, sample_mask = train_x[0], train_y[0]

    print('# create hrnet')
    model = create_hrnet_large(input_shape=(height,width, 3,))
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], options=run_opts)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='nadins_hrnet_1.png', show_shapes=True, expand_nested=False)

    # Specify the training configuration (optimizer, loss, metrics)
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
    #               # Loss function to minimize
    #               loss=tf.keras.losses.KLDivergence(),
    #               # List of metrics to monitor
    #               metrics=[tf.keras.metrics.MeanSquaredError()])

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics")
    file_writer.set_as_default()

    print('# Fit model on training data')

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=3,
                        # steps_per_epoch=1,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(val_x, val_y),
                        callbacks=[
                            DisplayCallback(
                                model, sample_image, sample_mask, file_writer),
                            tf.keras.callbacks.TensorBoard(
                                log_dir=f"{Path.cwd()}/logs")
                        ])

    print('\nhistory dict:', history.history)

    print('\n# Evaluate on test data')
    results = model.evaluate(val_x, val_y, batch_size=128)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('\n# Generate predictions for 3 samples')
    predictions = model.predict(val_x[:3])
    print('predictions shape:', predictions.shape)


# ValueError: Error when checking input: expected input_1 to have 4 dimensions, but got array with shape (16, 2, 8000)