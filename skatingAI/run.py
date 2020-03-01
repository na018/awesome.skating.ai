from pathlib import Path

import tensorflow as tf

from skatingAI.nets.hrnet.hrnet import HRNet
from skatingAI.utils.DsGenerator import DsGenerator
from skatingAI.utils.utils import DisplayCallback, set_gpus

if __name__ == "__main__":
    batch_size = 10
    prefetch_batch_buffer = 5
    epoch_steps = 32
    epoch_log_n = epoch_steps // 2
    epochs = 64

    set_gpus()

    generator = DsGenerator()

    sample_pair = next(generator.get_next_pair())
    sample_frame = sample_pair['frame']
    sample_mask = sample_pair['mask']

    img_shape = sample_frame.shape

    train_ds = generator.buid_iterator(img_shape, batch_size, prefetch_batch_buffer)
    iter = train_ds.as_numpy_iterator()

    model = HRNet(img_shape).build_model()
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='nadins_hrnet_1.png', show_shapes=True, expand_nested=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, amsgrad=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_acc_metric = tf.keras.metrics.Accuracy()

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics")
    file_writer.set_as_default()
    progress_tracker = DisplayCallback(model, sample_frame, sample_mask, file_writer, epochs)

    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")

        for step in range(epoch_steps):
            batch = next(iter)

            with tf.GradientTape() as tape:
                logits = model(batch['frame'], training=True)
                loss_value = loss_fn(batch['mask'], logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric(batch['mask'], tf.argmax(logits, axis=-1))

            # Log every 200 batches.
            if step % epoch_log_n == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))

        progress_tracker.on_epoch_end(epoch, loss=loss_value, accuracy=train_acc_metric.result(), show_img=False)
        train_acc_metric.reset_states()

    progress_tracker.on_train_end()
