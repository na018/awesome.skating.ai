from pathlib import Path

import tensorflow as tf
import numpy as np

from skatingAI.nets.hrnet.hrnet import HRNet
from skatingAI.utils.DsGenerator import DsGenerator
from skatingAI.utils.utils import DisplayCallback, set_gpus, Metric

if __name__ == "__main__":
    batch_size = 3
    prefetch_batch_buffer = 1
    epoch_steps = 12
    epoch_log_n = epoch_steps // 2
    epochs = 256

    strategy = set_gpus()

    generator = DsGenerator()

    sample_pair = next(generator.get_next_pair())
    sample_frame = sample_pair['frame']
    sample_mask = sample_pair['mask']

    img_shape = sample_frame.shape

    train_ds = generator.buid_iterator(img_shape, batch_size, prefetch_batch_buffer)
    iter = train_ds.as_numpy_iterator()

    if strategy:
        with strategy.scope():
            model = HRNet(img_shape).model
    else:
        model = HRNet(img_shape).model
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='nadins_hrnet_1.png', show_shapes=True, expand_nested=False)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, amsgrad=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn = tf.keras.losses.CategoricalHinge()
    train_acc_metric = tf.keras.metrics.Accuracy()
    train_acc_metric_custom = 0
    # train_rec_metric = tf.keras.metrics.Recall()
    # train_true_pos_metric = tf.keras.metrics.TruePositives()

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics")
    file_writer.set_as_default()
    progress_tracker = DisplayCallback(model, sample_frame, sample_mask, file_writer, epochs)

    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")
        train_acc_metric_custom = 0

        for step in range(epoch_steps):
            batch = next(iter)

            with tf.GradientTape() as tape:
                logits = model(batch['frame'], training=True)
                loss_value = loss_fn(batch['mask'], logits)
                max_logits = tf.argmax(logits, axis=-1)
                max_logits = max_logits[..., tf.newaxis]
                #loss_value = tf.convert_to_tensor(np.sum(batch['mask']!=max_logits.numpy())/ max_logits.numpy().size, tf.float16)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            max_logits = tf.argmax(logits, axis=-1)
            max_logits = max_logits[..., tf.newaxis]
            train_acc_metric_custom += np.sum(batch['mask']==max_logits.numpy()) / max_logits.numpy().size
            train_acc_metric(batch['mask'], max_logits)
            # train_rec_metric.update_state(batch['mask'], max_logits)
            # train_true_pos_metric(batch['mask'], logits)


            if step % epoch_log_n == 0:
                print(f"Training loss (for one batch) at step {step}: {loss_value:#.2f}")
                print(f"Seen so far: {(step + 1) * batch_size} samples")

        progress_tracker.on_epoch_end(epoch,
                                      loss=round(float(loss_value), 2),
                                      metrics=[
                                          Metric(metric=train_acc_metric_custom / epoch_steps, name='custom_accuracy'),
                                          Metric(metric=train_acc_metric.result(), name= 'accuracy'),
                                               ], #, train_rec_metric.result(), train_true_pos_metric.result(),
                                      show_img=False)
        train_acc_metric.reset_states()
        # train_rec_metric.reset_states()
        # train_true_pos_metric.reset_states()

    progress_tracker.on_train_end()
