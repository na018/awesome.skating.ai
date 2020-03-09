from pathlib import Path

import tensorflow as tf
import numpy as np
from datetime import datetime

from skatingAI.nets.hrnet.hrnet import HRNet
from skatingAI.utils.DsGenerator import DsGenerator
from skatingAI.utils.utils import DisplayCallback, set_gpus, Metric, Logger
from skatingAI.utils.losses import GeneralisedWassersteinDiceLoss

if __name__ == "__main__":
    batch_size = 3
    prefetch_batch_buffer = 1
    epoch_steps = 32
    epoch_log_n = 5
    epochs = 512

    set_gpus()

    generator = DsGenerator()

    sample_pair = next(generator.get_next_pair())
    sample_frame = sample_pair['frame']
    sample_mask = sample_pair['mask']

    img_shape = sample_frame.shape
    n_classes = np.max(sample_mask) + 1

    train_ds = generator.buid_iterator(img_shape, batch_size, prefetch_batch_buffer)
    iter = train_ds.as_numpy_iterator()


    hrnet = HRNet(img_shape, n_classes)


    model = hrnet.model
    model.summary()
    #model.load_weights('./ckpt/hrnet-255.ckpt')
    tf.keras.utils.plot_model(
        model, to_file='nadins_hrnet_1.png', show_shapes=True, expand_nested=False)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, amsgrad=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn = GeneralisedWassersteinDiceLoss(n_classes)

    train_acc_metric = tf.keras.metrics.Accuracy()
    train_acc_metric_custom = 0
    # train_rec_metric = tf.keras.metrics.Recall()
    # train_true_pos_metric = tf.keras.metrics.TruePositives()

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics/{datetime.now().strftime('%Y_%m_%d__%H_%M')}")
    file_writer.set_as_default()
    progress_tracker = DisplayCallback(model, sample_frame, sample_mask, file_writer, epochs)
    logger=Logger(log=False)
    log2=Logger(log=True)

    for epoch in range(epochs):
        log2.log(message=f"Start of epoch {epoch}", block=True)
        train_acc_metric_custom = 0

        for step in range(epoch_steps):
            logger.log(message=f"Step {step}", block=True)
            batch = next(iter)
            logger.log(message=f"got batch")

            with tf.GradientTape() as tape:
                logits = model(batch['frame'], training=True)
                loss_value = loss_fn(batch['mask'], logits)

            logger.log(message=f"Calculated Logits & loss value")
            grads = tape.gradient(loss_value, model.trainable_weights)
            logger.log(message=f"Calculated Grads")

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            logger.log(message=f"Optimized gradients")
            max_logits = tf.argmax(logits, axis=-1)
            max_logits = max_logits[..., tf.newaxis]
            # train_acc_metric_custom += np.sum(batch['mask']==max_logits.numpy()) / max_logits.numpy().size
            train_acc_metric(batch['mask'], max_logits)
            # train_rec_metric.update_state(batch['mask'], max_logits)
            # train_true_pos_metric(batch['mask'], logits)


            # if step % epoch_log_n == 0:
            #     logger.log(f"Training loss (for one batch) at step {step}: {tf.reduce_sum(loss_value).numpy():#.2f}")
            #     logger.log(f"Seen so far: {(step + 1) * batch_size} samples")

        if epoch % epoch_log_n == 0:
            progress_tracker.on_epoch_end(epoch,
                                          loss=round(tf.reduce_sum(loss_value).numpy(), 2),
                                          metrics=[
                                              # Metric(metric=train_acc_metric_custom / epoch_steps, name='custom_accuracy'),
                                              Metric(metric=train_acc_metric.result(), name= 'accuracy'),
                                                   ], #, train_rec_metric.result(), train_true_pos_metric.result(),
                                          show_img=False)
        train_acc_metric.reset_states()
        # train_rec_metric.reset_states()
        # train_true_pos_metric.reset_states()

    progress_tracker.on_train_end()
    logger.log_end()
