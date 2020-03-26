import argparse
from collections import namedtuple

ArgsNamespace = namedtuple('ArgNamespace', ['gpu', 'name', 'wcounter'])

from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from skatingAI.nets.hrnet.v7 import HRNet
from skatingAI.utils.DsGenerator import DsGenerator
from skatingAI.utils.losses import GeneralisedWassersteinDiceLoss
from skatingAI.utils.utils import DisplayCallback, set_gpus, Metric, Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train nadins awesome network :)')
    parser.add_argument('--gpu', default=1, help='Which gpu shoud I use?')
    parser.add_argument('--name', default="hrnet_v7", help='Name for training')
    parser.add_argument('--wcounter', default=2600, help='Weight counter')
    args: ArgsNamespace = parser.parse_args()

    batch_size = 3
    prefetch_batch_buffer = 1
    epoch_steps = 64
    epoch_log_n = 5
    epochs = 5555

    set_gpus(int(args.gpu))

    generator = DsGenerator(resize_shape=(240, 320))

    sample_pair = next(generator.get_next_pair())
    sample_frame = sample_pair['frame']
    sample_mask = sample_pair['mask']

    img_shape = sample_frame.shape
    n_classes = np.max(sample_mask) + 1

    train_ds = generator.build_iterator(img_shape, batch_size, prefetch_batch_buffer)
    iter = train_ds.as_numpy_iterator()

    hrnet = HRNet(img_shape, int(n_classes))

    model = hrnet.model
    model.summary()
    if int(args.wcounter) != -1:
        model.load_weights(f"./ckpt{args.gpu}/hrnet-{args.wcounter}.ckpt")
        wcounter = int(args.wcounter)
    else:
        wcounter = 0
    tf.keras.utils.plot_model(
        model, to_file=f'nadins_{args.name}_e.png', show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=f'nadins_{args.name}.png', show_shapes=True, expand_nested=False)
    lr_start = 0.1
    optimizer_decay = 0.001

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8, amsgrad=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_start, momentum=0.9, decay=optimizer_decay, nesterov=True)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn = GeneralisedWassersteinDiceLoss(n_classes)

    train_acc_metric = tf.keras.metrics.Accuracy()
    train_acc_metric_custom = 0
    # train_rec_metric = tf.keras.metrics.Recall()
    # train_true_pos_metric = tf.keras.metrics.TruePositives()

    file_writer = tf.summary.create_file_writer(
        f"{Path.cwd()}/logs/metrics/{args.name}/{datetime.now().strftime('%Y_%m_%d__%H_%M')}")
    file_writer.set_as_default()
    progress_tracker = DisplayCallback(model, sample_frame, sample_mask, file_writer, epochs, args.gpu)
    logger = Logger(log=False)
    log2 = Logger(log=True)

    for epoch in range(wcounter, epochs):
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
            lr = lr_start * (1. / (1. + optimizer_decay * (epoch - wcounter) * epoch_steps))
            log2.log(message=f"Learning Rate: [{lr}]    loss: [{loss_value}]", block=False)

            progress_tracker.on_epoch_end(epoch,
                                          loss=tf.reduce_sum(loss_value).numpy(),
                                          metrics=[
                                              Metric(metric=loss_fn.correct_predictions.astype(np.float32),
                                                     name='correct_px'),
                                              Metric(metric=loss_fn.correct_body_part_pred.astype(np.float32),
                                                     name='correct_body_part_px'),
                                              Metric(metric=(
                                                      loss_fn.correct_body_part_pred / loss_fn.body_part_px_n_true)
                                                     .astype(np.float32),
                                                     name='accuracy_body_part'),
                                              Metric(metric=train_acc_metric.result(), name='accuracy'),
                                              Metric(metric=lr, name='learning_rate'),
                                          ],
                                          show_img=False)
            log2.log(message=f"Seen images: {generator.seen_samples}", block=True)
        train_acc_metric.reset_states()

    progress_tracker.on_train_end()
    logger.log_end()
