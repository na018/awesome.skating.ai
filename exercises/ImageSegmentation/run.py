from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import matplotlib.pyplot as plt

import argparse

import tensorflow_datasets as tfds
from ImageSegmentation.utils import load_image_test, load_image_train, display_image, \
    show_predictions, DisplayCallback, calculate_loss
from ImageSegmentation.nets.mobilenet_v2 import decoder, encoder, unet_model
from ImageSegmentation.evaluate import evaluate
import tensorflow as tf
from pathlib import Path


sys.path.append(
    '/home/nadin-katrin/CodeProjects/00_skating_ai/awesome.skating.ai/exercises')
tfds.disable_progress_bar()


def train(buffer_size, batch_size, epochs, n_subsplits, epoch_steps=100,  OUTPUT_CHANNELS=3):

    # download datasets
    dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)
    train_length = info.splits['train'].num_examples

    train = dataset['train'].map(
        load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(batch_size)

    # for image, mask in train.take(1):
    #     sample_image, sample_mask = image, mask
    # display_image([sample_image, sample_mask])

    down_stack = encoder()
    up_stack = decoder()

    # train the model
    model = unet_model(OUTPUT_CHANNELS, down_stack, up_stack)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # save model architecture
    tf.keras.utils.plot_model(
        model, to_file='unet_architecture_1.png', show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file='unet_architecture_2.png', show_shapes=True, expand_nested=False)

    for image, mask in train.take(1):
        sample_image, sample_mask = image, mask
    show_predictions(model, sample_image, sample_mask)

    file_writer = tf.summary.create_file_writer(f"{Path.cwd()}/logs/metrics")
    file_writer.set_as_default()

    validation_steps = info.splits['test'].num_examples//batch_size//n_subsplits
    model_history = model.fit(train_dataset,
                              epochs=epochs,
                              steps_per_epoch=epoch_steps,
                              validation_steps=validation_steps,
                              validation_data=test_dataset,
                              callbacks=[
                                  DisplayCallback(
                                      model, sample_image, sample_mask, file_writer),
                                  tf.keras.callbacks.TensorBoard(
                                      log_dir=f"{Path.cwd()}/logs")
                              ])

    calculate_loss(model_history, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a simple neural net to recognize number images from the MNIST dataset and apply the correct labels')
    parser.add_argument('--epochs', default=5,
                        help='Amount of batches the net trains on')
    parser.add_argument('--epoch_steps', default=50,
                        help='Amount of batches the net trains on')
    parser.add_argument('--n_subsplits', default=5,
                        help='Amount of subsplits')
    parser.add_argument('--buffer_size', default=500,
                        help='Amount of batches the net trains on')
    parser.add_argument('--batch_size', default=16,
                        help='Number of training samples inside one batch')
    parser.add_argument('--train', default=True,
                        help='Run train or evaluate script.')

    args = parser.parse_args()

    if args.train:
        train(args.buffer_size, args.batch_size, args.epochs,
              args.n_subsplits, epoch_steps=args.epoch_steps)
    else:
        evaluate()
