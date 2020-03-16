import time
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from tensorflow import keras, summary

path = Path.cwd()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

class BodyParts(Enum):
    bg = 0
    Head = 1
    RUpArm = 2
    RForeArm = 3
    RHand = 4
    LUpArm = 5
    LForeArm = 6
    LHand = 7
    torso = 8
    RThigh = 9
    RLowLeg = 10
    RFoot = 11
    LThigh = 12
    LLowLeg = 13
    LFoot = 14


body_part_classes = {
    BodyParts.bg.name: 0,
    BodyParts.Head.name: 1,
    BodyParts.RUpArm.name: 2,
    BodyParts.RForeArm.name: 3,
    BodyParts.RHand.name: 4,
    BodyParts.LUpArm.name: 2,
    BodyParts.LForeArm.name: 3,
    BodyParts.LHand.name: 4,
    BodyParts.torso.name: 5,
    BodyParts.RThigh.name: 6,
    BodyParts.RLowLeg.name: 7,
    BodyParts.RFoot.name: 8,
    BodyParts.LThigh.name: 6,
    BodyParts.LLowLeg.name: 7,
    BodyParts.LFoot.name: 8
}

segmentation_class_colors = {
    BodyParts.bg.name: [153, 153, 153],
    BodyParts.Head.name: [128, 64, 0],
    BodyParts.RUpArm.name: [128, 0, 128],
    BodyParts.RForeArm.name: [128, 128, 255],
    BodyParts.RHand.name: [255, 128, 128],
    BodyParts.LUpArm.name: [0, 0, 255],
    BodyParts.LForeArm.name: [128, 128, 0],
    BodyParts.LHand.name: [0, 128, 0],
    BodyParts.torso.name: [128, 0, 0],
    BodyParts.RThigh.name: [128, 255, 128],
    BodyParts.RLowLeg.name: [255, 255, 128],
    BodyParts.RFoot.name: [255, 0, 255],
    BodyParts.LThigh.name: [0, 0, 128],
    BodyParts.LLowLeg.name: [0, 128, 128],
    BodyParts.LFoot.name: [255, 128, 0]
}


def mask2rgb(mask):
    img = mask.numpy()
    body_mask = np.zeros((*img.shape[:2], 3))

    for i, key in enumerate(segmentation_class_colors.values()):
        if i > 0:
            body_mask[(img == i).all(axis=2)] = key

    return body_mask


def set_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            # tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
            tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            Logger().log(f"{len(gpus)} Physical GPUs {len(logical_gpus)} Logical GPU", block=True)

            # strategy = tf.distribute.MirroredStrategy()
            # return strategy
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


class Timer(object):
    def __init__(self):
        self._time_start = time.perf_counter()
        self._moment = time.perf_counter()

    def get_moment(self) -> float:
        duration = time.perf_counter() - self._moment
        self._moment = time.perf_counter()

        return duration

    def total_time(self) -> float:
        return time.perf_counter() - self._time_start


class Logger(object):
    def __init__(self, log=True):
        self._log = log
        self._Timer = Timer()

    def log(self, message: str, block=False) -> float:
        if self._log:
            if block:
                print('*' * 100)
            print(f"[{self._Timer.get_moment():#.2f}s] {message}")
        return self._Timer.get_moment()

    def log_end(self):
        print('*' * 100)
        print(f"Complete execution duration: {self._Timer.total_time() / 60:#.2f} minutes")


class Metric(object):
    def __init__(self, name: str, metric: tf.keras.metrics):
        self.name = name
        self.metric = metric


class DisplayCallback(object):
    def __init__(self, model, sample_image, sample_mask, file_writer, epochs=5):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.model = model
        self.epochs = epochs
        self.file_writer = file_writer

    def on_epoch_end(self, epoch: int, loss: float, metrics: List[Metric], show_img=False):

        self.model.save_weights(
            f"{Path.cwd()}/ckpt/hrnet-{epoch}.ckpt")

        predicted_mask = create_mask(self.model.predict(self.sample_image[tf.newaxis, ...])[0])

        K.clear_session()
        fig = plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        display_imgs = [tf.keras.preprocessing.image.array_to_img(self.sample_image),
                        tf.keras.preprocessing.image.array_to_img(self.sample_mask),
                        tf.keras.preprocessing.image.array_to_img(predicted_mask)]

        for i, img in enumerate(display_imgs):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.draw()
            plt.imshow(display_imgs[i])

        if show_img:
            plt.show()
        fig.savefig(f"{path}/img_train/{epoch}_train.png")

        summary_images = [self.sample_image, mask2rgb(self.sample_mask), mask2rgb(predicted_mask)]
        #summary_images = tf.cast(summary_images, tf.uint8)

        tf.summary.image(f"{epoch}_training", summary_images, step=epoch, max_outputs=3)
        tf.summary.scalar('loss', loss, step=epoch)

        for i, item in enumerate(metrics):
            tf.summary.scalar(item.name, item.metric, step=epoch)

    def on_train_end(self):
        # clear_output(wait=True)
        print('train_end')
        # K.clear_session()
        print(f"\n\n{'=' * 100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' ' * 5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")
