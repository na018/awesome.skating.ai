import io
import os
import time
from enum import Enum
from pathlib import Path
from typing import List, Tuple

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.layer_utils import count_params
from tensorflow import keras, summary

# # needed for savefig to work
# matplotlib.use("Qt4Agg")

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
            body_mask[(img == i).all(axis=-1)] = key

    return body_mask


def kps_upscale_reshape(shape: Tuple[int, int], kps: np.array):
    kps = np.reshape(kps, (kps.size // 2, -1)).copy()
    kps[:, 0] *= shape[0]
    kps[:, 1] *= shape[1]
    kps[:, 0] = np.clip(kps[:, 0], 0, shape[1])
    kps[:, 1] = np.clip(kps[:, 1], 0, shape[0])

    return kps


def set_gpus(version: int):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            # tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
            tf.config.experimental.set_visible_devices(gpus[version], 'GPU')
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


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer_kps.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer_kps before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))


class Metric(object):
    def __init__(self, name: str, value=None, max_size=None, smooth_weight=0.7, diff=0.02):
        self.name = name
        self.value = value
        self.metrics = []
        self.smoothed = []
        self.max_size = max_size
        self.smooth_weight = smooth_weight
        self.diff = diff

    def append(self, value: float):
        if self.max_size and (len(self.metrics) - 1) > self.max_size:
            self.metrics.pop(0)

        self.metrics.append(value)

    def expo_smooth_avg(self):
        last = self.metrics[0]
        smoothed = []
        for point in self.metrics:
            smoothed_val = last * self.smooth_weight + (1 - self.smooth_weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        self.smoothed = smoothed

    def is_curve_steep(self):
        self.expo_smooth_avg()
        curve_start_avg = np.array(self.smoothed)[5:].mean()
        curve_end_avg = np.array(self.smoothed)[-5:].mean()
        curve_diff = np.abs(curve_start_avg - curve_end_avg)
        print(f"curve_start_avg: [{curve_start_avg}] curve_end_avg: [{curve_end_avg}] diff: [{curve_diff}]")

        return curve_diff > self.diff

    def get_median(self, reset: bool = True):
        if self.value is not None:
            median = self.value
        else:
            median = np.median(self.metrics)
        if reset:
            self.metrics = []

        return median


def plot2img(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class DisplayCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, model: tf.keras.Model,
                 epochs=5, gpu: int = 1,
                 log_dir='logs', sub_dir="",
                 **kwargs):
        super().__init__(log_dir=f"{log_dir}/model/", histogram_freq=1, profile_batch='500,520', **kwargs)
        # log_dir = f"{log_dir}/model/train", histogram_freq = 1, profile_batch = '500,520', ** kwargs
        self.sub_dir = sub_dir

        self.model = model

        self.epochs = epochs
        # self._file_writer = tf.summary.create_file_writer(f"{log_dir}/scalars/train")
        self._file_writer = tf.summary.create_file_writer(f"{log_dir}/{sub_dir}/scalars/train")
        self.gpu = gpu

        self.histogram_freq = 10

    def track_metrics_on_train_start(self, model: tf.keras.Model, name, optimizer_name, loss_function, learning_rate,
                                     train_bg: bool, train_hp: bool, train_kps: bool, training_time, epochs,
                                     epoch_steps, batch_size,
                                     description):
        text_summary = f'|name    |{name}    |\n|----|----|\n'
        trainable_params = count_params(model.trainable_weights)
        non_trainable_params = count_params(model.non_trainable_weights)
        layers = len(model.layers)

        for item in (['optimizer_name', optimizer_name],
                     ['epochs:epoch_steps:batch_size', f"{epochs}:{epoch_steps}:{batch_size}"],
                     ['loss_function', loss_function],
                     ['train_bg', train_bg],
                     ['train_body_parts', train_hp],
                     ['train_keypoints', train_kps],
                     ['learning_rate', str(learning_rate)],
                     ['all_params', f"{trainable_params + non_trainable_params:,}"],
                     ['trainable_params', f"{trainable_params:,}"],
                     ['non_trainable_params', f"{non_trainable_params:,}"],
                     ['layers', layers],
                     ['training_time_1_epoch', training_time],
                     ['description', description],
                     ):
            text_summary += f"|**{item[0].ljust(30)[:30]}**|{str(item[1]).ljust(120)[:120]}|\n"
        print(text_summary)

        with self._file_writer.as_default():
            tf.summary.text(name, tf.constant(text_summary), description=text_summary, step=0)

    def track_img_on_epoch_end(self, img, epoch: int, metrics: List[Metric] = []):

        self.model.save_weights(
            f"{Path.cwd()}/ckpt{self.gpu}/{self.sub_dir}-{epoch}.ckpt")

        K.clear_session()

        with self._file_writer.as_default():
            tf.summary.image(f"{epoch}_{self.sub_dir}_img_train", img, step=epoch)

            for i, item in enumerate(metrics):
                tf.summary.scalar(item.name, item.get_median(), step=epoch)

        plt.close('all')
        self.on_epoch_end(epoch)

    def log_on_train_end(self):
        # clear_output(wait=True)
        print('train_end')

        print(f"\n\n{'=' * 100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' ' * 5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")
        self.on_train_end()


def create_dir(path: str, msg: str = ''):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise AssertionError(f"{path} already exists. {msg}")


def create_dir(path: str, msg: str = ''):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise AssertionError(f"{path} already exists. {msg}")
