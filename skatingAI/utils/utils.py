import numpy as np
import tensorflow as tf
from tensorflow import keras, summary, newaxis
from keras import backend as K
import itertools
from pathlib import Path
import random
import os

from skimage import draw
from matplotlib import pyplot as plt

path = Path.cwd()
amount_of_files = len(next(os.walk(f"{path}/Data/3dhuman/processed/numpy/rgbs"))[2])


def get_video_batches(batch_size):
    for i in itertools.count(1):
        counter = int(random.random() * 394) + 1
        video = np.load(f"{path}/Data/3dhuman/processed/numpy/rgbbs/{counter}.npz")['arr_0']
        mask = np.load(f"{path}/Data/3dhuman/processed/numpy/masks/{counter}.npz")['arr_0']
        mask_shape = np.array(mask.shape)
        mask = mask.reshape((*mask_shape, -1))

        randoms = np.random.randint(video.shape[0], size=1)
        video = video[randoms, :]

        mask = mask[randoms, :]
        print(f"[{i}]",'*'*100,f"[{counter}]-yield")
        yield video, mask


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def get_segmentation_body():
    return {
        'bg': [153, 153, 153],
        'Head': [128, 64, 0],
        'RUpArm': [128, 0, 128],
        'RForeArm': [128, 128, 255],
        'RHand': [255, 128, 128],
        'LUpArm': [0, 0, 255],
        'LForeArm': [128, 128, 0],
        'LHand': [0, 128, 0],
        'torso': [128, 0, 0],
        'RThigh': [128, 255, 128],
        'RLowLeg': [255, 255, 128],
        'RFoot': [255, 0, 255],
        'LThigh': [0, 128, 128],
        'LLowLeg': [0, 0, 128],
        'LFoot': [255, 128, 0]
    }


def mask2rgb(mask):
    img = mask.numpy()
    body_mask = np.zeros((*img.shape[:2], 3))
    sb = get_segmentation_body()

    for i, key in enumerate(sb.values()):
        if i > 0:
            body_mask[(img == i).all(axis=2)] = key

    return body_mask


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, sample_image, sample_mask, file_writer, epochs=5):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.model = model
        self.epochs = epochs
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None, show_img=False):
        print('epoch_end')

        # clear_output(wait=True)
        print("after clear output")

        self.model.save_weights(
            f"{Path.cwd()}/ckpt/hrnet-{epoch}.ckpt")
        print('saved weights')

        predicted_mask = create_mask(self.model.predict(self.sample_image[tf.newaxis, ...])[0])

        K.clear_session()
        fig = plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        display_imgs = [self.sample_image, self.sample_mask, predicted_mask]

        for i, img in enumerate(display_imgs):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            if show_img:
                plt.imshow(tf.keras.preprocessing.image.array_to_img(display_imgs[i]))

        fig.savefig(f"{epoch}_train.png")

        summary_images = tf.constant([self.sample_image.numpy(), mask2rgb(self.sample_mask), mask2rgb(predicted_mask)])
        summary_images = tf.cast(summary_images, tf.uint8)

        tf.summary.image(f"{epoch}_training", summary_images, step=epoch, max_outputs=3)

        print(f"\nSimple Prediction after epoch {epoch + 1}")

    def on_train_end(self, logs=None):
        # clear_output(wait=True)
        print('train_end')
        # K.clear_session()
        print(f"\n\n{'=' * 100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' ' * 5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")
