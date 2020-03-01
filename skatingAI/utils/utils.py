from pathlib import Path

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


def set_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
            tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


class DisplayCallback(object):
    def __init__(self, model, sample_image, sample_mask, file_writer, epochs=5):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.model = model
        self.epochs = epochs
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, loss, accuracy, show_img=False):
        print('epoch_end')

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
            plt.draw()
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_imgs[i]))

        if show_img:
            plt.show()
        fig.savefig(f"{path}/img_train/{epoch}_train.png")

        summary_images = tf.constant([self.sample_image.numpy(), mask2rgb(self.sample_mask), mask2rgb(predicted_mask)])
        summary_images = tf.cast(summary_images, tf.uint8)

        tf.summary.image(f"{epoch}_training", summary_images, step=epoch, max_outputs=3)
        tf.summary.scalar('loss', loss, step=epoch)
        tf.summary.scalar('accuracy', accuracy, step=epoch)

        print(f"\nSimple Prediction after epoch {epoch + 1}")

    def on_train_end(self):
        # clear_output(wait=True)
        print('train_end')
        # K.clear_session()
        print(f"\n\n{'=' * 100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' ' * 5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")
