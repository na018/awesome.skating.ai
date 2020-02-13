from enum import Enum
import numpy as np
import tensorflow as tf
from tensorflow import keras, summary, newaxis
from keras import backend as K
from IPython.display import clear_output
from pathlib import Path
import random
import os
import cv2

from skimage import draw
from matplotlib import pyplot as plt

path = Path.cwd()
amount_of_files = len(next(os.walk(f"{path}/Data/3dhuman/processed/numpy/rgbs"))[2])

def get_video_batches(amount=1, level=1):

    for i in range(amount):
        counter = int(random.random()*394)
        video =  np.load(f"{path}/Data/3dhuman/processed/numpy/rgbbs/{counter}.npz")['arr_0']
        mask = np.load(f"{path}/Data/3dhuman/processed/numpy/masks/{counter}.npz")['arr_0']

        mask = mask.reshape((mask.shape[0],480,640,-1))
        yield video, mask




class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, sample_image, sample_mask, file_writer, epochs=5):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.model = model
        self.epochs = epochs
        self.file_writer = file_writer


    def on_epoch_end(self, epoch, logs=None):
        print('epoch_end')

        clear_output(wait=True)
        # show_predictions(self.model, self.sample_image,
        #                  self.sample_mask, name=f"prediction-{epoch}")

        self.model.save_weights(
            f"{Path.cwd()}/ckpt/hrnet-{epoch}.ckpt")
        print('saved weights')

        K.clear_session()
        #plt.imshow(self.sample_mask)   #
        # for i, img in enumerate([self.sample_image[newaxis, ...], self.model.predict(self.sample_image[newaxis, ...])]):
        #     summary.image(f"Training_{epoch}", img, step=i)

        # tf.summary.image(f"Training_{epoch}", [
        #                  self.sample_image, self.sample_mask, self.model.predict(self.sample_image[tf.newaxis, ...])], step=0)

        print(f"\nSimple Prediction after epoch {epoch+1}")

    def on_train_end(self, logs=None):
        #clear_output(wait=True)
        print('train_end')
        #K.clear_session()
        print(f"\n\n{'='*100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' '*5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")


