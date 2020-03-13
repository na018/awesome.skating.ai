import math
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.nets.hrnet.hrnet import HRNet
from skatingAI.utils.DsGenerator import Frame, Mask


class Evaluater():
    def __init__(self):
        pass

    def _get_frame(self, video_n=1, frame_n=1) -> Tuple[Frame, Mask]:
        video_path_rgbs: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/rgbbs"
        mask_path: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/masks"
        frame: np.ndarray = np.load(f"{video_path_rgbs}/{video_n}.npz")['arr_0'][frame_n]
        mask: np.ndarray = np.load(f"{mask_path}/{video_n}.npz")['arr_0'][frame_n]
        return frame, mask

    def show_featuremaps(self, weight_counter=5):
        frame, mask = self._get_frame()
        img_shape = frame.shape
        n_classes = np.max(mask) + 1

        hrnet = HRNet(img_shape, n_classes)
        print(hrnet.model.summary())
        hrnet.model.load_weights(f"{Path.cwd()}/ckpt/hrnet-{weight_counter}.ckpt")
        model = tf.keras.models.Model(inputs=hrnet.model.inputs, outputs=hrnet.model.layers[-2].output)

        for i, layer in enumerate(model.layers):
            print(i, layer)
            layer.trainable = False

        print(model.summary())
        print(f"Trainable weights: {len(model.trainable_weights)}")
        print('start the prediction fun part')

        feature_maps = model.predict([[frame]])[0]
        print(feature_maps.shape)

        fig = plt.figure(figsize=(15, 8))
        square = int(math.sqrt(feature_maps.shape[-1]))
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn off axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(feature_maps[:, :, ix - 1], cmap='inferno')
                ix += 1

        # show the figure
        plt.show()


Evaluater().show_featuremaps()
