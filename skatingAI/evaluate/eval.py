import argparse
import math
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.nets.hrnet import v0, v1, v2, v3, v4, v5, v6, v7
from skatingAI.nets.mobilenet import v0 as mobilenetv0
from skatingAI.utils.DsGenerator import Frame, Mask


class Evaluater():
    def __init__(self, weight_counter=5, nn_version='v2', name='', subfolder='', c=''):
        self.dir_img_eval = f"{os.getcwd()}/evaluate/img/{nn_version}_{weight_counter}_{name}"
        self.weight_path = f"{os.getcwd()}/ckpt{c}/{subfolder}hrnet-{weight_counter}.ckpt"

        versions = {
            'v0': v0.HRNet,
            'v1': v1.HRNet,
            'v2': v2.HRNet,
            'v3': v3.HRNet,
            'v4': v4.HRNet,
            'v5': v5.HRNet,
            'v6': v6.HRNet,
            'v7': v7.HRNet,
            'u0': mobilenetv0.MobileNetV2,
        }
        self.NN = versions[nn_version]

        if not os.path.exists(self.dir_img_eval):
            os.makedirs(self.dir_img_eval)
        else:
            raise AssertionError(f"{self.dir_img_eval} already exists. Please add a unique name.")

    def _get_frame(self, video_n=1, frame_n=1) -> Tuple[Frame, Mask]:
        video_path_rgbs: str = f"{os.getcwd()}/Data/3dhuman/processed/numpy/rgbb"
        mask_path: str = f"{os.getcwd()}/Data/3dhuman/processed/numpy/masks"
        frame: np.ndarray = np.load(f"{video_path_rgbs}/{video_n}.npz")['arr_0'][frame_n]
        mask: np.ndarray = np.load(f"{mask_path}/{video_n}.npz")['arr_0'][frame_n]
        return frame, mask

    def show_featuremaps(self):
        frame, mask = self._get_frame()
        img_shape = frame.shape
        n_classes = np.max(mask) + 1

        hrnet = self.NN(img_shape, n_classes)
        print(hrnet.model.summary())
        hrnet.model.load_weights(self.weight_path)

        for i, layer in enumerate(hrnet.model.layers):
            print(layer.name)

            if 'conv' in layer.name or 'output' in layer.name or 'add' in layer.name or 'concat' in layer.name:
                model = tf.keras.models.Model(inputs=hrnet.model.inputs, outputs=hrnet.model.layers[i].output)

                for layer in model.layers:
                    layer.trainable = False

                print(model.summary())
                print(f"Trainable weights: {len(model.trainable_weights)}")
                print('start the prediction fun part')

                feature_maps = model.predict([[frame]])[0]
                print(feature_maps.shape)

                print('*' * 100)
                print(layer.name)
                print('*' * 100)

                # fig, ax = plt.subplots(figsize=(15, 8), title=layer.name)

                square = int(math.sqrt(feature_maps.shape[-1]))

                if feature_maps.shape[-1] % square != 0:
                    square += 1
                fig, ax = plt.subplots(square, square, figsize=(7, 6.5))
                fig.suptitle(layer.name)

                ix = 1
                for i in range(square):
                    for j in range(square):
                        # specify subplot and turn off axis
                        ax[i, j] = plt.subplot(square, square, ix)
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])
                        ax[i, j].set_title(f"[{i},{j}]", fontsize='small', alpha=0.6, color='blue')
                        # plot filter channel in grayscale

                        if ix <= feature_maps.shape[-1]:
                            plt.imshow(feature_maps[:, :, ix - 1], cmap='inferno')
                        else:
                            plt.imshow(feature_maps[:, :, feature_maps.shape[-1] - 1], cmap='Greys')
                        ix += 1

                # show the figure
                fig.canvas.set_window_title(layer.name)
                plt.savefig(f"{self.dir_img_eval}/{layer.name}")
                # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate featuremaps of trained model')
    parser.add_argument('--wcounter', default=1595, help='Number of weight')
    parser.add_argument('--v', default='v7', help='version of hrnet')
    parser.add_argument('--c', default='2', help='gpu number the net has trained on')
    parser.add_argument('--name', default='adam_1595', help='unique name to save images in')
    args = parser.parse_args()

    Evaluater(weight_counter=args.wcounter, nn_version=args.v, name=args.name, c=args.c).show_featuremaps()
