import argparse
import math
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.nets.hrnet import v0, v1, v2, v3, v4, v5, v6, v7
from skatingAI.nets.mobilenet import v0 as mobilenetv0
from skatingAI.utils.DsGenerator import Frame, Mask, DsGenerator
from skatingAI.utils.utils import create_mask, create_dir

VERSIONS = {
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


class Evaluator():
    def __init__(self, weight_counter=5, nn_version='v2', name='', subfolder='', c=''):
        self.dir_img_eval = f"{os.getcwd()}/evaluate/img/{nn_version}_{weight_counter}_{name}"
        self.weight_path = f"{os.getcwd()}/ckpt{c}/{subfolder}hrnet-{weight_counter}.ckpt"
        self.frame, self.mask = self._get_frame()

        self.NN = VERSIONS[nn_version]

        create_dir(self.dir_img_eval, 'Please add a unique name.')

    def _get_frame(self, video_n=1, frame_n=1) -> Tuple[Frame, Mask]:
        self.generator = DsGenerator(resize_shape=(240, 320))
        sample_pair = next(self.generator.get_next_pair())

        return sample_pair['frame'], sample_pair['mask']

    def draw_prediction(self, layer: tf.keras.layers, predicted_mask: Mask, layer_n: int):
        fig = plt.figure(figsize=(7, 5))
        fig.suptitle(f"predicted_{layer.name}")
        title = ['Input', f'Predicted: {layer.name}']
        display_imgs = [self.frame,
                        tf.keras.preprocessing.image.array_to_img(predicted_mask)]
        for i, img in enumerate(display_imgs):
            ax = fig.add_subplot(1, 2, i + 1)
            ax.set_title(title[i], fontsize='small', alpha=0.6, color='blue')
            plt.imshow(display_imgs[i])

        fig.canvas.set_window_title(layer.name)
        plt.savefig(f"{self.dir_img_eval}/{layer_n}_predicted_mask_{layer.name}")

    def draw_feature_maps(self, layer: tf.keras.layers, feature_maps: Mask, layer_n: int):
        square = int(math.sqrt(feature_maps.shape[-1]))

        if feature_maps.shape[-1] % square != 0:
            square += 1
        fig = plt.figure(figsize=(7, 6.5))
        fig.suptitle(layer.name)

        for i in range(square * square):
            ax = fig.add_subplot(square, square, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"[{i}]", fontsize='small', alpha=0.6, color='blue')

            # plot filter channel in grayscale
            if i <= feature_maps.shape[-1]:
                plt.imshow(feature_maps[:, :, i - 1], cmap='inferno')
            else:
                plt.imshow(feature_maps[:, :, feature_maps.shape[-1] - 1], cmap='Greys')

        # show the figure
        fig.canvas.set_window_title(layer.name)
        plt.savefig(f"{self.dir_img_eval}/{layer_n}_{layer.name}")

    def show_featuremaps(self):
        img_shape = self.frame.shape
        n_classes = np.max(self.mask) + 1

        hrnet = self.NN(img_shape, int(n_classes))
        print(hrnet.model.summary())
        hrnet.model.load_weights(self.weight_path)
        hrnet.model.trainable = False

        for i, layer in enumerate(hrnet.model.layers):
            print(layer.name)

            if 'conv' in layer.name or 'output' in layer.name or 'add' in layer.name or 'concat' in layer.name:
                model = tf.keras.models.Model(inputs=hrnet.inputs, outputs=hrnet.model.layers[i].output)

                for layer in model.layers:
                    layer.trainable = False

                print(model.summary())
                print(f"Trainable weights: {len(model.trainable_weights)}")
                print('start the prediction fun part')
                print(layer.name)
                print('*' * 100)

                feature_maps = model.predict([[self.frame]])[0]
                predicted_mask = create_mask(feature_maps)

                self.draw_prediction(layer, predicted_mask, i)
                self.draw_feature_maps(layer, feature_maps, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate featuremaps of trained model')
    parser.add_argument('--wcounter', default=1595, help='Number of weight')
    parser.add_argument('--v', default='v7', help='version of hrnet')
    parser.add_argument('--c', default='2', help='gpu number the net has trained on')
    parser.add_argument('--name', default='adam_1595_mask2', help='unique name to save images in')
    args = parser.parse_args()

    Evaluator(weight_counter=args.wcounter, nn_version=args.v, name=args.name, c=args.c).show_featuremaps()
