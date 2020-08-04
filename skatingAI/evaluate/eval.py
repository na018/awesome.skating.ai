import argparse
import math
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.nets import hrnet, bg, keypoint
from skatingAI.nets.mobilenet import v0 as mobilenetv0
from skatingAI.utils.DsGenerator import Frame, Mask, DsGenerator
from skatingAI.utils.utils import create_mask, create_dir

VERSIONS = {
    'v0': hrnet.v0.HPNet,
    'v1': hrnet.v1.HPNet,
    'v2': hrnet.v2.HPNet,
    'v3': hrnet.v3.HPNet,
    'v4': hrnet.v4.HPNet,
    'v5': hrnet.v5.HPNet,
    'v6': hrnet.v6.HPNet,
    'v7': hrnet.v7.HPNet,
    'u0': mobilenetv0.MobileNetV2,
}
VERSIONSBG = {
    'v0': bg.v0.BGNet,
    'v1': bg.v7.BGNet

}
VERSIONSKP = {
    'v0': keypoint.v0.KPDetector,
    'v1': keypoint.v1.KPDetector,
    'v2': keypoint.v2.KPDetector,
    'v3': keypoint.v3.KPDetector

}


class Evaluator():
    def __init__(self, weight_counter=5, nn_hp_version='v7', nn_bg_version='v0', nn_kps_version='v3', name='',
                 subfolder='', c='1'):
        self.dir_img_eval = f"{os.getcwd()}/evaluate/img/{nn_hp_version}_{weight_counter}_{name}"
        self.weight_path = f"{os.getcwd()}/ckpt{c}/{subfolder}hrnet-{weight_counter}.ckpt"
        self.frame, self.mask = self._get_frame()

        self.NNbg = VERSIONSBG[nn_bg_version]
        self.NNhp = VERSIONS[nn_hp_version]
        self.NNkps = VERSIONSKP[nn_kps_version]

        self.bg_weight_path = f"{os.getcwd()}/ckpt1/bg-{weight_counter}.ckpt"
        self.hp_weight_path = f"{os.getcwd()}/ckpt1/hp-{weight_counter}.ckpt"
        self.kps_weight_path = f"{os.getcwd()}/ckpt1/kps-{weight_counter}.ckpt"

        create_dir(self.dir_img_eval, 'Please add a unique name.')

    def _get_frame(self, video_n=1, frame_n=1) -> Tuple[Frame, Mask]:
        self.generator = DsGenerator(resize_shape_x=240)
        sample_pair = next(self.generator.get_next_pair())

        return sample_pair['frame'], sample_pair['mask_hp']

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

        bg_net = self.NNbg(img_shape, 2)
        bg_net.model.load_weights(self.bg_weight_path)
        bg_net.trainable = False

        hp_net = self.NNhp(img_shape, bgnet_input=bg_net.model, output_channels=9)
        hp_net.model.load_weights(self.hp_weight_path)
        hp_net.trainable = False
        kps_net = self.NNkps(img_shape, bgnet_input=bg_net.model, hrnet_input=hp_net.model, output_channels=12)
        kps_net.model.load_weights(self.kps_weight_path)
        kps_net.trainable = False

        print(kps_net.model.summary())

        for i, layer in enumerate(hp_net.model.layers):
            print(layer.name)

            if 'conv' in layer.name or 'output' in layer.name or 'add' in layer.name or 'concat' in layer.name:
                model = tf.keras.models.Model(inputs=hp_net.inputs, outputs=hp_net.model.layers[i].output)

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
    parser.add_argument('--wcounter', default=9400, help='Number of weight')
    parser.add_argument('--v_bg', default='v0', help='version of hrnet')
    parser.add_argument('--v_hp', default='v7', help='version of hrnet')
    parser.add_argument('--v-kps', default='v3', help='version of hrnet')
    parser.add_argument('--c', default='1', help='gpu number the net has trained on')
    parser.add_argument('--name', default='all_modules_hp_4', help='unique name to save images in')
    args = parser.parse_args()

    Evaluator(weight_counter=args.wcounter, nn_hp_version=args.v_hp,
              nn_bg_version=args.v_bg,
              nn_kps_version=args.v_kps,
              name=args.name, c=args.c).show_featuremaps()
