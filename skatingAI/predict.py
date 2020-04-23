import argparse
import os
from collections import namedtuple
from datetime import datetime
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from skatingAI.evaluate.eval import VERSIONS
from skatingAI.utils.DsGenerator import DsGenerator, Frame, Mask, DsPair
from skatingAI.utils.utils import create_mask, mask2rgb, create_dir, Logger, set_gpus


class Predictor(object):
    def __init__(self, weight_counter: int, nn_version: str, name: str, gpu: int,
                 random_video: bool, random_image: bool, img_path: str, video_path: str, save_path: str):
        """predicts images or videos from specified network and weight

        Args:
            weight_counter:
            nn_version:
            name:
            gpu:
            random_video:
            random_image:
            img_path:
            video_path:
            save_path:
        """
        self.weight_counter = weight_counter
        self.NN_VERSION = VERSIONS[nn_version]
        self.name = name
        self.gpu = gpu
        self.random_video = random_video
        self.random_image = random_image
        self.img_path = img_path
        self.video_path = video_path
        self.save = save_path
        self.DsGenerator = DsGenerator(resize_shape_x=(240, 320), rgb=False,
                                       single_random_frame=random_image or len(img_path) > 1)
        self.weight_path = f"{os.getcwd()}/ckpt{gpu}/hrnet-{weight_counter}.ckpt"
        self.dir_save = f"{os.getcwd()}/predictions/{nn_version}_{weight_counter}"
        self.Logger = Logger()
        self.file_name = self._prepare_file_saving()

    def _prepare_file_saving(self) -> str:
        try:
            create_dir(self.dir_save)
        except:
            self.Logger.log(f"File will be saved in {self.dir_save}/{self.name}")

        return f"{self.name}/{datetime.now().strftime('%Y_%m_%d__%H_%M')}"

    def draw_prediction(self, frame: Frame, mask: Mask, predicted_mask: Mask, frame_n: int = 0):
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(f'Prediction {self.name}')
        title = ['Input', 'mask', f'Predicted: {self.name}']
        display_imgs = [frame, mask,
                        tf.keras.preprocessing.image.array_to_img(predicted_mask)]
        for i, img in enumerate(display_imgs):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_title(title[i], fontsize='small', alpha=0.6, color='blue')
            plt.imshow(display_imgs[i])

        fig.canvas.set_window_title(self.name)
        fig.savefig(f"{self.dir_save}/{self.file_name}_{frame_n}.jpg")
        plt.show()

        plt.close('all')

    def _predict_img(self, model: tf.keras.Model, prediction_img_amount: int):
        pairs = {'frames': [], 'masks': []}
        for _ in range(prediction_img_amount):
            sample_pair = next(self.DsGenerator.get_next_pair())
            pairs['frames'].append(sample_pair['frame'])
            pairs['masks'].append(sample_pair['mask'])

        feature_maps = model.predict([pairs['frames']])
        for i, feature_map in enumerate(feature_maps):
            predicted_mask = create_mask(feature_map)
            self.draw_prediction(pairs['frames'][i],
                                 np.reshape(pairs['masks'][i], pairs['masks'][i].shape[:-1]),
                                 predicted_mask, i)

    def _create_video(self, frames, masks, feature_maps):
        offset = 10
        width_total = feature_maps[0].shape[1] * 3 + offset * 4
        height_total = feature_maps[1].shape[0]
        new_img = Image.new('RGB', (width_total, height_total))

        for i, feature_map in enumerate(feature_maps):
            images = [
                tf.keras.preprocessing.image.array_to_img(frames[i]),
                tf.keras.preprocessing.image.array_to_img(mask2rgb(tf.constant(masks[i]))),
                tf.keras.preprocessing.image.array_to_img(mask2rgb(create_mask(feature_map))),
            ]

            title = ['Input', 'mask', f'prediction']
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/open-sans/OpenSans-Semibold.ttf', 15)
            x_offset = 0
            for j, img in enumerate(images):
                new_img.paste(img, (x_offset, 0))
                d = ImageDraw.Draw(new_img)
                d.text((x_offset + feature_map.shape[1] / 2.4, 5), title[j], font=fnt, fill=(255, 255, 255))
                x_offset += feature_map.shape[1] + offset

            if i == 0:
                out = cv2.VideoWriter(f"{self.dir_save}/{self.file_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 12,
                                      new_img.size)
            out.write(np.array(new_img))
            self.Logger.log(f"Ad frame {i} to video writer")
        out.release()
        Logger.log('successfully created video')

    def _predict_video(self, model, sample_pair):
        pairs = {'frames': [], 'masks': []}

        for i in range(sample_pair['size'] - 1):
            sample_pair = next(self.DsGenerator.get_next_pair(frame_i=i))
            pairs['frames'].append(sample_pair['frame'])
            pairs['masks'].append(sample_pair['mask'])
            Logger.log(f"Got frame [{i}]")

        feature_maps = model.predict([pairs['frames']])
        Logger.log('Prediction finished.\n start to generate video', block=True)
        self._create_video(pairs['frames'], pairs['masks'], feature_maps)

    def _get_model(self) -> Tuple[tf.keras.Model, Union[DsPair, DsPair]]:
        sample_pair = next(self.DsGenerator.get_next_pair())
        frame: Frame = sample_pair['frame']
        mask: Mask = sample_pair['mask']

        img_shape = frame.shape
        n_classes = np.max(mask) + 1

        NN = self.NN_VERSION(img_shape, int(n_classes))
        model: tf.keras.Model = NN.model
        model.load_weights(self.weight_path)
        model.trainable = False

        return model, sample_pair

    def predict(self, prediction_img_amount: int = 2):
        """start the prediction process.

        Args:
            prediction_img_amount: Amount of images which will be predicted.
        """
        set_gpus(self.gpu)
        model, sample_pair = self._get_model()
        if self.random_image:
            self._predict_img(model, prediction_img_amount)
        elif self.random_video:
            self._predict_video(model, sample_pair)
        elif len(self.img_path) > 1:
            # todo: load image
            pass
        elif len(self.video_path) > 1:
            # todo: load video and extract frames
            pass


if __name__ == "__main__":
    ArgsNamespace = namedtuple('ArgNamespace',
                               ['gpu', 'name', 'wcounter', 'v', 'video', 'image', 'random_video', 'random_image',
                                'save'])

    parser = argparse.ArgumentParser(
        description='Predict body parts from images or videos')
    parser.add_argument('--wcounter', default=1595, help='Number of weight')
    parser.add_argument('--v', default='v7', help='version of hrnet')
    parser.add_argument('--gpu', default='1', help='gpu number the net has trained on')
    parser.add_argument('--name', default='', help='unique name to save to save video/image')
    parser.add_argument('--video', default='/', help='absolute path to video file', type=bool)
    parser.add_argument('--image', default='/', help='absolute path to image file')
    parser.add_argument('--random_video', default=False, help='Random video')
    parser.add_argument('--random_image', default=True, help='Random image', type=bool)
    parser.add_argument('--save', default='', help='Path to save prediction in')
    args: ArgsNamespace = parser.parse_args()

    Predictor(args.wcounter, args.v, args.name, args.gpu, args.random_video, args.random_image, args.image,
              args.video, args.save).predict()
