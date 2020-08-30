import argparse
import os
from collections import namedtuple
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from skatingAI.nets.bg.v0 import BGNet
from skatingAI.nets.hrnet.v7 import HPNet
from skatingAI.nets.keypoint.v3 import KPDetector
from skatingAI.utils.DsGenerator import DsGenerator, Frame, Mask, DsPair
from skatingAI.utils.utils import create_mask, mask2rgb, create_dir, Logger, kps2frame


class Predictor(object):
    def __init__(self, bg_weight_counter: int, hp_weight_counter: int, kps_weight_counter: int, name: str,
                 random_video: bool, random_image: bool, img_path: str, video_path: str, video_sequence_path,
                 save_path: str):
        """predicts images or videos from specified network and weight

        """

        self.bg_weight_counter = bg_weight_counter
        self.hp_weight_counter = hp_weight_counter
        self.kps_weight_counter = kps_weight_counter

        if random_video:
            sequential = True
            self.generator, self.iter, self.sample_frame, \
            self.sample_mask, self.sample_kps = self._generate_dataset(sequential=sequential)
            self.name = self.generator.video_name
        elif video_path:
            self.name = video_path.split('.')[-2].split('/')[-1]
            self.video_frames = self.parse_video(video_path)
        else:
            self.name = video_sequence_path.split('.')[-1].split('/')[-1]
            self.video_frames = self.parse_video_sequence(video_sequence_path)

        self.random_video = random_video
        self.random_image = random_image
        self.img_path = img_path
        self.video_path = video_path
        self.save = save_path

        self.bg_weight_path = f"{os.getcwd()}/ckpt1/bg-{bg_weight_counter}.ckpt"
        self.hp_weight_path = f"{os.getcwd()}/ckpt1/hp-{hp_weight_counter}.ckpt"
        self.kps_weight_path = f"{os.getcwd()}/ckpt1/kps-{kps_weight_counter}.ckpt"
        self.dir_save = f"{os.getcwd()}/predictions/{self.name}_{kps_weight_counter}"
        self.Logger = Logger()
        self.file_name = self._prepare_file_saving()

        self.bg_extractor = self._get_bg_model()
        self.hp_model = self._get_hrnet_model()
        self.kps_model = self._get_kps_model()

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

    def _create_video_kp(self, frames, frames_extracted_bg, body_part_predictions, kps_predictions):
        offset = 10
        width_total = frames[0].shape[1]
        height_total = frames[0].shape[0]
        new_img = Image.new('RGB', (width_total, height_total))
        out = cv2.VideoWriter(f"{self.dir_save}/{self.name}-kp.avi", cv2.VideoWriter_fourcc(*'DIVX'), 12,
                              new_img.size)

        for i, frame in enumerate(frames):
            # kps =  tf.keras.preprocessing.image.array_to_img(kps2frame(create_mask(kps_predictions[i]), frame))

            frame_kps = kps2frame(create_mask(kps_predictions[i]),
                                  np.array(tf.keras.preprocessing.image.array_to_img(frame)))

            fnt = ImageFont.truetype('/usr/share/fonts/truetype/open-sans/OpenSans-Semibold.ttf', 15)
            f_prep = tf.keras.preprocessing.image.array_to_img(frame)
            new_img.paste(tf.keras.preprocessing.image.array_to_img(frame_kps), (0, 0))
            d = ImageDraw.Draw(new_img)
            d.text((100, 5), "Predictect Keypoints", font=fnt,
                   fill=(255, 255, 255))

            # if i == 0:
            #     out = cv2.VideoWriter(f"{self.dir_save}/{self.file_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 12,
            #                           new_img.size)
            out.write(np.array(new_img))
            self.Logger.log(f"Add frame {i} to video writer")
        out.release()
        self.Logger.log(f'successfully created video: {self.name}')

    def _create_video(self, frames, frames_extracted_bg, body_part_predictions, kps_predictions):
        offset = 10
        width_total = body_part_predictions[0].shape[1] * 3 + offset * 4
        height_total = (body_part_predictions[1].shape[0] + 100) * 2
        new_img = Image.new('RGB', (width_total, height_total))
        out = cv2.VideoWriter(f"{self.dir_save}/{self.name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 12,
                              new_img.size)

        for i, bp_prediction in enumerate(body_part_predictions):
            #frames_extracted_bg[i][frames_extracted_bg[i] == 2] = 0
            images = [

                tf.keras.preprocessing.image.array_to_img(frames_extracted_bg[i]),
                tf.keras.preprocessing.image.array_to_img(mask2rgb(create_mask(bp_prediction))),
                tf.keras.preprocessing.image.array_to_img(mask2rgb(create_mask(kps_predictions[i])))
            ]

            title = ['Extracted Background', 'Predicted Body Parts', f'Predicted Keypoints']
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/open-sans/OpenSans-Semibold.ttf', 15)
            new_img.paste(tf.keras.preprocessing.image.array_to_img(frames[i]), (bp_prediction.shape[1], 50))
            d = ImageDraw.Draw(new_img)
            d.text((bp_prediction.shape[1] + bp_prediction.shape[1] / 2.4, 5), "input frame", font=fnt,
                   fill=(255, 255, 255))
            x_offset = 0
            for j, img in enumerate(images):
                new_img.paste(img, (x_offset, height_total // 2))
                d = ImageDraw.Draw(new_img)
                d.text((x_offset + bp_prediction.shape[1] / 2.4, height_total // 2), title[j], font=fnt,
                       fill=(255, 255, 255))
                x_offset += bp_prediction.shape[1] + offset

            # if i == 0:
            #     out = cv2.VideoWriter(f"{self.dir_save}/{self.file_name}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 12,
            #                           new_img.size)
            out.write(np.array(new_img))
            self.Logger.log(f"Add frame {i} to video writer")
        out.release()
        self.Logger.log(f'successfully created video: {self.name}')

    def _predict_video(self, frames):
        f = np.array(frames)

        pred_bg, pred_hp, pred_kp = [], [], []
        for i in range(0, len(frames), 80):
            fi = f[i:i + 80:]
            self.Logger.log(f'Start prediction fun:bg_extractor {i}', block=True)

            extracted_bg = self.bg_extractor.predict([fi])
            imgs = np.argmax(extracted_bg, axis=-1)
            frames_extracted_bg = np.array(fi)
            frames_extracted_bg[imgs == 0] = 0
            self.Logger.log('Start prediction fun:body_parts', block=True)
            body_parts = self.hp_model.predict([fi])
            self.Logger.log('Start prediction fun:keypoints', block=True)
            kps = self.kps_model.predict([fi])

            if len(pred_hp) > 0:
                pred_bg = np.concatenate((pred_bg, frames_extracted_bg))
                pred_hp = np.concatenate([pred_hp, body_parts])
                pred_kp = np.concatenate([pred_kp, kps])
            else:
                pred_bg, pred_hp, pred_kp = frames_extracted_bg, body_parts, kps

        self.Logger.log('Prediction finished.\n start to generate video', block=True)
        self._create_video_kp(frames, pred_bg,
                              pred_hp,
                              pred_kp)
        self._create_video(frames, pred_bg,
                           pred_hp,
                           pred_kp)

    def parse_video(self, video_path):
        video_handle = cv2.VideoCapture(video_path)
        width = int(video_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            "width:", width,
            "height:", height,
            "amount of frames:", video_handle.get(cv2.CAP_PROP_FRAME_COUNT),
            "fps:", video_handle.get(cv2.CAP_PROP_FPS),
        )

        # all_frames = video_handle.get(cv2.CAP_PROP_FRAME_COUNT)

        frames = []

        i = 0
        eof = True
        while eof:
            eof, frame = video_handle.read()
            if eof:
                frame = cv2.resize(frame, (320, 240))
                frame = frame / 255
                frames.append(frame)

            i += 1

        self.IMG_SHAPE = frames[0].shape
        self.N_CLASS = 9
        self.KPS_COUNT = 11

        print('read in all frames')

        return np.array(frames)

    def parse_video_sequence(self, video_sequence_path):
        # rgb_dir = '/home/nadin-katrin/Videos/biellmann_sequence'
        rgb_dir = video_sequence_path
        file_names_rgb = [f.path for f in os.scandir(rgb_dir) if f.is_file()]

        frames = []

        for frame in sorted(file_names_rgb):
            f = cv2.resize(cv2.imread(frame), (320, 240))
            f = f / 255

            frames.append(f)

        self.IMG_SHAPE = f.shape
        self.N_CLASS = 9
        self.KPS_COUNT = 11

        print('read in all frames')

        return frames

    def _generate_dataset(self, test: bool = False, sequential: bool = False):
        generator = DsGenerator(resize_shape_x=240, test=test, sequential=sequential)

        sample_pair: DsPair = next(generator.get_next_pair())

        self.IMG_SHAPE = sample_pair['frame'].shape
        self.N_CLASS = np.max(sample_pair['mask_hp']).astype(int) + 1

        self.KPS_COUNT = len(sample_pair['kps'])

        ds = generator.build_iterator(1, 0)

        return generator, ds.as_numpy_iterator(), \
               sample_pair['frame'], sample_pair['mask_hp'], sample_pair['kps']

    def _get_bg_model(self) -> tf.keras.Model:
        bgnet = BGNet(self.IMG_SHAPE, 2)
        model = bgnet.model

        model.load_weights(self.bg_weight_path)
        model.trainable = False

        return model

    def _get_hrnet_model(self) -> tf.keras.Model:
        hrnet = HPNet(self.IMG_SHAPE, bgnet_input=self.bg_extractor, output_channels=9)
        model = hrnet.model

        model.load_weights(self.hp_weight_path)
        model.trainable = False

        return model

    def _get_kps_model(self) -> tf.keras.Model:
        kp_detector = KPDetector(input_shape=self.IMG_SHAPE, hrnet_input=self.hp_model, bgnet_input=self.bg_extractor,
                                 output_channels=12)
        model = kp_detector.model

        model.load_weights(self.kps_weight_path)
        model.trainable = False

        return model

    def predict(self, prediction_img_amount: int = 2):
        """start the prediction process.

        Args:
            prediction_img_amount: Amount of images which will be predicted.
        """
        # set_gpus(self.gpu)

        if self.random_image:
            self._predict_img(prediction_img_amount)
        elif self.random_video:
            frames = []
            for i in range(self.generator.video_size):
                sample_pair = next(self.iter)
                frames.append(sample_pair['frame'][0])
                self.Logger.log(f"Got frame [{i}]")

            self._predict_video(frames)

        elif len(self.img_path) > 1:
            # todo: load image
            pass
        elif len(self.video_path) > 1:
            # todo: load video and extract frames
            self._predict_video(self.video_frames)


if __name__ == "__main__":
    ArgsNamespace = namedtuple('ArgNamespace',
                               ['gpu', 'name', 'wcounter_bg', 'wcounter_hp', 'wcounter_kps',
                                'v', 'video', 'video_sequence', 'image', 'random_video', 'random_image',
                                'save'])

    # image sequence: /home/nadin-katrin/Videos/biellmann_sequence
    parser = argparse.ArgumentParser(
        description='Predict body parts from images or videos')
    parser.add_argument('--wcounter_bg', default=15200, help='Number of weight')
    parser.add_argument('--wcounter_hp', default=15200, help='Number of weight')
    parser.add_argument('--wcounter_kps', default=15200, help='Number of weight')
    parser.add_argument('--name', default='', help='unique name to save to save video/image')
    parser.add_argument('--video', default='/home/nadin-katrin/Documents/porsche_production4.mp4',
                        help='absolute path to video file', type=str)
    parser.add_argument('--video_sequence', default='/home/nadin-katrin/Videos/biellmann_sequence',
                        help='absolute path to video file', type=str)
    parser.add_argument('--image', default='/', help='absolute path to image file')
    parser.add_argument('--random_video', default=False, help='Random video')
    parser.add_argument('--random_image', default=False, help='Random image', type=bool)
    parser.add_argument('--save', default='', help='Path to save prediction in')
    args: ArgsNamespace = parser.parse_args()

    Predictor(args.wcounter_bg, args.wcounter_hp, args.wcounter_kps,
              args.name,
              args.random_video, args.random_image,
              args.image,
              args.video, args.video_sequence,
              args.save).predict()
