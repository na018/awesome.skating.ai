import itertools
import os
import random
from pathlib import Path
from typing import NewType, Tuple, Generator, List

import cv2
import numpy as np
import tensorflow as tf
from scipy import signal
from typing_extensions import TypedDict

# declare new type information
Frame = NewType('Frame', np.ndarray)
Video = NewType('Video', np.ndarray)
VideoMask = NewType('VideoMask', np.ndarray)
Mask = NewType('Mask', np.ndarray)
KeyPoints = NewType('KeyPoints', np.ndarray)


class DsPair(TypedDict):
    frame: np.ndarray
    mask_bg: np.ndarray
    mask_hp: np.ndarray
    kps: np.ndarray
    size: int


class DsGenerator(object):

    def __init__(self, resize_shape_x: int = None, test: bool = False, sequential: bool = False, batch_size=2,
                 epoch_steps=12):
        """dataset generator yielding processed images from the `3DPEOPLE DATASET <https://cv.iri.upc-csic.es/>`

        Args:
            resize_shape_x: resize image to (x,x) to increase performance and reduce memory
        """
        self.video_path_rgbbs: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/rgbb"
        self.video_path_rgbs: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/rgb"
        self.video_path_masks_hp: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/masks"
        self.video_path_kps: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/skeletons"
        self.file_names: List[str] = next(os.walk(self.video_path_masks_hp))[2]
        self.video_amount: int = len(self.file_names)
        self.test_start = int(self.video_amount * 0.9)
        self.train_end = self.test_start - 1
        self.seen_samples: int = 1
        self.sequential: bool = sequential
        self.test: bool = test
        self.new_video_counter = batch_size * epoch_steps
        self.batch_size = batch_size
        self.batch_counter = 0

        self.video, self.mask_bg, self.mask_hp, self.kps, self.video_name = self._get_random_video_mask_kp(test)
        self.videos, self.mask_bgs, self.mask_hps, self.kpss, self.video_names = [], [], [], [], []
        self.video_size: int = len(self.kps)
        self.frame_i: int = 0

        if resize_shape_x:
            self.resize_factor = self.video.shape[1] // resize_shape_x
            self.resize_shape = (resize_shape_x, self.video.shape[2] // self.resize_factor)

    def _get_random_video_mask_kp(self, test: bool = False) -> Tuple[Video, VideoMask, VideoMask, KeyPoints, str]:
        if test or self.test:
            random_n: int = int(random.randint(self.test_start, self.video_amount - 1))
        else:
            random_n: int = int(random.randint(0, self.train_end - 1))

        video: np.ndarray = np.load(
            f"{self.video_path_rgbs}/{self.file_names[random_n]}")['arr_0']

        rgbb: np.ndarray = np.load(
            f"{self.video_path_rgbbs}/{self.file_names[random_n]}")['arr_0']
        rgbb = np.sum(rgbb, axis=-1)
        mask_bg = np.zeros(rgbb.shape)
        mask_bg[rgbb > 0] = 1

        mask_hp: np.ndarray = np.load(
            f"{self.video_path_masks_hp}/{self.file_names[random_n]}")['arr_0']

        mask_shape: np.ndarray = np.array(mask_hp.shape)
        mask_bg: np.ndarray = mask_bg.reshape((*mask_shape, -1))
        mask_hp: np.ndarray = mask_hp.reshape((*mask_shape, -1))
        kps: np.ndarray = np.load(f"{self.video_path_kps}/{self.file_names[random_n]}")['arr_0']

        return video, mask_bg, mask_hp, kps, self.file_names[random_n].split('.')[0]

    def gkern(self, kernlen=18, std=3):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def get_kps_maps(self, kps, mask_shape, gaussian_size=6):
        map_shape = mask_shape[:2]

        gs_h = gaussian_size // 2
        gaussian_kernel = self.gkern(gaussian_size)

        feature_maps = []
        ii = 0
        #  [0, 1, 5, 9, 10, 11, 12, 33, 34, 35, 36, 57, 58, 59, 61, 62, 63, 64, 66]
        # joints = ["Hips", "Spine", "Head", "LeftShoulder","LeftArm","LeftForeArm","LeftHand","RightShoulder","RightArm","RightForeArm","RightHand",
        #           "RightUpLeg","RightLeg", "RightFoot","RightToe_End", "LeftUpLeg","LeftLeg", "LeftFoot","LeftToe_End"]
        joints = [[0], [1], [2], [3, 7], [4, 8], [5, 9], [6, 10], [11, 15],
                  [12, 16], [13, 17], [14, 18]]
        for joint in joints:
            feature_map = np.zeros((map_shape[0] + gaussian_size, map_shape[1] + gaussian_size))
            for idx in joint:
                kps_joint = kps[idx]
                # prevent errors if x or y is on the edges
                x, y = np.int(kps_joint[0]) + gs_h, np.int(kps_joint[1]) + gs_h
                feature_map[y - gs_h:y + gs_h, x - gs_h: x + gs_h] = gaussian_kernel
            feature_maps.append(feature_map)

        feature_maps = np.array(feature_maps)
        feature_maps = np.transpose(feature_maps[:, gs_h: -gs_h, gs_h: -gs_h], (1, 2, 0))

        left_cut = kps[4, 0].astype(int) - np.random.randint(0, 20, 1)[0] - 10
        left_cut = left_cut if left_cut >= 0 else 0
        bottom_cut = kps[1, 1].astype(int) + np.random.randint(0, 70, 1)[0]
        bottom_cut = bottom_cut if bottom_cut <= map_shape[0] else map_shape[0]

        random_feature_maps = np.zeros(feature_maps.shape)
        random_feature_maps[:bottom_cut, left_cut:] = feature_maps[:bottom_cut, left_cut:, ]
        all_keypoints = np.argmax(random_feature_maps, axis=-1)
        all_keypoints[all_keypoints > 0] = 1
        feature_map_bg = np.ones(all_keypoints.shape) - all_keypoints
        random_feature_maps = np.insert(random_feature_maps, 0, feature_map_bg, axis=-1)

        # img[0:fm2.shape[0],0:fm2.shape[1]]=fm2

        return random_feature_maps, left_cut, bottom_cut

    def get_image_amount(self) -> int:
        img_counter = 0
        for i in range(self.video_amount):
            img_counter += np.load(f"{self.video_path_rgbbs}/{self.file_names[i + 1]}")[
                'arr_0'].shape[0]

        return img_counter

    def _add_occlusion(self, img, random_joint):
        random_joint_y, random_joint_x = random_joint.astype(int)
        random_color_1 = random.randint(0, 255)
        random_color_2 = random.randint(0, 255)
        img[random_joint_y:random_joint_y + 5, random_joint_x:] = (random_color_1, 255, random_color_2)
        img[0:random_joint_y, random_joint_x:random_joint_x + 5] = (random_color_1, random_color_2, random_color_1)

    def set_new_video(self, test: bool = False):
        self.video, self.mask_hp, self.mask_bg, self.kps, self.video_name = self._get_random_video_mask_kp(test)

    def get_next_pair(self) -> Generator:
        for _ in itertools.count(1):
            if self.sequential:
                _frame_i: int = self.frame_i
                video_i, mask_bg_i, mask_hp_i, kps_i, video_name = self.video[_frame_i], self.mask_bg[_frame_i], \
                                                                   self.mask_hp[_frame_i], self.kps[
                                                                       _frame_i], self.video_name
                if self.frame_i + 1 < self.video_size:
                    self.frame_i += 1
            else:
                if self.seen_samples % self.new_video_counter == 0 or self.seen_samples == 1:
                    self.videos, self.mask_bgs, self.mask_hps, self.kpss, self.video_names = [], [], [], [], []
                    for _ in range(self.batch_size):
                        video, mask_bg, mask_hp, kps, video_name = self._get_random_video_mask_kp(
                            self.test)
                        self.videos.append(video)
                        self.mask_bgs.append(mask_bg)
                        self.mask_hps.append(mask_hp)
                        self.kpss.append(kps)
                        self.video_names.append(video_name)
                        print(f'{"-" * 20}-> train random frames from [{video_name}]')

                video, mask_bg, mask_hp, kps, video_name = self.videos[self.batch_counter], self.mask_bgs[
                    self.batch_counter], self.mask_hps[self.batch_counter], self.kpss[self.batch_counter], \
                                                           self.video_names[self.batch_counter]

                _frame_i: int = random.randint(0, video.shape[0] - 1)

                video_i, mask_bg_i, mask_hp_i, kps_i = video[_frame_i], mask_bg[_frame_i], mask_hp[_frame_i], kps[
                    _frame_i]

            frame_n = np.array(cv2.cvtColor(video_i, cv2.COLOR_BGR2RGB))
            mask_bg_n: Mask = mask_bg_i
            mask_hp_n: Mask = mask_hp_i

            if self.resize_shape:
                frame_n = cv2.resize(
                    frame_n, (self.resize_shape[1], self.resize_shape[0]))
                mask_bg_n = tf.image.resize(
                    mask_bg_n, size=self.resize_shape)
                mask_hp_n = tf.image.resize(
                    mask_hp_n, size=self.resize_shape)
                kps_i = kps_i / self.resize_factor

            kps_n = np.reshape(kps_i, (kps_i.shape[0] // 2, 2))
            kps_maps, left_cut, bottom_cut = self.get_kps_maps(kps_n, mask_hp_n.shape)

            random_frame_n = np.random.randint(0, 255, frame_n.shape)
            random_frame_n[:bottom_cut, left_cut:] = frame_n[:bottom_cut, left_cut:, ]
            self._add_occlusion(random_frame_n, kps_n[np.random.randint(0, kps_n.shape[0] - 1)])

            random_kernel = np.random.randint(1, 15, size=1)
            random_frame_n = cv2.blur(random_frame_n, (random_kernel, random_kernel)) / 255

            random_mask_bg_n = np.zeros(mask_bg_n.shape)
            random_mask_bg_n[:bottom_cut, left_cut:] = mask_bg_n[:bottom_cut, left_cut:, ]

            random_mask_hp_n = np.zeros(mask_hp_n.shape)
            random_mask_hp_n[:bottom_cut, left_cut:] = mask_hp_n[:bottom_cut, left_cut:, ]

            self.seen_samples += 1
            self.batch_counter += 1

            if self.batch_counter == self.batch_size:
                self.batch_counter = 0

            yield {'frame': tf.constant(random_frame_n), 'mask_bg': tf.constant(random_mask_bg_n),
                   'mask_hp': tf.constant(random_mask_hp_n), 'kps': tf.constant(kps_maps)}

    def build_iterator(self, batch_size: int = 10,
                       prefetch_batch_buffer: int = 5) -> tf.data.Dataset:

        dataset = tf.data.Dataset.from_generator(self.get_next_pair,
                                                 output_types={'frame': tf.float32, 'mask_bg': tf.float32,
                                                               'mask_hp': tf.float32,
                                                               'kps': tf.float64})

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)

        return dataset
