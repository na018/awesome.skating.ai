import itertools
import os
import random
from pathlib import Path
from typing import NewType, Tuple, Generator, List

import numpy as np
import tensorflow as tf
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

    def __init__(self, resize_shape_x: int = None, test: bool = False, sequential: bool = False):
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
        self.seen_samples: int = 0
        self.sequential: bool = sequential
        self.test: bool = test

        self.video, self.mask_bg, self.mask_hp, self.kps, self.video_name = self._get_random_video_mask_kp(test)
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

    def get_image_amount(self) -> int:
        img_counter = 0
        for i in range(self.video_amount):
            img_counter += np.load(f"{self.video_path_rgbbs}/{self.file_names[i + 1]}")[
                'arr_0'].shape[0]

        return img_counter

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
                video, mask_bg, mask_hp, kps, video_name = self._get_random_video_mask_kp(self.test)
                _frame_i: int = random.randint(0, video.shape[0] - 1)
                video_i, mask_bg_i, mask_hp_i, kps_i = video[_frame_i], mask_bg[_frame_i], mask_hp[_frame_i], kps[
                    _frame_i]

            frame_n: Frame = tf.convert_to_tensor(
                (video_i / 255), tf.float32)
            mask_bg_n: Mask = tf.convert_to_tensor(
                mask_bg_i, tf.int32)
            mask_hp_n: Mask = tf.convert_to_tensor(
                mask_hp_i, tf.int32)

            if self.resize_shape:
                frame_n = tf.image.resize(
                    frame_n, size=self.resize_shape)
                mask_bg_n = tf.image.resize(
                    mask_bg_n, size=self.resize_shape)
                mask_hp_n = tf.image.resize(
                    mask_hp_n, size=self.resize_shape)
                kps_i = kps_i / self.resize_factor

            kps_n = np.reshape(kps_i, (kps_i.size // 2, 2))
            kps_n[:, 0] /= mask_hp_n.shape[0]
            kps_n[:, 1] /= mask_hp_n.shape[1]
            kps_n = np.reshape(kps_n, (-1))
            kps_n = tf.convert_to_tensor(kps_n)

            self.seen_samples += 1

            yield {'frame': frame_n, 'mask_bg': mask_bg_n, 'mask_hp': mask_hp_n, 'kps': kps_n}

    def build_iterator(self, batch_size: int = 10,
                       prefetch_batch_buffer: int = 5) -> tf.data.Dataset:

        dataset = tf.data.Dataset.from_generator(self.get_next_pair,
                                                 output_types={'frame': tf.float32, 'mask_bg': tf.float32,
                                                               'mask_hp': tf.float32,
                                                               'kps': tf.float64})

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)

        return dataset
