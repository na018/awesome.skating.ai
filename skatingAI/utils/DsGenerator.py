import itertools
import os
import random
from pathlib import Path
from typing import NewType, Tuple, Generator

import numpy as np
import tensorflow as tf
from typing_extensions import TypedDict

# declare new type information
Frame = NewType('Frame', np.ndarray)
Video = NewType('Video', np.ndarray)
VideoMask = NewType('VideoMask', np.ndarray)
Mask = NewType('Mask', np.ndarray)


class DsPair(TypedDict):
    frame: Frame
    mask: Mask
    size: int


class DsGenerator(object):

    def __init__(self, resize_shape: Tuple[int, int] = None, single_random_frame=True, rgb=False):
        """dataset generator yielding processed images from the `3DPEOPLE DATASET <https://cv.iri.upc-csic.es/>`

        Args:
            resize_shape: resize image to (x,x) to increase performance and reduce memory
            single_random_frame: choose one random frame [True] or sequential frame from random video
            rgb: weather to include background
        """
        self.rgb = rgb
        self.single_random_frame = single_random_frame
        self.video_path_rgbbs: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/rgbb"
        self.video_path_rgbs: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/rgb"
        self.video_path_masks: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/masks"
        self.video_amount: int = len(next(os.walk(self.video_path_masks))[2])
        self.seen_samples = 0

        self.resize_shape = resize_shape
        if not single_random_frame:
            self.video, self.mask = self._get_random_video_mask_pair()

    def _get_random_video_mask_pair(self) -> Tuple[Video, VideoMask]:
        random_n: int = int(random.randint(0, self.video_amount - 1))
        if self.rgb:
            video: np.ndarray = np.load(f"{self.video_path_rgbs}/{random_n}.npz")['arr_0']
        else:
            video: np.ndarray = np.load(f"{self.video_path_rgbbs}/{random_n}.npz")['arr_0']

        mask: np.ndarray = np.load(f"{self.video_path_masks}/{random_n}.npz")['arr_0']
        mask_shape: np.ndarray = np.array(mask.shape)
        mask: np.ndarray = mask.reshape((*mask_shape, -1))

        return video, mask

    def get_image_amount(self) -> int:
        img_counter = 0
        for i in range(self.video_amount):
            img_counter += np.load(f"{self.video_path_rgbbs}/{i + 1}.npz")['arr_0'].shape[0]

        return img_counter

    def _yield_video_sequence(self) -> Generator[DsPair, DsPair, DsPair]:
        video, mask = self._get_random_video_mask_pair()
        print(video.shape)
        for i in itertools.count(1):
            frame = video[i]
            print(self.video_path_masks)
            frame_n: Frame = tf.convert_to_tensor((frame / 255), tf.float32)
            mask_n: Mask = tf.convert_to_tensor((mask[i]), tf.int32)

            if self.resize_shape:
                frame_n = tf.image.resize(frame_n, size=self.resize_shape)
                mask_n = tf.image.resize(mask_n, size=self.resize_shape)

            self.seen_samples += 1

            yield {'frame': frame_n, 'mask': mask_n, 'size': video.shape[0]}

    def get_next_pair(self, frame_i: int = 0) -> Generator[DsPair, DsPair, DsPair]:
        if not self.single_random_frame:
            for i in itertools.count(1):
                frame = self.video[frame_i]
                frame_n: Frame = tf.convert_to_tensor((frame / 255), tf.float32)
                mask_n: Mask = tf.convert_to_tensor((self.mask[frame_i]), tf.int32)

                if self.resize_shape:
                    frame_n = tf.image.resize(frame_n, size=self.resize_shape)
                    mask_n = tf.image.resize(mask_n, size=self.resize_shape)

                self.seen_samples += 1

                yield {'frame': frame_n, 'mask': mask_n, 'size': self.video.shape[0]}
        else:
            for i in itertools.count(1):
                video, mask = self._get_random_video_mask_pair()

                random_frame_n: int = random.randint(0, video.shape[0] - 1)

                random_frame: Frame = tf.convert_to_tensor((video[random_frame_n] / 255), tf.float32)
                random_mask: Mask = tf.convert_to_tensor((mask[random_frame_n]), tf.int32)

                if self.resize_shape:
                    random_frame = tf.image.resize(random_frame, size=self.resize_shape)
                    random_mask = tf.image.resize(random_mask, size=self.resize_shape)

                self.seen_samples += 1

                yield {'frame': random_frame, 'mask': random_mask, 'size': 1}


def build_iterator(self, img_shape=(480, 640, 3), batch_size: int = 10,
                   prefetch_batch_buffer: int = 5) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(self.get_next_pair,
                                             output_types={'frame': tf.float32, 'mask': tf.int32})

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_batch_buffer)

    return dataset
