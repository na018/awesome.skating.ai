import itertools
import os
import random
from pathlib import Path
from typing import NewType, Tuple

import numpy as np
import tensorflow as tf

# declare new type information
Frame = NewType('Frame', np.ndarray)
Video = NewType('Video', np.ndarray)
VideoMask = NewType('VideoMask', np.ndarray)
Mask = NewType('Mask', np.ndarray)


class DsGenerator(object):

    def __init__(self):
        self.video_path_rgbs: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/rgbb"
        self.video_path_masks: str = f"{Path.cwd()}/Data/3dhuman/processed/numpy/masks"
        self.video_amount: int = len(next(os.walk(self.video_path_masks))[2])
        self.seen_samples = 0

    def _get_random_video_mask_pair(self) -> Tuple[Video, VideoMask]:
        random_n: int = int(random.randint(0, self.video_amount - 1))
        video: np.ndarray = np.load(f"{self.video_path_rgbs}/{random_n}.npz")['arr_0']
        mask: np.ndarray = np.load(f"{self.video_path_masks}/{random_n}.npz")['arr_0']
        mask_shape: np.ndarray = np.array(mask.shape)
        mask: np.ndarray = mask.reshape((*mask_shape, -1))

        return video, mask

    def get_image_amount(self) -> int:
        img_couner = 0
        for i in range(self.video_amount):
            img_couner += np.load(f"{self.video_path_rgbs}/{i + 1}.npz")['arr_0'].shape[0]

        return img_couner

    def get_next_pair(self) -> Tuple[Frame, Mask]:
        for i in itertools.count(1):
            video, mask = self._get_random_video_mask_pair()
            random_frame_n: int = random.randint(0, video.shape[0] - 1)

            random_frame: Frame = tf.convert_to_tensor((video[random_frame_n] / 255), tf.float32)
            random_mask: Mask = tf.convert_to_tensor((mask[random_frame_n]), tf.int32)

            self.seen_samples += 1

            yield {'frame': random_frame, 'mask': random_mask}

    def buid_iterator(self, img_shape=(480, 640, 3), batch_size: int = 10,
                      prefetch_batch_buffer: int = 5) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(self.get_next_pair,
                                                 output_types={'frame': tf.float32, 'mask': tf.int32})

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)

        return dataset
