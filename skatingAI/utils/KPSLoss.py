import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import signal
from tensorflow.python.keras.utils import losses_utils

from skatingAI.utils.utils import kps_upscale_reshape


class KPSLoss(tf.keras.losses.Loss):

    def __init__(self, n_classes: int):
        """

        Args:
            n_classes:
        """
        super().__init__(name='KPSLoss')
        self.reduction = losses_utils.loss_reduction.ReductionV2.AUTO
        self.name = "KPSLoss"
        self._allow_sum_over_batch_size = True
        self.n_classes = n_classes
        self.y_true: tf.int32 = None
        self.y_pred: tf.float32 = None
        self.y_true_maps = []

    def call(self, kps: tf.int32, y_pred: tf.float32):
        self.y_true = kps
        self.y_pred = y_pred

        return self._calculate_loss()

    def gkern(self, kernlen=18, std=3):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def _calculate_loss(self) -> tf.float32:
        """ calculate loss

        Returns:
            loss value
        """
        map_shape = self.y_pred.shape[1:3]
        gaussian_size = 6
        gs_h = gaussian_size // 2
        gaussian_kernel = self.gkern(gaussian_size)
        y_true_maps = []

        for i, y_kps in enumerate(self.y_true):
            y_kps = kps_upscale_reshape(map_shape, y_kps)
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
                    kps = y_kps[idx]
                    # prevent errors if x or y is add the edges
                    x, y = np.int(kps[0]) + gs_h, np.int(kps[1]) + gs_h
                    feature_map[y - gs_h:y + gs_h, x - gs_h: x + gs_h] = gaussian_kernel
                feature_maps.append(feature_map)

            feature_maps = np.array(feature_maps)
            feature_maps = np.transpose(feature_maps[:, gs_h: -gs_h, gs_h: -gs_h], (1, 2, 0))
            all_keypoints = np.argmax(feature_maps, axis=-1)
            all_keypoints[all_keypoints > 0] = 1
            feature_map_bg = np.ones(map_shape) - all_keypoints
            feature_maps = np.insert(feature_maps, 0, feature_map_bg, axis=-1)
            y_true_maps.append(feature_maps)

        y_true_maps = np.array(y_true_maps)
        self.y_true_maps = y_true_maps

        result = tf.keras.losses.MeanSquaredError()(y_true_maps, self.y_pred)

        return result
