import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import signal
from tensorflow.python.keras.utils import losses_utils


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

        return tf.keras.losses.MeanSquaredError()(self.y_true, self.y_pred)
