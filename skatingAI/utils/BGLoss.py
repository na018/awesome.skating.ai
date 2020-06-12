import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils


class BGLoss(tf.keras.losses.Loss):

    def __init__(self, n_classes: int):
        """

        Args:
            n_classes:
        """
        super().__init__(name='BGLoss')
        self.reduction = losses_utils.loss_reduction.ReductionV2.AUTO
        self.name = "BGLoss"
        self._allow_sum_over_batch_size = True
        self.n_classes = n_classes
        self.y_true: tf.int32 = None
        self.y_pred: tf.float32 = None
        self.y_true_maps = []

    def call(self, mask: tf.int32, y_pred: tf.float32):
        self.y_true = mask
        self.y_pred = y_pred

        return self._calculate_loss()

    def _calculate_loss(self) -> tf.float32:
        """ calculate loss

        Returns:
            loss value
        """
        y_true = tf.one_hot(self.y_true.astype(np.int32), 2, axis=-1)
        y_true = tf.reshape(y_true, (*y_true.shape[:3], -1))
        self.y_true_maps = y_true

        result = tf.keras.losses.MeanSquaredError()(y_true, self.y_pred)

        return result
