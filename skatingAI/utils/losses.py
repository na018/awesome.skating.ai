import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils

from skatingAI.utils.human_distance_map import HumanDistanceMap


class CILoss(tf.keras.losses.Loss):
    """calculates class imbalance loss
    """

    def __init__(self, n_classes: int):
        """

        Args:
            n_classes:
        """
        super().__init__(name='ClassImbalanceLoss')
        self.reduction = losses_utils.loss_reduction.ReductionV2.AUTO
        self.name = "ClassImbalanceLoss"
        self._allow_sum_over_batch_size = True
        self.n_classes = n_classes
        self.y_true: tf.int32 = None
        self.y_pred: tf.float32 = None
        self.weighted_map = HumanDistanceMap().weighted_distances
        self.correct_predictions = 0
        self.correct_body_part_pred = 0
        self.body_part_px_n_pred = 0
        self.body_part_px_n_true = 0
        self.body_part_FNR = 0
        self.multiplicator = 1

    def call(self, y_true: tf.int32, y_pred: tf.float32):
        self.y_true = y_true
        self.y_pred = y_pred

        return self._calculate_class_imbalance_loss()

    def _calculate_class_imbalance_loss(self) -> tf.float32:
        """ calculate loss

        distance between the :prediction and :ground_truth with respect
        to the distance matrix :M on the label space M.

        Returns:
            loss value
        """

        y_true = np.reshape(self.y_true, self.y_true.shape[:-1])
        y_pred = tf.argmax(self.y_pred[0], axis=-1, output_type=tf.int32)
        correct_predictions = tf.cast(tf.math.equal(y_pred, y_true), dtype=tf.int8)
        self.correct_predictions = np.sum(correct_predictions)

        body_part_px_pred = tf.cast(tf.math.greater(y_pred, np.zeros(y_pred.shape)), dtype=tf.int8)
        self.body_part_px_n_pred = np.sum(body_part_px_pred)
        self.correct_body_part_pred = np.sum(tf.multiply(body_part_px_pred, correct_predictions))

        body_part_px_true = tf.cast(tf.math.greater(y_true, np.zeros(y_true.shape)), dtype=tf.int8)
        self.body_part_px_n_true = np.sum(body_part_px_true)
        self.body_part_FNR = (self.body_part_px_n_pred - self.correct_body_part_pred) / self.body_part_px_n_true

        y_true = tf.one_hot(y_true.astype(np.int32), self.n_classes, axis=-1)
        y_pred = self.y_pred

        delta = []

        for i, true_img in enumerate(y_true):
            delta.append(
                tf.multiply(tf.abs(tf.subtract(y_pred[i], true_img)),
                            self.weighted_map[tf.argmax(true_img, axis=-1)]) * self.multiplicator + 1e-6)

        return tf.add_n([
            tf.abs(
                tf.subtract(y_true, y_pred)
            ), delta])

# source https://github.com/imatge-upc/segmentation_DLMI/
