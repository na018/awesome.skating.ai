import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import losses_utils

from skatingAI.utils.human_distance_map import HumanDistanceMap




class GeneralisedWassersteinDiceLoss(tf.keras.losses.Loss):
    """calculates the Generalised Wasserstein Dice Loss

    defined in Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks. MICCAI 2017 (BrainLes)
    """

    def __init__(self, n_classes: int):
        """

        Args:
            n_classes:
        """
        super(tf.keras.losses.Loss, self)
        self.reduction = losses_utils.loss_reduction.ReductionV2.AUTO
        self.name = "GeneralisedWassersteinDiceLoss"
        self.n_classes = n_classes
        self.y_true: tf.int32 = None
        self.y_pred: tf.float32 = None
        self.weighed_map = HumanDistanceMap().weighted_distances
        self.correct_predictions = 0
        self.correct_body_part_pred = 0
        self.body_part_px_n = 0

    def call(self, y_true: tf.int32, y_pred: tf.float32):
        self.y_true = y_true
        self.y_pred = y_pred

        return self._generalised_wasserstein_dice_loss()

    def _generalised_wasserstein_dice_loss(self) -> tf.float32:
        """ calculate the pixel-wise Wasserstein distance

        distance between the :prediction and :ground_truth with respect
        to the distance matrix :M on the label space M.

        Returns:
            loss value
        """

        y_true = np.reshape(self.y_true, self.y_true.shape[:-1])
        y_pred = tf.argmax(self.y_pred[0], axis=-1, output_type=tf.int32)
        correct_predictions = tf.cast(tf.math.equal(y_pred, y_true[0]), dtype=tf.int8)
        self.correct_predictions = np.sum(correct_predictions)

        body_part_px = tf.cast(tf.math.greater(y_pred, np.zeros(y_pred.shape)), dtype=tf.int8)
        self.body_part_px_n = np.sum(body_part_px)
        self.correct_body_part_pred = np.sum(tf.multiply(body_part_px, correct_predictions))

        y_true = tf.one_hot(y_true.astype(np.int32), self.n_classes, axis=-1)
        y_pred = ops.convert_to_tensor(self.y_pred)[0]


        sum = []

        for i, row in enumerate(y_true[0]):
            sum.append(
                tf.multiply(tf.abs(tf.subtract(row, y_pred[i])), self.weighed_map[tf.argmax(row, axis=-1)]))

        return tf.add_n(sum)

# source https://github.com/imatge-upc/segmentation_DLMI/
