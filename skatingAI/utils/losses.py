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
        self.reduction = losses_utils.loss_reduction.ReductionV2.NONE
        self.name = "GeneralisedWassersteinDiceLoss"
        self.n_classes = n_classes
        self.y_true: tf.int32 = None
        self.y_pred: tf.float32 = None
        self.weighed_map = HumanDistanceMap().weighted_distances

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
        y_true = tf.one_hot(y_true.astype(np.int32), self.n_classes, axis=-1)
        y_pred = ops.convert_to_tensor(self.y_pred)


        sum = []

        for i, row in enumerate(y_true[0]):
            sum.append(
                tf.multiply(tf.abs(tf.subtract(row, y_pred[0, i])), self.weighed_map[tf.argmax(row, axis=-1)]))

        return tf.add_n(sum)

# source https://github.com/imatge-upc/segmentation_DLMI/
