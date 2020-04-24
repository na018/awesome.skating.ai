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
        super(tf.keras.losses.Loss, self)
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
        gaussian_size = 5
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
                feature_map = np.zeros(map_shape)
                for idx in joint:
                    kps = y_kps[idx]
                    x, y = np.int(kps[0]), np.int(kps[1])
                    feature_map[y - gaussian_size // 2:y + gaussian_size // 2 + 1,
                    x - gaussian_size // 2: x + gaussian_size // 2 + 1] = gaussian_kernel
                feature_maps.append(feature_map)

            y_true_maps.append(np.transpose(feature_maps, (1, 2, 0)))

        y_true_maps = np.array(y_true_maps)
        self.y_true_maps = y_true_maps

        result = tf.keras.losses.MeanSquaredError()(y_true_maps, self.y_pred)

        return result

# source https://github.com/imatge-upc/segmentation_DLMI/
# create 3d cube
# import math
# import matplotlib.pyplot as plot
# import mpl_toolkits.mplot3d.axes3d as axes3d
#
#
# def cube_marginals(cube, normalize=False):
#     c_fcn = np.mean if normalize else np.sum
#     xy = c_fcn(cube, axis=0)
#     xz = c_fcn(cube, axis=1)
#     yz = c_fcn(cube, axis=2)
#     return (xy, xz, yz)
#
#
# def plot_cube(cube, x=None, y=None, z=None, normalize=False, plot_front=False):
#     """Use contourf to plot cube marginals"""
#     (Z, Y, X) = cube.shape
#     (xy, xz, yz) = cube_marginals(cube, normalize=normalize)
#     if x == None: x = np.arange(X)
#     if y == None: y = np.arange(Y)
#     if z == None: z = np.arange(Z)
#
#     fig = plot.figure()
#     ax = fig.gca(projection='3d')
#
#     # draw edge marginal surfaces
#     offsets = (Z - 1, 0, X - 1) if plot_front else (0, Y - 1, 0)
#     cset = ax.contourf(x[None, :].repeat(Y, axis=0), y[:, None].repeat(X, axis=1), xy, zdir='z', offset=offsets[0],
#                        cmap=plot.cm.coolwarm, alpha=0.75)
#     cset = ax.contourf(x[None, :].repeat(Z, axis=0), xz, z[:, None].repeat(X, axis=1), zdir='y', offset=offsets[1],
#                        cmap=plot.cm.coolwarm, alpha=0.75)
#     cset = ax.contourf(yz, y[None, :].repeat(Z, axis=0), z[:, None].repeat(Y, axis=1), zdir='x', offset=offsets[2],
#                        cmap=plot.cm.coolwarm, alpha=0.75)
#
#     # draw wire cube to aid visualization
#     ax.plot([0, X - 1, X - 1, 0, 0], [0, 0, Y - 1, Y - 1, 0], [0, 0, 0, 0, 0], 'k-')
#     ax.plot([0, X - 1, X - 1, 0, 0], [0, 0, Y - 1, Y - 1, 0], [Z - 1, Z - 1, Z - 1, Z - 1, Z - 1], 'k-')
#     ax.plot([0, 0], [0, 0], [0, Z - 1], 'k-')
#     ax.plot([X - 1, X - 1], [0, 0], [0, Z - 1], 'k-')
#     ax.plot([X - 1, X - 1], [Y - 1, Y - 1], [0, Z - 1], 'k-')
#     ax.plot([0, 0], [Y - 1, Y - 1], [0, Z - 1], 'k-')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plot.show()
#
#
# plot_cube(np.array(feature_maps))
