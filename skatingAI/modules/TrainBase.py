from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.python.ops.summary_ops_v2 import SummaryWriter

from skatingAI.utils.hyper_paramater import HyperParameterParams
from skatingAI.utils.utils import DisplayCallback


# from tensorflow_core.python.ops.summary_ops_v2 import SummaryWriter


class TrainBase(object):

    def __init__(self, name: str, img_shape, optimizer_name: str, lr_start: float,
                 loss_fct: tf.keras.losses, params,
                 description, train: bool, w_counter, gpu: int, epochs: int):

        self.img_shape = img_shape
        self.name = name
        self.optimizer_name = optimizer_name
        self.lr_start = lr_start
        self.loss_fct = loss_fct
        self.params = params
        self.description = description
        self._train = train
        self.w_counter = w_counter
        self.gpu = gpu
        self.epochs = epochs

    def _get_model(self, NN) -> tf.keras.Model:
        raise NotImplementedError

    def track_logs(self, sample_image, sample_mask, epoch, sample_kp=None):
        """

        Args:
            sample_image:
            sample_mask:
            epoch:
            loss:
        """
        raise NotImplementedError

    def track_metrics_on_train_start(self, train_HP, train_KP, time, epoch_steps, batch_size):
        """

        Args:
            sample_image:
            sample_mask:
            epoch:
            loss:
            metrics:
        """
        raise NotImplementedError

    def test_model(self, epoch: int, epoch_steps: int, iter_test):
        """

        Args:
            epoch:
            epoch_steps:
            iter_test:
        """
        raise NotImplementedError

    def train_model(self, iter):
        raise NotImplementedError

    def _create_display_cb(self, model: tf.keras.Model, sub_dir: str) -> (DisplayCallback, SummaryWriter):
        """

        Args:
            model:

        Returns:

        """
        log_dir = f"{Path.cwd()}/logs/metrics/{self.name}/{sub_dir}/{datetime.now().strftime('%Y_%m_%d__%H_%M')}"
        file_writer_test = tf.summary.create_file_writer(
            f"{log_dir}/scalars/test")

        progress_tracker = DisplayCallback(model, self.epochs,
                                           self.gpu, log_dir=log_dir, sub_dir=sub_dir)
        progress_tracker.set_model(model)

        return progress_tracker, file_writer_test

    def _get_optimizer(self, optimizer_name, lr_start, params: HyperParameterParams, decay_rate_counter=None,
                       optimizer_decay=None):
        """

        Args:
            optimizer_name:
            lr_start:
            params:
            decay_rate_counter:
            optimizer_decay:

        Returns:

        """
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_start, epsilon=params.epsilon,
                                                 amsgrad=params.amsgrad)  # 0.001
        elif optimizer_name == "nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr_start,
                                                  beta_1=params.beta_1, beta_2=params.beta_2,
                                                  epsilon=params.epsilon)
        elif optimizer_name == "sgd_clr":
            decay_rate = params.sgd_clr_decay_rate
            optimizer_decay = decay_rate[decay_rate_counter]
            if decay_rate_counter + 1 < len(decay_rate):
                decay_rate_counter += 1

            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_start, momentum=0.9, decay=optimizer_decay,
                                                nesterov=True)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_start, momentum=0.9, decay=optimizer_decay,
                                                nesterov=True)

        return optimizer, decay_rate_counter

    def _calculate_lr(self, epoch, epoch_steps, metric, optimizer_name,
                      optimizer_decay, decay_rate_counter, params,
                      w_counter, lr_start, step_custom_lr) -> [any, float]:
        """

        Args:
            epoch:
            epoch_steps:
            metric:
            optimizer_name:
            optimizer_decay:
            decay_rate_counter:
            params:
            w_counter:
            lr_start:
            step_custom_lr:

        Returns:

        """
        _optimizer = None
        _decay_rate_counter = decay_rate_counter

        if optimizer_name == 'sgd':
            lr = lr_start * (
                    1. / (1. + optimizer_decay * (epoch - w_counter) * epoch_steps))
        elif optimizer_name == 'sgd_clr':
            lr = lr_start * 1 / (1 + optimizer_decay * step_custom_lr)

            if metric.is_curve_steep() == False:
                _optimizer, _decay_rate_counter = self._get_optimizer(optimizer_name, lr_start, params,
                                                                      decay_rate_counter, optimizer_decay)
                step_custom_lr = 0
        else:
            lr = lr_start
        return _optimizer, lr, step_custom_lr, _decay_rate_counter
