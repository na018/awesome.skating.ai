import argparse
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow_core.python.ops.summary_ops_v2 import ResourceSummaryWriter

from skatingAI.nets.bg import BGNetBase
from skatingAI.nets.hrnet.HRNetBase import HRNetBase
from skatingAI.nets.keypoint import KPDetectorBase
from skatingAI.utils.DsGenerator import DsGenerator, DsPair
from skatingAI.utils.hyper_paramater import HyperParameterParams
from skatingAI.utils.losses import CILoss
from skatingAI.utils.train_program_menu import TrainProgram
from skatingAI.utils.utils import DisplayCallback, set_gpus, Metric, Logger


class MainLoop(object):
    def __init__(self, GPU: int, NAME: str,
                 MODEL_BG: type(BGNetBase), MODEL_HP: type(HRNetBase), MODEL_KPS: type(KPDetectorBase),
                 OPTIMIZER_NAME_HP: str, LR_START_HP: float,
                 OPTIMIZER_NAME_BG: str, LR_START_BG: float,
                 OPTIMIZER_NAME_KPS: str, LR_START_KPS: float,
                 LOSS_FCT_BG: tf.keras.losses, LOSS_FCT_HP: tf.keras.losses, LOSS_FCT_KPS: tf.keras.losses,
                 PARAMS_BG: HyperParameterParams, PARAMS_HP: HyperParameterParams, PARAMS_KPS: HyperParameterParams,
                 DESCRIPTION_BG: str, DESCRIPTION_HP: str, DESCRIPTION_KPS: str,
                 TRAIN_BG: bool = False, TRAIN_HP: bool = False, TRAIN_KPS: bool = False,
                 W_COUNTER_BG: int = -1, W_COUNTER_HP: int = -1, W_COUNTER_KPS: int = -1, EPOCH_START=-1,
                 BG: bool = False,
                 BATCH_SIZE=3,
                 PREFETCH_BATCH_BUFFER=1, EPOCH_STEPS=64, EPOCHS=5555,
                 EPOCH_LOG_N=5):
        """this is the class handling the main loop

        Args:
            MODEL_HP:
            MODEL_KPS:
            LOSS_FCT:
            TRAIN_HP:
            TRAIN_KPS:
            W_COUNTER_KPS:
            PARAMS:
            GPU:
            NAME:
            W_COUNTER_HP:
            optimizer_kps:
            LR_START:
            OPTIMIZER_DECAY:
            BG:
            BATCH_SIZE:
            PREFETCH_BATCH_BUFFER:
            EPOCH_STEPS:
            EPOCHS:
            EPOCH_LOG_N:
            EPOCH_SGD_PLATEAU_CHECK:
        """
        self.DESCRIPTION_KPS = DESCRIPTION_KPS
        self.DESCRIPTION_HP = DESCRIPTION_HP
        self.DESCRIPTION_BG = DESCRIPTION_BG
        self.PARAMS_KPS = PARAMS_KPS
        self.PARAMS_HP = PARAMS_HP
        self.PARAMS_BG = PARAMS_BG
        self.TRAIN_KPS = TRAIN_KPS
        self.TRAIN_HP = TRAIN_HP
        self.TRAIN_BG = TRAIN_BG
        self.MODEL_KPS = MODEL_KPS
        self.MODEL_HP = MODEL_HP
        self.MODEL_BG = MODEL_BG
        self.BATCH_SIZE: int = BATCH_SIZE
        self.PREFETCH_BATCH_BUFFER: int = PREFETCH_BATCH_BUFFER
        self.EPOCH_START = EPOCH_START
        self.EPOCH_STEPS: int = EPOCH_STEPS
        self.EPOCHS: int = EPOCHS
        self.EPOCH_LOG_N: int = EPOCH_LOG_N

        self.GPU: int = GPU
        self.NAME: str = NAME
        self.W_COUNTER_BG: int = W_COUNTER_BG
        self.W_COUNTER_HP: int = W_COUNTER_HP
        self.W_COUNTER_KPS: int = W_COUNTER_KPS
        self.SGD_CLR_DECAY_COUNTER = 0
        self.OPTIMIZER_NAME_BG = OPTIMIZER_NAME_BG
        self.OPTIMIZER_NAME_HP = OPTIMIZER_NAME_HP
        self.OPTIMIZER_NAME_KPS = OPTIMIZER_NAME_KPS
        self.LR_START_BG, self.LR_START_HP, self.LR_START_KPS = LR_START_BG, LR_START_HP, LR_START_KPS
        self.bg_loss_fn, self.hp_loss_fn, self.kps_loss_fn = LOSS_FCT_BG, LOSS_FCT_HP, LOSS_FCT_KPS
        self.step_custom_lr = 0

        self.N_CLASS: int = 9
        self.IMG_SHAPE: [int, int, int] = (240, 320, 3)

        self.kps_subdir = 'kps'
        if self.kps_loss_fn.name == 'KPSLoss':
            self.kps_subdir = 'kps_map'

        set_gpus(GPU)

        self.generator, self.iter, self.sample_frame, self.sample_mask, self.sample_kps = self._generate_dataset(BG)
        _, self.iter_test, _, _, _ = self._generate_dataset(BG, test=True)

        self.bgmodel = self._get_bg_model()
        self.bgmodel.summary()

        self.base_model = self._get_hrnet_model()
        self.base_model.summary()
        if self.TRAIN_HP:
            tf.keras.utils.plot_model(
                self.base_model, to_file=f'nets/imgs/{self.NAME}_e.png', show_shapes=True, expand_nested=True)
            tf.keras.utils.plot_model(
                self.base_model, to_file=f'nets/imgs/{self.NAME}.png', show_shapes=True, expand_nested=False)

        # prepare bg model training
        self.decay_rate_bg, self.optimizer_decay_bg = PARAMS_KPS.sgd_clr_decay_rate, PARAMS_KPS.decay
        self.optimizer_bg, self.bg_decay_rate_counter = self._get_optimizer(OPTIMIZER_NAME_KPS, LR_START_KPS,
                                                                            PARAMS_KPS,
                                                                            0,
                                                                            self.optimizer_decay_bg)

        if self.TRAIN_BG:
            tf.keras.utils.plot_model(
                self.bgmodel, to_file=f'nets/imgs/{self.NAME}_e.png', show_shapes=True, expand_nested=True)
            tf.keras.utils.plot_model(
                self.bgmodel, to_file=f'nets/imgs/{self.NAME}.png', show_shapes=True, expand_nested=False)

        # prepare hp model training
        self.decay_rate_hp, self.optimizer_decay_hp = PARAMS_BG.sgd_clr_decay_rate, PARAMS_BG.decay
        self.optimizer_hp, self.hp_decay_rate_counter = self._get_optimizer(OPTIMIZER_NAME_BG, LR_START_BG, PARAMS_BG,
                                                                            0,
                                                                            self.optimizer_decay_hp)

        # prepare kps model training
        self.decay_rate_kps, self.optimizer_decay_kps = PARAMS_KPS.sgd_clr_decay_rate, PARAMS_KPS.decay
        self.optimizer_kps, self.kps_decay_rate_counter = self._get_optimizer(OPTIMIZER_NAME_KPS, LR_START_KPS,
                                                                              PARAMS_KPS,
                                                                              0,
                                                                              self.optimizer_decay_kps)
        self.model = self._get_kps_model()
        self.model.summary()
        if self.TRAIN_KPS:
            tf.keras.utils.plot_model(
                self.model, to_file=f'nets/imgs/{self.NAME}_e.png', show_shapes=True, expand_nested=True)
            tf.keras.utils.plot_model(
                self.model, to_file=f'nets/imgs/{self.NAME}.png', show_shapes=True, expand_nested=False)

        self.metric_hp_correct_px_train = Metric('hp_correct_px')
        self.metric_hp_correct_px_test = Metric('hp_correct_px')
        self.metric_hp_acc_body_part_train = Metric('hp_correct_body_part_px_ratio')
        self.metric_hp_acc_body_part_test = Metric('hp_correct_body_part_px_ratio')
        self.metric_hp_correct_px_body_part_train = Metric('hp_correct_body_part_px')
        self.metric_hp_correct_px_body_part_test = Metric('hp_correct_body_part_px')

        self.metric_bg_loss_train = Metric(f'bg_loss')
        self.metric_bg_loss_test: Metric = Metric(f'bg_loss')

        self.metric_hp_loss_train = Metric('hp_loss')
        self.metric_hp_loss_test = Metric('hp_loss')

        self.metric_kps_loss_train = Metric(f'{self.kps_subdir}_loss')
        self.metric_kps_loss_test: Metric = Metric(f'{self.kps_subdir}_loss')

        # self.loss_fn = CILoss(self.N_CLASS)
        # self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _generate_dataset(self, BG: bool, test: bool = False, sequential: bool = False):
        generator = DsGenerator(resize_shape_x=240, rgb=BG, test=test, sequential=sequential)

        sample_pair: DsPair = next(generator.get_next_pair())

        self.IMG_SHAPE = sample_pair['frame'].shape
        self.N_CLASS = np.max(sample_pair['mask']).astype(int) + 1
        self.KPS_COUNT = len(sample_pair['kps'])

        ds = generator.build_iterator(self.BATCH_SIZE, self.PREFETCH_BATCH_BUFFER)

        return generator, ds.as_numpy_iterator(), sample_pair['frame'], sample_pair['mask'], sample_pair['kps']

    def _get_bg_model(self) -> tf.keras.Model:
        bgnet = self.MODEL_BG(self.IMG_SHAPE, 2)

        model = bgnet.model

        if self.W_COUNTER_BG != -1:
            model.load_weights(f"./ckpt{self.GPU}/hp-{self.W_COUNTER_BG}.ckpt")
        elif not self.TRAIN_BG:
            model.load_weights(f"./ckpt/bg-4400.ckpt")
        if not self.TRAIN_BG:
            model.trainable = False

        return model

    def _get_hrnet_model(self) -> tf.keras.Model:
        hrnet = self.MODEL_HP(self.IMG_SHAPE, 9)

        model = hrnet.model

        if self.W_COUNTER_HP != -1:
            model.load_weights(f"./ckpt{self.GPU}/hp-{self.W_COUNTER_HP}.ckpt")
        elif not self.TRAIN_HP:
            model.load_weights(f"./ckpt/hrnet-4400.ckpt")
        if not self.TRAIN_HP:
            model.trainable = False

        return model

    def _get_kps_model(self) -> tf.keras.Model:
        kp_detector = self.MODEL_KPS(input_shape=self.IMG_SHAPE, hrnet_input=self.base_model,
                                     output_channels=int(self.KPS_COUNT))
        model = kp_detector.model
        if self.W_COUNTER_KPS != -1:
            model.load_weights(f"./ckpt{self.GPU}/{self.kps_subdir}-{self.W_COUNTER_KPS}.ckpt")

        return model

    def _get_optimizer(self, optimizer_name, lr_start, params: HyperParameterParams, decay_rate_counter=None,
                       optimizer_decay=None):

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

    def _calculate_lr(self, epoch, metric, optimizer_name,
                      optimizer_decay, decay_rate_counter, params,
                      w_counter, lr_start, step_custom_lr) -> [any, float]:

        _optimizer = None
        _decay_rate_counter = decay_rate_counter

        if optimizer_name == 'sgd':
            lr = lr_start * (
                    1. / (1. + optimizer_decay * (epoch - w_counter) * self.EPOCH_STEPS))
        elif optimizer_name == 'sgd_clr':
            lr = lr_start * 1 / (1 + optimizer_decay * step_custom_lr)

            if metric.is_curve_steep() == False:
                _optimizer, _decay_rate_counter = self._get_optimizer(optimizer_name, lr_start, params,
                                                                      decay_rate_counter, optimizer_decay)
                step_custom_lr = 0
        else:
            lr = lr_start
        return _optimizer, lr, step_custom_lr, _decay_rate_counter

    def _track_logs(self, subdir: str, model, log_detail: bool = False):
        log_dir = f"{Path.cwd()}/logs/metrics/{subdir}/{self.NAME}/{datetime.now().strftime('%Y_%m_%d__%H_%M')}"
        file_writer_test = tf.summary.create_file_writer(
            f"{log_dir}/scalars/test")
        # file_writer.set_as_default()
        progress_tracker = DisplayCallback(self.bgmodel, self.base_model, self.model, subdir, self.sample_frame,
                                           self.sample_mask,
                                           self.sample_kps, self.EPOCHS,
                                           self.GPU, log_dir=log_dir)
        progress_tracker.set_model(model)

        return file_writer_test, progress_tracker, f"{log_dir}/train"

    def _calculate_metrics_hp_train(self, loss_value: float):
        self.metric_hp_loss_train.append(loss_value)
        if isinstance(self.hp_loss_fn, CILoss):
            self.metric_hp_correct_px_train.append(self.hp_loss_fn.correct_predictions.astype(np.float32))
            self.metric_hp_correct_px_body_part_train.append(self.hp_loss_fn.correct_body_part_pred.astype(np.float32))
            self.metric_hp_acc_body_part_train.append(
                (self.hp_loss_fn.correct_body_part_pred / self.hp_loss_fn.body_part_px_n_true).astype(np.float32))

    def _calculate_metrics_hp_test(self, loss_value: float):
        self.metric_hp_loss_test.append(loss_value)
        if isinstance(self.hp_loss_fn, CILoss):
            self.metric_hp_correct_px_test.append(self.hp_loss_fn.correct_predictions.astype(np.float32))
            self.metric_hp_correct_px_body_part_test.append(self.hp_loss_fn.correct_body_part_pred.astype(np.float32))
            self.metric_hp_acc_body_part_test.append(
                (self.hp_loss_fn.correct_body_part_pred / self.hp_loss_fn.body_part_px_n_true).astype(np.float32))

    def _test_model_bg(self, file_writer: ResourceSummaryWriter, epoch: int,
                       sequential: bool = False) -> float:

        for _ in range(self.EPOCH_STEPS):
            batch = next(self.iter_test)
            logits = self.bgmodel(batch['frame'], training=False)
            loss_value = self.bg_loss_fn(batch['mask'], logits)
            self.metric_bg_loss_test.append(float(loss_value))

        with file_writer.as_default():
            tf.summary.scalar(self.metric_bg_loss_test.name, self.metric_bg_loss_test.get_median(), step=epoch)

        return loss_value

    def _test_model_hp(self, file_writer: ResourceSummaryWriter, epoch: int,
                       sequential: bool = False):
        test_accuracy = tf.keras.metrics.Accuracy()

        for i in range(self.EPOCH_STEPS):
            # training=False is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            batch = next(self.iter_test)
            logits = self.base_model(batch['frame'], training=False)
            loss_value = self.hp_loss_fn(batch['mask'], logits)
            max_logits = tf.argmax(logits, axis=-1)
            prediction = max_logits[..., tf.newaxis]
            test_accuracy(prediction, batch['mask'])

            self._calculate_metrics_hp_test(loss_value)

        loss = self.metric_hp_loss_test.get_median()

        with file_writer.as_default():
            tf.summary.scalar(self.metric_hp_loss_test.name, loss, step=epoch)
            tf.summary.scalar('hp_accuracy', test_accuracy.result(), step=epoch)

            if isinstance(self.hp_loss_fn, CILoss):
                tf.summary.scalar(self.metric_hp_acc_body_part_test.name,
                                  self.metric_hp_acc_body_part_test.get_median(), step=epoch)
                tf.summary.scalar(self.metric_hp_correct_px_body_part_test.name,
                                  self.metric_hp_correct_px_body_part_test.get_median(), step=epoch)
                tf.summary.scalar(self.metric_hp_correct_px_test.name, self.metric_hp_correct_px_test.get_median(),
                                  step=epoch)

        return test_accuracy.result()

    def _test_model_kps(self, file_writer: ResourceSummaryWriter, epoch: int,
                        sequential: bool = False) -> float:

        for _ in range(self.EPOCH_STEPS):
            batch = next(self.iter_test)
            logits = self.model(batch['frame'], training=False)
            loss_value = self.kps_loss_fn(batch['kps'], logits)
            self.metric_kps_loss_test.append(float(loss_value))

        with file_writer.as_default():
            tf.summary.scalar(self.metric_kps_loss_test.name, self.metric_kps_loss_test.get_median(), step=epoch)

        return loss_value

    def _train_bg(self):
        batch: DsPair = next(self.iter)

        with tf.GradientTape() as tape:
            logits = self.bgmodel(batch['frame'], training=True)
            loss_value = self.bg_loss_fn(batch['mask'], logits)

        grads = tape.gradient(loss_value, self.bgmodel.trainable_weights)

        self.optimizer_bg.apply_gradients(zip(grads, self.bgmodel.trainable_weights))
        self.metric_bg_loss_train.append(float(loss_value))

        return self.bg_loss_fn.y_true_maps, tf.abs(logits)

    def _train_hp(self, train_acc_metric):
        batch: DsPair = next(self.iter)

        with tf.GradientTape() as tape:
            logits = self.base_model(batch['frame'], training=True)
            loss_value = self.hp_loss_fn(batch['mask'], logits)

        grads = tape.gradient(loss_value, self.base_model.trainable_weights)
        self.optimizer_hp.apply_gradients(zip(grads, self.base_model.trainable_weights))

        max_logits = tf.argmax(logits, axis=-1)
        max_logits = max_logits[..., tf.newaxis]

        train_acc_metric(batch['mask'], max_logits)
        self._calculate_metrics_hp_train(loss_value)

        return self.metric_hp_acc_body_part_train.get_median(False)

    def _train_kps(self):
        batch: DsPair = next(self.iter)

        with tf.GradientTape() as tape:
            logits = self.model(batch['frame'], training=True)
            loss_value = self.kps_loss_fn(batch['kps'], logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer_kps.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric_kps_loss_train.append(float(loss_value))

        return self.kps_loss_fn.y_true_maps, tf.abs(logits)

    def _extract_bg(self):
        pass

    def start_train_loop(self):
        bg_train_acc_metric = tf.keras.metrics.Accuracy()
        hp_train_acc_metric = tf.keras.metrics.Accuracy()
        kps_train_acc_metric = tf.keras.metrics.Accuracy()
        bg_metric_avg_acc = Metric('metric_avg_acc_bg')
        hp_metric_avg_acc_body_part = Metric('metric_avg_acc_body_part')
        kps_metric_avg_acc = Metric('metric_avg_acc_kps', max_size=50)
        step_custom_lr_kps = 0
        step_custom_lr_hp = 0
        step_custom_lr_bg = 0
        time_start = 0
        logger = Logger()
        bg_progress_tracker, hp_progress_tracker, kps_progress_tracker = None, None, None
        bg_file_writer_test, hp_file_writer_test, kps_file_writer_test = None, None, None

        if self.TRAIN_BG:
            bg_file_writer_test, bg_progress_tracker, log_dir = self._track_logs('bg', self.bgmodel)
        if self.TRAIN_HP:
            hp_file_writer_test, hp_progress_tracker, log_dir = self._track_logs('hp', self.base_model)
        if self.TRAIN_KPS:
            kps_file_writer_test, kps_progress_tracker, log_dir = self._track_logs(self.kps_subdir, self.model)

        if self.EPOCH_START == -1:
            start = 0
        else:
            start = self.EPOCH_START

        for epoch in range(start, self.EPOCHS + start):

            if epoch == 3:
                time_start = time.perf_counter()

            for step in range(self.EPOCH_STEPS):
                if self.TRAIN_BG:
                    bg_batch, bg_logits = self._train_bg()
                    bg_train_acc_metric(bg_batch, bg_logits)
                    bg_metric_avg_acc.append(bg_train_acc_metric.result())

                if self.TRAIN_HP:
                    hp_metric_avg_acc_body_part.append(self._train_hp(hp_train_acc_metric))

                if self.TRAIN_KPS:
                    kps_batch, kps_logits = self._train_kps()
                    kps_train_acc_metric(kps_batch, kps_logits)
                    kps_metric_avg_acc.append(kps_train_acc_metric.result())

            if epoch == start + 3:
                if self.TRAIN_BG:
                    bg_progress_tracker \
                        .track_metrics_on_train_start(self.bgmodel, self.NAME, self.OPTIMIZER_NAME_BG,
                                                      self.bg_loss_fn.name, self.LR_START_BG,
                                                      self.TRAIN_BG, self.TRAIN_HP, self.TRAIN_KPS,
                                                      f"{time.perf_counter() - time_start:#.2f}s", self.EPOCHS,
                                                      self.EPOCH_STEPS, self.BATCH_SIZE,
                                                      description=self.DESCRIPTION_BG)
                if self.TRAIN_HP:
                    hp_progress_tracker \
                        .track_metrics_on_train_start(self.base_model, self.NAME, self.OPTIMIZER_NAME_HP,
                                                      self.hp_loss_fn.name, self.LR_START_HP,
                                                      self.TRAIN_BG, self.TRAIN_HP, self.TRAIN_KPS,
                                                      f"{time.perf_counter() - time_start:#.2f}s", self.EPOCHS,
                                                      self.EPOCH_STEPS, self.BATCH_SIZE,
                                                      description=self.DESCRIPTION_HP)
                if self.TRAIN_KPS:
                    kps_progress_tracker \
                        .track_metrics_on_train_start(self.model, self.NAME, self.OPTIMIZER_NAME_KPS,
                                                      self.kps_loss_fn.name, self.LR_START_KPS,
                                                      self.TRAIN_BG, self.TRAIN_HP, self.TRAIN_KPS,
                                                      f"{time.perf_counter() - time_start:#.2f}s", self.EPOCHS,
                                                      self.EPOCH_STEPS, self.BATCH_SIZE,
                                                      description=self.DESCRIPTION_KPS)
            if epoch % self.EPOCH_LOG_N == 0:

                if self.TRAIN_BG:
                    _optimizer_bg, bg_lr, step_custom_lr_bg, self.bg_decay_rate_counter = self._calculate_lr(epoch,
                                                                                                             bg_metric_avg_acc,
                                                                                                             self.OPTIMIZER_NAME_BG,
                                                                                                             self.optimizer_decay_bg,
                                                                                                             self.bg_decay_rate_counter,
                                                                                                             self.PARAMS_BG,
                                                                                                             self.W_COUNTER_BG,
                                                                                                             self.LR_START_BG,
                                                                                                             step_custom_lr_bg)

                    if _optimizer_bg:
                        self.optimizer_bg = _optimizer_bg
                        self.LR_START_BG = self.decay_rate_bg[self.bg_decay_rate_counter]

                    step_custom_lr_bg += 1
                    bg_loss = self.metric_bg_loss_train.get_median()
                    bg_progress_tracker.track_img_on_epoch_end(epoch,
                                                               loss=bg_loss,
                                                               metrics=[

                                                                   Metric(value=bg_train_acc_metric.result(),
                                                                          name='bg_accuracy'),
                                                                   Metric(value=bg_lr, name='bg_learning_rate'),
                                                               ],
                                                               )
                    bg_loss_test = self._test_model_bg(bg_file_writer_test, epoch)
                    bg_train_acc_metric.reset_states()

                    logger.log(
                        f"[{epoch}:{self.EPOCHS + start}]: Body Parts [loss]: {bg_loss} [loss test]: {bg_loss_test}")

                if self.TRAIN_HP:
                    _optimizer_hp, hp_lr, step_custom_lr_hp, self.hp_decay_rate_counter = self._calculate_lr(epoch,
                                                                                                             hp_metric_avg_acc_body_part,
                                                                                                             self.OPTIMIZER_NAME_HP,
                                                                                                             self.optimizer_decay_hp,
                                                                                                             self.hp_decay_rate_counter,
                                                                                                             self.PARAMS_HP,
                                                                                                             self.W_COUNTER_HP,
                                                                                                             self.LR_START_HP,
                                                                                                             step_custom_lr_hp)


                    if _optimizer_hp:
                        self.optimizer_hp = _optimizer_hp
                        self.LR_START_HP = self.decay_rate_hp[self.hp_decay_rate_counter]

                    step_custom_lr_hp += 1
                    hp_loss = self.metric_hp_loss_train.get_median()
                    hp_progress_tracker.track_img_on_epoch_end(epoch,
                                                               loss=hp_loss,
                                                               metrics=[
                                                                   self.metric_hp_correct_px_train,
                                                                   self.metric_hp_correct_px_body_part_train,
                                                                   self.metric_hp_acc_body_part_train,
                                                                   Metric(value=hp_train_acc_metric.result(),
                                                                          name='hp_accuracy'),
                                                                   Metric(value=hp_lr, name='hp_learning_rate'),
                                                               ],
                                                               )
                    hp_loss_test = self._test_model_hp(hp_file_writer_test, epoch)
                    hp_train_acc_metric.reset_states()

                    logger.log(
                        f"[{epoch}:{self.EPOCHS + start}]: Body Parts [loss]: {hp_loss} [loss test]: {hp_loss_test}")

                if self.TRAIN_KPS:
                    _optimizer_kps, kps_lr, step_custom_lr_kps, self.kps_decay_rate_counter = self._calculate_lr(epoch,
                                                                                                                 self.metric_kps_loss_train,
                                                                                                                 self.OPTIMIZER_NAME_KPS,
                                                                                                                 self.optimizer_decay_kps,
                                                                                                                 self.kps_decay_rate_counter,
                                                                                                                 self.PARAMS_KPS,
                                                                                                                 self.W_COUNTER_KPS,
                                                                                                                 self.LR_START_KPS,
                                                                                                                 step_custom_lr_kps)

                    if _optimizer_kps:
                        self.optimizer_kps = _optimizer_kps
                        self.LR_START_KPS = self.decay_rate_kps[self.kps_decay_rate_counter]

                    step_custom_lr_kps += 1
                    kps_loss = self.metric_kps_loss_train.get_median()
                    kps_progress_tracker.track_img_on_epoch_end(epoch,
                                                                loss=kps_loss,
                                                                metrics=[
                                                                    Metric(value=kps_lr, name='kps_learning_rate'),
                                                                ],
                                                                show_img=False)
                    kps_loss_test = self._test_model_kps(kps_file_writer_test, epoch)
                    kps_train_acc_metric.reset_states()

                    logger.log(
                        f"[{epoch}:{self.EPOCHS + start}]: Keypoint [loss]: {kps_loss} [loss test]: {kps_loss_test}")

                logger.log(message=f"Seen images: {self.generator.seen_samples}", block=True)

        if self.TRAIN_BG:
            bg_progress_tracker.log_on_train_end()
        if self.TRAIN_HP:
            hp_progress_tracker.log_on_train_end()
        if self.TRAIN_KPS and not self.TRAIN_HP:
            kps_progress_tracker.log_on_train_end()



if __name__ == "__main__":
    ArgsNamespace = namedtuple('ArgNamespace',
                               ['gpu', 'name', 'wcounter', 'wcounter_base', 'lr', 'decay', 'opt', 'bs', 'steps',
                                'epochs', 'log_n',
                                'bg'])

    parser = argparse.ArgumentParser(
        description='Train skatingAIs awesome network :)')
    parser.add_argument('--gpu', default=1, help='Which gpu shoud I use?', type=int)
    parser.add_argument('--name', default="kpsdetector_relu_reduce_max_hrnet_v7", help='Name for training')
    parser.add_argument('--wcounter', default=-1, help='Weight counter', type=int)
    parser.add_argument('--wcounter_base', default=4400, help='Weight counter for base net', type=int)
    parser.add_argument('--lr', default=0.005, help='Initial learning rate', type=float)
    parser.add_argument('--decay', default=0.01, help='learning rate decay', type=float)
    parser.add_argument('--opt', default="adam", help='Optimizer [nadam, adam, sgd, sgd_clr]')
    parser.add_argument('--bs', default=3, help='Batch size', type=int)
    parser.add_argument('--steps', default=64, help='Epoch steps', type=int)
    parser.add_argument('--epochs', default=5556, help='Epochs', type=int)
    parser.add_argument('--log_n', default=5, help='Epoch steps', type=int)
    parser.add_argument('--bg', default=True, help='Use training images with background', type=bool)
    args: ArgsNamespace = parser.parse_args()

    optimizer = args.opt
    lr = args.lr
    name = args.name

    general_param, bg_params, hps_params, kps_params, train_bg, train_hp, train_kps = TrainProgram().create_menu()

    if not train_hp:
        name = f"kps_{kps_params.name}"
    elif not train_kps:
        name = f"hp_{hps_params.name}"
    else:
        name = f"hp_{hps_params.name}:kps_{kps_params.name}"

    MainLoop(GPU=general_param.gpu, NAME=name,
             MODEL_BG=bg_params.model, MODEL_HP=hps_params.model, MODEL_KPS=kps_params.model,
             OPTIMIZER_NAME_BG=bg_params.optimizer_name, LR_START_BG=bg_params.learning_rate,
             OPTIMIZER_NAME_HP=hps_params.optimizer_name, LR_START_HP=hps_params.learning_rate,
             OPTIMIZER_NAME_KPS=kps_params.optimizer_name, LR_START_KPS=hps_params.learning_rate,
             LOSS_FCT_BG=bg_params.loss_fct, LOSS_FCT_HP=hps_params.loss_fct, LOSS_FCT_KPS=kps_params.loss_fct,
             PARAMS_BG=bg_params.params, PARAMS_HP=hps_params.params, PARAMS_KPS=kps_params.params,
             DESCRIPTION_BG=bg_params.description, DESCRIPTION_HP=hps_params.description,
             DESCRIPTION_KPS=kps_params.description,
             TRAIN_BG=train_bg, TRAIN_HP=train_hp, TRAIN_KPS=train_kps,
             W_COUNTER_BG=general_param.wcounter_bg, W_COUNTER_HP=general_param.wcounter_hp,
             W_COUNTER_KPS=general_param.wcounter_kps,
             EPOCH_START=general_param.epoch_start,
             BATCH_SIZE=general_param.batch_size,
             PREFETCH_BATCH_BUFFER=1, EPOCH_STEPS=general_param.epoch_steps, EPOCHS=general_param.epochs,
             EPOCH_LOG_N=general_param.epoch_log_n, BG=args.bg,
             ).start_train_loop()
