import argparse
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from skatingAI.nets.hrnet.v7 import HRNet
from skatingAI.utils.DsGenerator import DsGenerator
from skatingAI.utils.losses import CILoss
from skatingAI.utils.utils import DisplayCallback, set_gpus, Metric, Logger


class MainLoop(object):
    """this is the class handling the main loop

     Args:
            GPU:
            NAME:
            W_COUNTER:
            LR_START:
            OPTIMIZER_DECAY:
            BATCH_SIZE:
            PREFETCH_BATCH_BUFFER:
            EPOCH_STEPS:
            EPOCHS:
            EPOCH_LOG_N:
     """

    def __init__(self, GPU: int, NAME: str, W_COUNTER: int, optimizer: str, LR_START: float, OPTIMIZER_DECAY: float,
                 BATCH_SIZE=3,
                 PREFETCH_BATCH_BUFFER=1, EPOCH_STEPS=64, EPOCHS=5555, EPOCH_LOG_N=5):
        self.BATCHSIZE: int = BATCH_SIZE
        self.PREFETCH_BATCH_BUFFER: int = PREFETCH_BATCH_BUFFER
        self.EPOCH_STEPS: int = EPOCH_STEPS
        self.EPOCHS: int = EPOCHS
        self.EPOCH_LOG_N: int = EPOCH_LOG_N

        self.GPU: int = GPU
        self.NAME: str = NAME
        self.W_COUNTER: int = W_COUNTER
        self.LR_START: float = LR_START
        self.OPTIMIZER_DECAY: int = OPTIMIZER_DECAY
        self.optimizer = optimizer

        self.N_CLASS: int = 9
        self.IMG_SHAPE: Tuple[int] = (240, 320, 3)

        set_gpus(GPU)

        self.iter, self.sample_frame, self.sample_mask = self._generate_dataset()
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()

        self.loss_fn = CILoss(self.N_CLASS)

    def _generate_dataset(self):
        self.generator = DsGenerator(resize_shape=(240, 320))

        sample_pair = next(self.generator.get_next_pair())

        self.IMG_SHAPE = sample_pair['frame'].shape
        self.N_CLASS = np.max(sample_pair['mask']).astype(int) + 1

        train_ds = self.generator.build_iterator(self.IMG_SHAPE, self.BATCHSIZE, self.PREFETCH_BATCH_BUFFER)

        return train_ds.as_numpy_iterator(), sample_pair['frame'], sample_pair['mask']

    def _get_model(self) -> tf.keras.Model:
        hrnet = HRNet(self.IMG_SHAPE, int(self.N_CLASS))

        model = hrnet.model
        model.summary()

        if self.W_COUNTER != -1:
            model.load_weights(f"./ckpt{self.GPU}/hrnet-{self.W_COUNTER}.ckpt")
        else:
            self.W_COUNTER = 0

        tf.keras.utils.plot_model(
            model, to_file=f'nadins_{self.NAME}_e.png', show_shapes=True, expand_nested=True)
        tf.keras.utils.plot_model(
            model, to_file=f'nadins_{self.NAME}.png', show_shapes=True, expand_nested=False)

        return model

    def _get_optimizer(self):

        if self.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8, amsgrad=True)
        elif self.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.LR_START, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.LR_START, momentum=0.9, decay=self.OPTIMIZER_DECAY,
                                                nesterov=True)

        return optimizer

    def _track_logs(self, log_detail: bool = False):
        file_writer = tf.summary.create_file_writer(
            f"{Path.cwd()}/logs/metrics/{self.NAME}/{datetime.now().strftime('%Y_%m_%d__%H_%M')}")
        file_writer.set_as_default()
        progress_tracker = DisplayCallback(self.model, self.sample_frame, self.sample_mask, file_writer, self.EPOCHS,
                                           self.GPU)
        log1 = Logger(log=log_detail)
        log2 = Logger(log=True)

        return file_writer, progress_tracker, log1, log2

    def _calculate_metrics(self):
        self.metric_acc.append(self.loss_fn.correct_predictions.astype(np.float32))
        self.metric_correct_px_body_part.append(self.loss_fn.correct_body_part_pred.astype(np.float32))
        self.metric_acc_body_part.append(
            (self.loss_fn.correct_body_part_pred / self.loss_fn.body_part_px_n_true).astype(np.float32))
        self.metric_loss.append(self.loss_value)

    def start_train_loop(self):

        train_acc_metric = tf.keras.metrics.Accuracy()
        file_writer, progress_tracker, log_v, log2 = self._track_logs()

        self.metric_correct_px, self.metric_correct_px_body_part, self.metric_acc_body_part, self.metric_acc, self.metric_loss = (
            Metric('correct_px'),
            Metric('correct_body_part_px'),
            Metric('accuracy_body_part'),
            Metric('accuracy'),
            Metric('loss')
        )

        for epoch in range(self.W_COUNTER, self.EPOCHS):
            log2.log(message=f"Start of epoch {epoch}", block=True)

            for step in range(self.EPOCH_STEPS):
                log_v.log(message=f"Step {step}", block=True)
                batch = next(self.iter)
                log_v.log(message=f"got batch")

                with tf.GradientTape() as tape:
                    logits = self.model(batch['frame'], training=True)
                    self.loss_value = self.loss_fn(batch['mask'], logits)

                log_v.log(message=f"Calculated Logits & loss value")
                grads = tape.gradient(self.loss_value, self.model.trainable_weights)
                log_v.log(message=f"Calculated Grads")

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                log_v.log(message=f"Optimized gradients")

                max_logits = tf.argmax(logits, axis=-1)
                max_logits = max_logits[..., tf.newaxis]
                train_acc_metric(batch['mask'], max_logits)
                self._calculate_metrics()

            if epoch % self.EPOCH_LOG_N == 0:
                if self.optimizer in 'nadam':
                    lr = self.LR_START * (
                            1. / (1. + self.OPTIMIZER_DECAY * (epoch - self.W_COUNTER) * self.EPOCH_STEPS))
                else:
                    lr = self.LR_START
                log2.log(
                    message=f"learning_rate: [{lr}]    "
                            f"loss: [{self.loss_value.numpy()}]   "
                            f"acc_body_part: [{np.median(self.metric_acc_body_part.metrics)}]",
                    block=False)

                progress_tracker.on_epoch_end(epoch,
                                              loss=self.metric_loss.get_median(),
                                              metrics=[
                                                  self.metric_correct_px,
                                                  self.metric_correct_px_body_part,
                                                  self.metric_acc_body_part,
                                                  Metric(value=train_acc_metric.result(), name='accuracy'),
                                                  Metric(value=lr, name='learning_rate'),
                                              ],
                                              show_img=False)
                log2.log(message=f"Seen images: {self.generator.seen_samples}", block=True)
            train_acc_metric.reset_states()

        progress_tracker.on_train_end()
        log_v.log_end()


if __name__ == "__main__":
    ArgsNamespace = namedtuple('ArgNamespace',
                               ['gpu', 'name', 'wcounter', 'lr', 'decay', 'opt', 'bs', 'steps', 'log_n'])

    parser = argparse.ArgumentParser(
        description='Train nadins awesome network :)')
    parser.add_argument('--gpu', default=1, help='Which gpu shoud I use?', type=int)
    parser.add_argument('--name', default="hrnet_v7", help='Name for training')
    parser.add_argument('--wcounter', default=2935, help='Weight counter', type=int)
    parser.add_argument('--lr', default=0.01, help='Initial learning rate', type=float)
    parser.add_argument('--decay', default=0.0001, help='learning rate decay', type=float)
    parser.add_argument('--opt', default="nadam", help='Optimizer [Nadam, SGD]')
    parser.add_argument('--bs', default=3, help='Batch size', type=int)
    parser.add_argument('--steps', default=64, help='Epoch steps', type=int)
    parser.add_argument('--log_n', default=5, help='Epoch steps', type=int)
    args: ArgsNamespace = parser.parse_args()

    MainLoop(args.gpu, args.name, args.wcounter, args.opt, args.lr, args.decay, BATCH_SIZE=args.bs,
             EPOCH_STEPS=args.steps, EPOCH_LOG_N=args.log_n).start_train_loop()
