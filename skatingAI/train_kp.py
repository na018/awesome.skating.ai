import argparse
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from skatingAI.nets.hrnet.v7 import HRNet
from skatingAI.nets.keypoint.v2 import KPDetector
from skatingAI.utils.DsGenerator import DsGenerator, DsPair
from skatingAI.utils.utils import DisplayCallback, set_gpus, Metric, Logger


class MainLoop(object):
    def __init__(self, GPU: int, NAME: str, W_COUNTER: int, W_COUNTER_BASE: int, optimizer: str, LR_START: float,
                 OPTIMIZER_DECAY: float,
                 BG: bool,
                 BATCH_SIZE=3,
                 PREFETCH_BATCH_BUFFER=1, EPOCH_STEPS=64, EPOCHS=5555, EPOCH_LOG_N=5, EPOCH_SGD_PLATEAUCHECK=50):
        """this is the class handling the main loop

        Args:
            GPU:
            NAME:
            W_COUNTER:
            optimizer:
            LR_START:
            OPTIMIZER_DECAY:
            BG:
            BATCH_SIZE:
            PREFETCH_BATCH_BUFFER:
            EPOCH_STEPS:
            EPOCHS:
            EPOCH_LOG_N:
            EPOCH_SGD_PLATEAUCHECK:
        """
        self.BATCHSIZE: int = BATCH_SIZE
        self.PREFETCH_BATCH_BUFFER: int = PREFETCH_BATCH_BUFFER
        self.EPOCH_STEPS: int = EPOCH_STEPS
        self.EPOCHS: int = EPOCHS
        self.EPOCH_LOG_N: int = EPOCH_LOG_N
        self.EPOCH_SGD_PLATEAUCHECK: int = EPOCH_SGD_PLATEAUCHECK

        self.GPU: int = GPU
        self.NAME: str = NAME
        self.W_COUNTER: int = W_COUNTER
        self.W_COUNTER_BASE: int = W_COUNTER_BASE
        self.LR_START: float = LR_START
        self.SGD_CLR_DECAY_RATE = [1e-5, 1e-4, 1e-3, 0.01]
        self.SGD_CLR_DECAY_COUNTER = 0
        self.OPTIMIZER_DECAY: float = OPTIMIZER_DECAY
        self.OPTIMIZER_NAME = optimizer
        self.step_custom_lr = 0

        self.N_CLASS: int = 9
        self.IMG_SHAPE: [int, int, int] = (240, 320, 3)

        set_gpus(GPU)

        self.iter, self.sample_frame, self.sample_mask, self.sample_kps = self._generate_dataset(BG)
        self.base_model = self._get_hrnet_model()
        self.model = self._get_model()
        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file=f'{self.NAME}_e.png', show_shapes=True, expand_nested=True)
        tf.keras.utils.plot_model(
            self.model, to_file=f'{self.NAME}.png', show_shapes=True, expand_nested=False)
        self.optimizer = self._get_optimizer()

        # self.loss_fn = CILoss(self.N_CLASS)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _generate_dataset(self, BG: bool):
        self.generator = DsGenerator(resize_shape_x=240, rgb=BG)

        sample_pair: DsPair = next(self.generator.get_next_pair())

        self.IMG_SHAPE = sample_pair['frame'].shape
        self.N_CLASS = np.max(sample_pair['mask']).astype(int) + 1
        self.KPS_COUNT = len(sample_pair['kps'])

        train_ds = self.generator.build_iterator(self.IMG_SHAPE, self.BATCHSIZE, self.PREFETCH_BATCH_BUFFER)

        return train_ds.as_numpy_iterator(), sample_pair['frame'], sample_pair['mask'], sample_pair['kps']

    def _get_hrnet_model(self) -> tf.keras.Model:
        hrnet = HRNet(self.IMG_SHAPE, int(self.N_CLASS))

        model = hrnet.model
        model.load_weights(f"./ckpt1/hrnet-{self.W_COUNTER_BASE}.ckpt")
        base_model = tf.keras.Model(inputs=hrnet.inputs, outputs=hrnet.model.layers[-1].output)
        base_model.trainable = False

        return base_model

    def _get_model(self) -> tf.keras.Model:
        kp_detector = KPDetector(input_shape=(240, 320, 3), hrnet_input=self.base_model,
                                 output_channels=int(self.KPS_COUNT))
        model = kp_detector.model
        return model

    def _get_optimizer(self):

        if self.OPTIMIZER_NAME == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.LR_START, epsilon=1e-8, amsgrad=True)  # 0.001
        elif self.OPTIMIZER_NAME == "nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.LR_START, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        elif self.OPTIMIZER_NAME == "sgd_clr":
            self.OPTIMIZER_DECAY = self.SGD_CLR_DECAY_RATE[self.SGD_CLR_DECAY_COUNTER]
            if self.SGD_CLR_DECAY_COUNTER + 1 < len(self.SGD_CLR_DECAY_RATE):
                self.SGD_CLR_DECAY_COUNTER += 1

            optimizer = tf.keras.optimizers.SGD(learning_rate=self.LR_START, momentum=0.9, decay=self.OPTIMIZER_DECAY,
                                                nesterov=True)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.LR_START, momentum=0.9, decay=self.OPTIMIZER_DECAY,
                                                nesterov=True)

        return optimizer

    def _track_logs(self, log_detail: bool = False):
        file_writer = tf.summary.create_file_writer(
            f"{Path.cwd()}/logs/metrics/kp/{self.NAME}/{datetime.now().strftime('%Y_%m_%d__%H_%M')}")
        file_writer.set_as_default()
        progress_tracker = DisplayCallback(self.base_model, self.model, self.sample_frame, self.sample_mask,
                                           self.sample_kps, file_writer, self.EPOCHS,
                                           self.GPU)
        log1 = Logger(log=log_detail)
        log2 = Logger(log=True)

        return file_writer, progress_tracker, log1, log2

    def _calculate_metrics(self):
        self.metric_correct_px.append(self.loss_fn.correct_predictions.astype(np.float32))
        self.metric_correct_px_body_part.append(self.loss_fn.correct_body_part_pred.astype(np.float32))
        self.metric_acc_body_part.append(
            (self.loss_fn.correct_body_part_pred / self.loss_fn.body_part_px_n_true).astype(np.float32))
        self.metric_loss.append(self.loss_value)

    def start_train_loop(self):
        file_writer, progress_tracker, log_v, log2 = self._track_logs()
        self.metric_loss = Metric('loss')

        if self.W_COUNTER == -1:
            start = 0
        else:
            start = self.W_COUNTER

        for epoch in range(start, self.EPOCHS):
            log2.log(message=f"Start of epoch {epoch}", block=True)

            for step in range(self.EPOCH_STEPS):
                log_v.log(message=f"Step {step}", block=True)
                batch: DsPair = next(self.iter)
                log_v.log(message=f"got batch")

                with tf.GradientTape() as tape:
                    logits = self.model(batch['frame'], training=True)
                    self.loss_value = self.loss_fn(batch['kps'], logits)

                log_v.log(message=f"Calculated Logits & loss value")
                grads = tape.gradient(self.loss_value, self.model.trainable_weights)
                log_v.log(message=f"Calculated Grads")

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                log_v.log(message=f"Optimized gradients")

                self.metric_loss.append(self.loss_value)
                self.step_custom_lr += 1

            if epoch % self.EPOCH_LOG_N == 0:
                log2.log(
                    message=f"loss: [{self.loss_value.numpy()}]", block=False)

                progress_tracker.on_epoch_end(epoch,
                                              loss=self.metric_loss.get_median(),
                                              show_img=False)
                log2.log(message=f"Seen images: {self.generator.seen_samples}", block=True)

        progress_tracker.on_train_end()
        log_v.log_end()


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
    parser.add_argument('--bg', default=False, help='Use training images with background', type=bool)
    args: ArgsNamespace = parser.parse_args()

    MainLoop(args.gpu, args.name, args.wcounter, args.wcounter_base, args.opt, args.lr, args.decay, args.bg,
             BATCH_SIZE=args.bs,
             EPOCH_STEPS=args.steps, EPOCHS=args.epochs, EPOCH_LOG_N=args.log_n).start_train_loop()
