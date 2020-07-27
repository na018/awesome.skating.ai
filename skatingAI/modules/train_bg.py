from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.modules.TrainBase import TrainBase
from skatingAI.nets.bg import BGNetBase
from skatingAI.utils.DsGenerator import DsPair
from skatingAI.utils.utils import Metric, create_mask, plot2img


class TrainBG(TrainBase):

    def __init__(self, NN: type(BGNetBase), name: str, img_shape, optimizer_name: str,
                 lr_start: float, loss_fct: tf.keras.losses, params,
                 description, train: bool, w_counter, gpu: int, epochs: int):

        super().__init__(name, img_shape, optimizer_name, lr_start, loss_fct, params, description, train, w_counter,
                         gpu, epochs)
        self.name = 'bg'
        self.model = self._get_model(NN)
        self.progress_tracker, self.file_writer_test = self._create_display_cb(self.model, 'bg')

        if train:
            self.model.summary()
            tf.keras.utils.plot_model(self.model, 'nets/imgs/bg_model.png', show_shapes=True, expand_nested=True)

            self.decay_rate, self.optimizer_decay = params.sgd_clr_decay_rate, params.decay
            self.optimizer, self.decay_rate_counter = self._get_optimizer(optimizer_name, lr_start,
                                                                          params,
                                                                          0,
                                                                          self.optimizer_decay)
            self.metric_loss_train = Metric(f'bg_loss')
            self.metric_loss_test: Metric = Metric(f'bg_loss')

    def _get_model(self, NN) -> tf.keras.Model:

        bgnet = NN(self.img_shape, 2)

        model = bgnet.model

        if self.w_counter != -1:
            model.load_weights(f"./ckpt{self.gpu}/bg-{self.w_counter}.ckpt")
        elif not self._train:
            model.load_weights(f"./ckpt/bg-240.ckpt")
        if not self._train:
            model.trainable = False

        return model

    def track_logs(self, sample_image, sample_mask, epoch, **kwargs):

        predicted_bg = create_mask(self.model.predict(sample_image[tf.newaxis, ...])[0])

        display_imgs = [sample_image.numpy(),
                        np.reshape(sample_mask, sample_mask.shape[:-1]),
                        np.array(np.reshape(predicted_bg, sample_mask.shape[:-1]), dtype=np.float32),
                        ]
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        fig = plt.figure(figsize=(15, 4))
        for i, img in enumerate(display_imgs):
            ax = fig.add_subplot(1, 3, i + 1)

            ax.set_title(title[i], fontsize='small', alpha=0.6, color='blue')
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        fig.savefig(f"{Path.cwd()}/img_train{self.gpu}/{epoch}_bg_train.png")

        img = plot2img(fig)

        self.model.save_weights(
            f"{Path.cwd()}/ckpt/{self.name}-{epoch}.ckpt")

        self.progress_tracker.track_img_on_epoch_end(img, epoch, metrics=[
            self.metric_loss_train,
            self.metric_loss_test
        ])


    def track_metrics_on_train_start(self, do_train_hp, do_train_kp, time, epoch_steps, batch_size):

        self.progress_tracker \
            .track_metrics_on_train_start(self.model, self.name, self.optimizer_name,
                                          self.loss_fct.name, self.lr_start,
                                          self._train, do_train_hp, do_train_kp,
                                          time, self.epochs,
                                          epoch_steps, batch_size,
                                          description=self.description)

    def test_model(self, epoch: int, epoch_steps: int, test_batch) -> float:
        if self._train:

            loss_value = 0

            for i, batch in enumerate(test_batch):
                logits = self.model(batch['frame'], training=False)
                loss_value = self.loss_fct(batch['mask_bg'], logits)
                self.metric_loss_test.append(float(loss_value))

            with self.file_writer_test.as_default():
                tf.summary.scalar(self.metric_loss_test.name, self.metric_loss_test.get_median(), step=epoch)

            self.metric_loss_test.append(loss_value)
            return self.metric_loss_test.get_median(False)

    def train_model(self, iter):
        batch: DsPair = next(iter)

        with tf.GradientTape() as tape:
            logits = self.model(batch['frame'], training=True)
            loss_value = self.loss_fct(batch['mask_bg'], logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric_loss_train.append(float(loss_value))

        return loss_value
        # return self.loss_fct.y_true_maps, tf.abs(logits)
