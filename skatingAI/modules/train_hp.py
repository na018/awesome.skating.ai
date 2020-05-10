from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.modules.TrainBase import TrainBase
from skatingAI.nets.hrnet import HPNetBase
from skatingAI.utils.DsGenerator import DsPair
from skatingAI.utils.utils import Metric, create_mask, plot2img, mask2rgb


class TrainHP(TrainBase):

    def __init__(self, NN: type(HPNetBase), name: str, img_shape, optimizer_name: str,
                 lr_start: float, loss_fct: tf.keras.losses, params,
                 description, train: bool, w_counter, gpu: int, epochs: int, bg_extractor):

        super().__init__(name, img_shape, optimizer_name, lr_start, loss_fct, params, description, train, w_counter,
                         gpu, epochs)

        self.model = self._get_model(NN)
        self.progress_tracker, self.file_writer_test = self._create_display_cb(self.model, 'hp')

        if train:
            self.model.summary()

            self.decay_rate, self.optimizer_decay = params.sgd_clr_decay_rate, params.decay
            self.optimizer, self.decay_rate_counter = self._get_optimizer(optimizer_name, lr_start,
                                                                          params,
                                                                          0,
                                                                          self.optimizer_decay)
            self.metric_loss_train = Metric(f'hp_loss')
            self.metric_loss_test: Metric = Metric(f'hp_loss')

            self.bg_extractor: tf.keras.Model = bg_extractor

    def _get_model(self, NN) -> tf.keras.Model:

        hpnet = NN(self.img_shape, 9)

        model = hpnet.model

        if self.w_counter != -1:
            model.load_weights(f"{Path.cwd()}/ckpt{self.gpu}/hp-{self.w_counter}.ckpt")
        elif not self._train:
            model.load_weights(f"{Path.cwd()}/ckpt/hp-4400.ckpt")
        if not self._train:
            model.trainable = False

        return model

    def track_logs(self, sample_image, sample_mask, epoch, **kwargs):
        if self._train:
            extracted_bg = self.bg_extractor.predict(sample_image[tf.newaxis, ...])[0]
            imgs = np.argmax(extracted_bg, axis=-1)
            frames_extracted_bg = sample_image.numpy()
            frames_extracted_bg[imgs == 0] = 2

            predicted_hp = mask2rgb(create_mask(self.model.predict(frames_extracted_bg[tf.newaxis, ...])[0]))

            display_imgs = [cv2.cvtColor(frames_extracted_bg, cv2.COLOR_BGR2RGB),
                            np.reshape(sample_mask, sample_mask.shape[:-1]),
                            predicted_hp]
            title = ['Input Image', 'True Mask', 'Predicted Mask']
            fig = plt.figure(figsize=(15, 4))
            for i, img in enumerate(display_imgs):
                ax = fig.add_subplot(1, 3, i + 1)
                ax.set_title(title[i], fontsize='small', alpha=0.6, color='blue')
                ax.imshow(img)

            fig.savefig(f"{Path.cwd()}/img_train{self.gpu}/{epoch}_hp_train.png")

            img = plot2img(fig)

            self.progress_tracker.track_img_on_epoch_end(img, epoch, metrics=[
                self.metric_loss_train,
                self.metric_loss_test
            ])

    def track_metrics_on_train_start(self, do_train_bg, do_train_kp, time, epoch_steps, batch_size):
        if self._train:
            self.progress_tracker \
                .track_metrics_on_train_start(self.model, self.name, self.optimizer_name,
                                              self.loss_fct.name, self.lr_start,
                                              do_train_bg, self._train, do_train_kp,
                                              time, self.epochs,
                                              epoch_steps, batch_size,
                                              description=self.description)

    def test_model(self, epoch: int, epoch_steps: int, iter_test) -> float:
        if self._train:
            loss_value = 0

            for _ in range(epoch_steps):
                batch: DsPair = next(iter_test)
                extracted_bg = self.bg_extractor.predict([batch['frame']])
                imgs = np.argmax(extracted_bg, axis=-1)
                frames_extracted_bg = np.array(batch['frame'])
                frames_extracted_bg[imgs == 0] = 2

                logits = self.model(frames_extracted_bg, training=False)
                loss_value = self.loss_fct(batch['mask_hp'], logits)
                self.metric_loss_test.append(float(loss_value))

            with self.file_writer_test.as_default():
                tf.summary.scalar(self.metric_loss_test.name, self.metric_loss_test.get_median(),
                                  step=epoch)

            self.metric_loss_test.append(loss_value)

            return loss_value

    def train_model(self, iter):
        if self._train:
            batch: DsPair = next(iter)

            extracted_bg = self.bg_extractor.predict([batch['frame']])
            imgs = np.argmax(extracted_bg, axis=-1)
            frames_extracted_bg = np.array(batch['frame'])
            frames_extracted_bg[imgs == 0] = 2

            with tf.GradientTape() as tape:
                logits = self.model(frames_extracted_bg, training=True)
                loss_value = self.loss_fct(batch['mask_hp'], logits)

            grads = tape.gradient(loss_value, self.model.trainable_weights)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            self.metric_loss_train.append(float(loss_value))

            return loss_value
