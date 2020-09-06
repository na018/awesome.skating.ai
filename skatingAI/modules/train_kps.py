from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skatingAI.modules.TrainBase import TrainBase
from skatingAI.nets.hrnet import HPNetBase
from skatingAI.utils.DsGenerator import DsPair
from skatingAI.utils.utils import Metric, plot2img


class TrainKP(TrainBase):

    def __init__(self, NN: type(HPNetBase), name: str, img_shape, optimizer_name: str,
                 lr_start: float, loss_fct: tf.keras.losses, params,
                 description, train: bool, w_counter, gpu: int, epochs: int,
                 bg_extractor: tf.keras.Model, hp_model: tf.keras.Model):

        super().__init__(name, img_shape, optimizer_name, lr_start, loss_fct, params, description, train, w_counter,
                         gpu, epochs)

        self.name = 'kps'
        if train:
            self.bg_extractor: tf.keras.Model = bg_extractor
            self.hp_model = hp_model

            self.model = self._get_model(NN)
            self.model.summary()
            tf.keras.utils.plot_model(self.model, 'nets/imgs/kps_model.png', show_shapes=True, expand_nested=True)

            self.decay_rate, self.optimizer_decay = params.sgd_clr_decay_rate, params.decay
            self.optimizer, self.decay_rate_counter = self._get_optimizer(optimizer_name, lr_start,
                                                                          params,
                                                                          0,
                                                                          self.optimizer_decay)
            self.metric_loss_train = Metric(f'kp_loss')
            self.metric_loss_test: Metric = Metric(f'kp_loss')

            self.progress_tracker, self.file_writer_test = self._create_display_cb(self.model, 'kps')

    def _get_model(self, NN) -> tf.keras.Model:

        kpnet = NN(input_shape=self.img_shape, bgnet_input=self.bg_extractor,
                   hrnet_input=self.hp_model, output_channels=12)

        model = kpnet.model

        if self.w_counter != -1:
            model.load_weights(f"{Path.cwd()}/ckpt{self.gpu}/kps-{self.w_counter}.ckpt")

        return model

    def track_logs(self, sample_image, sample_kp, epoch, counter, **kwargs):
        if self._train:

            extracted_bg = self.bg_extractor.predict(sample_image[tf.newaxis, ...])[0]
            frames_extracted_bg = sample_image.numpy()

            predicted_kp = self.model.predict(sample_image[tf.newaxis, ...])[0]
            # predicted_mask = mask2rgb(create_mask(self.hp_model.predict(sample_image[tf.newaxis, ...])[0]))

            sample_kp = tf.argmax(sample_kp, axis=-1)
            predicted_kp_img = tf.argmax(predicted_kp, axis=-1)
            display_imgs = [[frames_extracted_bg, []],
                            [sample_kp, []],
                            [predicted_kp_img, []]
                            ]
            title = ['Input Image', 'Predicted Mask & True Keypoints', 'Predicted Keypoints']
            fig = plt.figure(figsize=(15, 4))
            for i, img in enumerate(display_imgs):
                ax = fig.add_subplot(1, 3, i + 1)
                for circle in img[1]:
                    ax.add_patch(circle)
                ax.set_title(title[i], fontsize='small', alpha=0.6, color='blue')
                ax.imshow(img[0])

            fig.savefig(f"{Path.cwd()}/img_train{self.gpu}/{epoch}_{counter}_kps_train.png")

            img = plot2img(fig)


            self.progress_tracker.track_img_on_epoch_end(img, epoch, metrics=[
                self.metric_loss_train,
                self.metric_loss_test
            ])


    def track_metrics_on_train_start(self, do_train_bg, do_train_hp, time, epoch_steps, batch_size):
        if self._train:
            self.progress_tracker \
                .track_metrics_on_train_start(self.model, self.name, self.optimizer_name,
                                              self.loss_fct.name, self.lr_start,
                                              do_train_hp, do_train_bg, self._train,
                                              time, self.epochs,
                                              epoch_steps, batch_size,
                                              description=self.description)

    def test_model(self, epoch: int, epoch_steps: int, test_batch) -> float:
        if self._train:

            loss_value = 0

            for i, batch in enumerate(test_batch):
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
            return self.metric_loss_test.get_median(False)

    def train_model(self, iter):
        batch: DsPair = next(iter)


        with tf.GradientTape() as tape:
            logits = self.model(batch, training=True)
            loss_value = self.loss_fct(batch['kps'], logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric_loss_train.append(float(loss_value))

        return loss_value
