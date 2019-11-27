import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import clear_output
import random
from pathlib import Path
import os


# normalize image to [0,1]
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # segmentation mask {1,2,3} -> {0,1,2}
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    # randomly flip image
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display_image(display_list, name='segmented_image', save_img=True, show=False):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    if show:
        plt.show()
    if save_img:
        plt.savefig(f"{Path.cwd()}/images/{name}.png")


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, sample_image=None, sample_mask=None, dataset=None, num=1, save_img=True, name='segmented_image'):
    if dataset:
        i = 0
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_image(
                [image[0], mask[0], create_mask(pred_mask)], name=f"{name}-{i}", save_img=save_img)
            i += 1
    else:
        display_image([sample_image, sample_mask,
                       create_mask(model.predict(sample_image[tf.newaxis, ...]))], name=name)


def calculate_loss(model_history, epochs):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(epochs)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()


plt.show()
plt.savefig('training_loss.png')


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_image, sample_mask, file_writer, epochs=5):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.model = model
        self.epochs = epochs
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.model, self.sample_image,
                         self.sample_mask, name=f"prediction-{epoch}")
        self.model.save_weights(
            f"{Path.cwd()}/ckpt/mobilenet_v2-{epoch}.ckpt")

        for i, img in enumerate([self.sample_image[tf.newaxis, ...], self.model.predict(self.sample_image[tf.newaxis, ...])]):
            tf.summary.image(f"Training_{epoch}", img, step=i)

        # tf.summary.image(f"Training_{epoch}", [
        #                  self.sample_image, self.sample_mask, self.model.predict(self.sample_image[tf.newaxis, ...])], step=0)

        print(f"\nSimple Prediction after epoch {epoch+1}")

    def on_train_end(self, logs=None):
        print(f"\n\n{'='*100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' '*5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")
