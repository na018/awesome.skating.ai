import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import time
from ImageSegmentation.nets.mobilenet_v2 import unet_model, encoder, decoder
from ImageSegmentation.utils import show_predictions, load_image_test


def evaluate(img_amount=5):
    model = unet_model(3, encoder(), decoder())

    model.load_weights(tf.train.latest_checkpoint(f"{Path.cwd()}/data"))

    dataset = tfds.load('oxford_iiit_pet:3.0.0', with_info=False)

    test = dataset['test'].map(load_image_test)
    test_imgs = test.batch(img_amount)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    _, calculated_time, calculated_time_avg = calculate_time(
        model, test_imgs, img_amount)
    save_timings(calculated_time, calculated_time_avg)

    print(f"\n\n{'='*100}\nFor evaluation (loss/ accuracy) please run \n${' '*5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
          f"and open your webbrowser at `http://localhost:6006`")


def calculate_time(model, test_imgs, img_amount):
    predicted_imgs = []

    print(f"\n\n{'-'*100}\nstart predictions [{img_amount}]:")

    start = time.time()
    for image, mask in test_imgs.take(img_amount):
        predicted_imgs.append(model.predict(image))

    calculated_time = time.time()-start
    calculated_time_avg = calculated_time/img_amount

    print(
        f"Time for model to calculate {img_amount} images (no display): {calculated_time}\n" +
        f"avg image calculation time: {calculated_time_avg}")

    return predicted_imgs, calculated_time, calculated_time_avg


def save_timings(calculated_time, calculated_time_avg):
    save_path = Path.cwd() / 'data/timings.md'
    text = ''

    if Path.exists(save_path):
        text = save_path.read_text()

    save_path.write_text(text + "\n", calculate_time, calculated_time_avg)
