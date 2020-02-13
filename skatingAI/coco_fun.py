import tensorflow as tf
from enum import Enum
from pycocotools.coco import COCO

coco = COCO("Data/coco/annotations/person_keypoints_train2017.json") #64115 images
imgIds = coco.getImgIds(catIds=[1])
# print(f"Coco dataset contains {len(imgIds)} annotated images.")
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 255, 255]]

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

# normalize image to [0,1]
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # segmentation mask {1,2,3} -> {0,1,2}
    input_mask -= 1
    return input_image, input_mask


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




def create_image_mask(kps, width=640, height=427):
    img = np.zeros((height, width, 1))
    for i, kp in enumerate(kps):
          for j, p in enumerate(kp):
               rr, cc = draw.circle(p[1],p[0], 5, shape=(height, width))
               img[rr,cc]=i

    # blur image smoothly
    return img

def get_random_train_image(amount, width=640, height=427):
    i=0
    while i < amount:
        kps=[]

        img = None
        while img is None:
            randomImg = imgIds[np.random.randint(0, len(imgIds))]
            coco_img = coco.loadImgs([randomImg])[0]

            if coco_img['width'] == width and coco_img['height'] == height:
                annIds = coco.getAnnIds(imgIds=coco_img['id'], iscrowd=None)
                anns = coco.loadAnns(annIds)
                kps = []

                for ann in anns:
                    kp = np.array(ann.get('keypoints')).reshape((-1, 3))[:, :2].reshape(-1,2)
                    kps.append(kp)

                img = cv2.imread(f"Data/coco/imgs/train2017/{coco_img['file_name']}")


        mask = create_image_mask(tuple(zip(*kps))).astype('float32') /255
        img = img.astype('float32') / 255
        i += 1
        yield img, mask

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
class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, sample_image, sample_mask, file_writer, epochs=5):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.model = model
        self.epochs = epochs
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None):
        print('epoch_end')
        clear_output(wait=True)
        show_predictions(self.model, self.sample_image,
                         self.sample_mask, name=f"prediction-{epoch}")

        self.model.save_weights(
            f"{Path.cwd()}/ckpt/hrnet-{epoch}.ckpt")


        #plt.imshow(self.sample_mask)   #
        for i, img in enumerate([self.sample_image[newaxis, ...], self.model.predict(self.sample_image[newaxis, ...])]):
            summary.image(f"Training_{epoch}", img, step=i)

        # tf.summary.image(f"Training_{epoch}", [
        #                  self.sample_image, self.sample_mask, self.model.predict(self.sample_image[tf.newaxis, ...])], step=0)

        print(f"\nSimple Prediction after epoch {epoch+1}")

    def on_train_end(self, logs=None):
        #clear_output(wait=True)
        print('train_end')
        #K.clear_session()
        print(f"\n\n{'='*100}\nSuccessfully trained {self.epochs} epochs.\n" +
              f"For evaluation (loss/ accuracy) please run \n${' '*5}`tensorboard --logdir {Path.cwd()}/logs`\n" +
              f"and open your webbrowser at `http://localhost:6006`\n")

#tensorflow.python.framework.errors_impl.InvalidArgumentError: Shapes of all inputs must match: values[0].shape = [128,128,3] != values[1].shape = [128,128,1] [Op:Pack] name: Training_0/write_summary/assert_rank/x/

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








