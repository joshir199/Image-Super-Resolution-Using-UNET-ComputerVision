import tensorflow as tf
from utils.Utils import Utils


def normalize(input_low, input_high):
    input_low = tf.cast(input_low, tf.float32) / 255.0
    input_high = tf.cast(input_high, tf.float32) / 255.0
    return input_low, input_high


@tf.function
def load_image_train(input_low, input_high):

    # Image Augmentation layers : random flip
    if tf.random.uniform(()) > 0.5:
        input_low = tf.image.flip_left_right(input_low)
        input_high = tf.image.flip_left_right(input_high)
        input_low = tf.image.flip_up_down(input_low)
        input_high = tf.image.flip_up_down(input_high)

    # input_low, input_high = normalize(input_low, input_high)

    return input_low, input_high


def getTrainDataPipeline(high_img, low_img, SIZE):
    print("DataPipeline: getTrainDataPipeline")
    train_high_image, train_low_image = Utils.get_train_dataset(high_img, low_img, SIZE)

    train_data = tf.data.Dataset.from_tensor_slices((train_low_image, train_high_image))

    return train_data
