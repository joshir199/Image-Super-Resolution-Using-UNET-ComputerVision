import tensorflow as tf
from CustomDatasetClass import ImageDatasetClass

def normalize(input_low, input_high):
    input_low = tf.cast(input_low, tf.float32) / 255.0
    input_high = tf.cast(input_high, tf.float32) / 255.0
    return input_low, input_high


@tf.function
def load_image_train(input_low, input_high):
    # Preprocessing layer
    input_low = tf.image.resize(input_low, (256, 256, 3))
    input_high = tf.image.resize(input_high, (256, 256, 3))

    # Image Augmentation layers : random flip
    if tf.random.uniform(()) > 0.5:
        input_low = tf.image.flip_left_right(input_low)
        input_high = tf.image.flip_left_right(input_high)
        input_low = tf.image.flip_up_down(input_low)
        input_high = tf.image.flip_up_down(input_high)

    # Image augmentation layer : random crop

    input_low, input_mask = normalize(input_low, input_high)

    return input_low, input_mask


def getTrainDataPipeline():
    print("DataPipeline: getTrainDataPipeline")
    dataset_class = ImageDatasetClass()
    print("dataset_size: ", dataset_class.getsize())

    #train_data = tf.data.Dataset.from_tensor_slices((train_X))