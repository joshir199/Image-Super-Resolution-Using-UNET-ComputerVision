from pprint import pprint
import tensorflow as tf
import warnings
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from data.CustomDatasetClass import CustomDatasetClass
from model.ImageResolutionModel import ImageResolutionModel
from utils.Utils import Utils

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    np.random.seed(33)

    pprint(" Start")
    print(tf.__version__)

    # Set constants
    SIZE = 256
    channels = 3
    param = [3, 16, 15, 0.001]  # kernel_size, batch_size, epoch, learning rate
    params = [param]

    # extract and prepare inputs
    zipped_path = './dataset.zip'
    unzipped_folder = './dataset'
    CustomDatasetClass.unzip_dataset(zipped_path, unzipped_folder)

    high_image_path = './dataset/dataset/Raw data/high'
    low_image_path = './dataset/dataset/Raw data/low'
    high_img = CustomDatasetClass.create_dataset(high_image_path, SIZE)
    low_img = CustomDatasetClass.create_dataset(low_image_path, SIZE)

    train_high_img, train_low_img = Utils.get_train_dataset(high_img, low_img, SIZE)
    validate_high_img, validate_low_img = Utils.get_validate_dataset(high_img, low_img, SIZE)

    client = MlflowClient()
    mlflow.set_experiments('image_super_resolution')

    for param in params:
        with mlflow.start_run() as run:

            mlflow.tensorflow.autolog()
            kernel_size = param[0]
            batch_size = param[1]
            epochs = param[2]
            learning_rate = param[3]

            mlflow.log_param("kernel_size", kernel_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)

            model_class = ImageResolutionModel(kernel_size, SIZE, channels)
            model = model_class.unet_model()

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)
            model.fit(train_low_img, train_high_img, batch_size=batch_size, epochs=epochs,
                      validation_data=(validate_low_img, validate_high_img))



    # check for training details in mlruns folder created by mlflow
