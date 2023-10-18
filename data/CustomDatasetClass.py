import os
import re
import zipfile
import cv2 as cv2
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt


class ImageDatasetClass:

    @staticmethod
    def unzip_dataset(zip_path, unzip_folder_path):
        """
        :param zip_path: path of zipped(.zip) dataset
        :param unzip_folder_path: folder path to store dataset after unzipping
        :return:
        """

        with zipfile.ZipFile(zip_path, 'r') as zipper:
            zipper.extractall(unzip_folder_path)

    @staticmethod
    def sort_filenames(filenames):
        """
        :param filenames: name of each image files to be sorted
        :return: sorted filenames
        """

        convert_filename = lambda filename: int(filename) if filename.isdigit() else filename.lower()
        alphanum_key = lambda key: [convert_filename(c) for c in re.split('([0-9]+)', key)]
        return sorted(filenames, key=alphanum_key)

    @staticmethod
    def create_dataset(dataset_path, SIZE):
        """
        :param dataset_path: relative path of unzipped dataset folder
        :param SIZE: size of Input images
        :return: image dataset list
        """

        img_data_list = []
        files = os.listdir(dataset_path)
        files = ImageDatasetClass.sort_filenames(files)

        for index in tqdm(files):
            img = cv2.imread(dataset_path + '/' + index, 1)
            # OpenCV reads image in BGR format, so convert it into RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image to proper dimension and normalize it
            img = cv2.resize(img, (SIZE,SIZE))

            img = img.astype('float32')/255.0
            img_data_list.append(tf.keras.preprocessing.image.img_to_array(img))

        return img_data_list

