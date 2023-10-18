import os
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt


# High level Singleton class containing static methods which
# can be accessed using just single instance of the class
class Utils:

    @staticmethod
    def showImageData(high_res_imagelist, low_res_imagelist):
        """
        :param high_res_imagelist: data which contains high resolution image
        :param low_res_imagelist: data which contains low resolution image
        :return: nothing. Just plot image and save it
        """

        img_list_size = len(high_res_imagelist)
        for i in range(2):
            random_index = np.random.randint(0, img_list_size)
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.title('High Resolution Image', color='green', font='20')
            plt.imshow(high_res_imagelist[random_index])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title('Low Resolution Image', color='green', font='20')
            plt.imshow(low_res_imagelist[random_index])
            plt.axis('off')
            path = 'images/input_image_' + random_index + '.png'
            plt.savefig(path)

    @staticmethod
    def dataset_split_index(high_img_dataset):
        """
        :param high_img_dataset: image data list
        :return: split index of train, test and validation data
        """
        img_list_size = len(high_img_dataset)
        train_idx = int(img_list_size * 0.8)
        test_idx = int(img_list_size * 0.1)
        validate_idx = int(img_list_size * 0.1)
        return train_idx, test_idx, validate_idx

    @staticmethod
    def get_train_dataset(high_img_list, low_img_list, SIZE):
        """
        :param high_img_list: high resolution image list
        :param low_img_list: low resolution image list
        :param SIZE: image Size
        :return: train dataset
        """
        train_idx, test_idx, val_idx = Utils.dataset_split_index(high_img_list)
        train_high_image = high_img_list[:train_idx]
        train_low_image = low_img_list[:train_idx]
        train_high_image = np.reshape(train_high_image, (len(train_high_image), SIZE, SIZE, 3))
        train_low_image = np.reshape(train_low_image, (len(train_low_image), SIZE, SIZE, 3))

        return train_high_image, train_low_image

    @staticmethod
    def get_test_dataset(high_img_list, low_img_list, SIZE):
        """
        :param high_img_list: high resolution image list
        :param low_img_list: low resolution image list
        :param SIZE: image Size
        :return: test dataset needed after training
        """
        train_idx, test_idx, val_idx = Utils.dataset_split_index(high_img_list)
        test_high_image = high_img_list[train_idx: train_idx + test_idx]
        test_low_image = low_img_list[train_idx: train_idx + test_idx]
        test_high_image = np.reshape(test_high_image, (len(test_high_image), SIZE, SIZE, 3))
        test_low_image = np.reshape(test_low_image, (len(test_low_image), SIZE, SIZE, 3))

        return test_high_image, test_low_image

    @staticmethod
    def get_validate_dataset(high_img_list, low_img_list, SIZE):
        """
        :param high_img_list: high resolution image list
        :param low_img_list: low resolution image list
        :param SIZE: image Size
        :return: validation dataset needed during training
        """
        train_idx, test_idx, val_idx = Utils.dataset_split_index(high_img_list)
        validate_high_image = high_img_list[train_idx + test_idx:]
        validate_low_image = low_img_list[train_idx + test_idx:]
        validate_high_image = np.reshape(validate_high_image, (len(validate_high_image), SIZE, SIZE, 3))
        validate_low_image = np.reshape(validate_low_image, (len(validate_low_image), SIZE, SIZE, 3))

        return validate_high_image, validate_low_image
