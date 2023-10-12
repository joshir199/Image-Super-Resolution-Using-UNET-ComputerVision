import os
import tqdm
import numpy
from sklearn.model_selection import train_test_split


class ImageDatasetClass():

    def __init__(self):

        # path where image datasets are stored
        DATA_PATH_X = '/archive/dataset/RawData/low_res'
        DATA_PATH_Y = '/archive/dataset/RawData/high_res'

        self.image_high_files = []
        self.image_low_files = []

        # Fetching Input image data path into list
        for root, dirs, files in os.walk(DATA_PATH_X):
            for file in tqdm(files):
                img_path = os.path.join(DATA_PATH_X, file)
                mask_path = os.path.join(DATA_PATH_Y, file)
                self.image_low_files.append(img_path)
                self.image_high_files.append(mask_path)


        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            self.image_low_files,
            self.image_high_files,
            test_size=0.1,
            random_state=33)

    def getsize(self):
        return len(self.image_low_files)

    def getRandomImage(self):
        num = numpy.random.randint(0, 500, 1)
        return self.image_high_files[num], self.image_low_files[num]

    def getTrainImageList(self):
        return self.train_X, self.train_Y

    def gettestImageList(self):
        return self.test_X, self.test_Y

