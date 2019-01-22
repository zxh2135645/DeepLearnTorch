import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class DatasetFetcher:
    def __init__(self):
        """
            A tool used to automatically download, check, split and get
            relevant information on the dataset
        """
        self.train_data = None
        self.test_data = None
        self.train_masks_data = None
        self.train_files = None
        self.test_files = None
        self.train_masks_files = None

    def fetch_dataset(self):
        """
        Fetches the dataset and return the input paths
        Args:
            This version does not have input

        Returns:
            list: [train_data, test_data, train_masks_data]

        """

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # print(script_dir)
        destination_path = os.path.join(script_dir, '../../input/')
        prefix = ""
        datasets_path = [destination_path + "TrainingData" + prefix, destination_path + "TestingData" + prefix,
                         destination_path + "TrainingLabel"]
        is_datasets_present = True

        # If the folders already exists then the files may already be extracted
        # This is a bit hacky but it's sufficient for our needs
        for dir_path in datasets_path:
            if not os.path.exists(dir_path):
                is_datasets_present = False

        if not is_datasets_present:
            raise FileExistsError('The folder does not exists, please check.')
        else:
            print("All datasets are present.")

        self.train_data = datasets_path[0]
        self.test_data = datasets_path[1]
        self.train_masks_data = datasets_path[2]
        self.train_files = sorted(os.listdir(self.train_data))
        self.test_files = sorted(os.listdir(self.test_data))
        self.train_masks_files = sorted(os.listdir(self.train_masks_data))
        return datasets_path

    def get_image_files(self, image_id, test_file=False, get_mask=False):
        if get_mask:
            if image_id + ".tif" in self.train_masks_files:
                return self.train_masks_data + "/" + image_id + ".tif"
            elif image_id + ".png" in self.train_masks_files:
                return self.train_masks_data + "/" + image_id + ".png"
            else:
                raise Exception("No mask with this ID found")
        elif test_file:
            if image_id + ".tif" in self.test_files:
                return self.test_data + "/" + image_id + ".tif"
        else:
            if image_id + ".tif" in self.train_files:
                return self.train_data + "/" + image_id + ".tif"
        raise Exception("No image with this ID found")

    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.uint8)

    def get_image_size(self, image):
        img = Image.open(image)
        return img.size

    def get_train_files(self, validation_size=0.2, sample_size=None):
        """

        Args:
            validation_size (float):
                 Value between 0 and 1
            sample_size (float, None):
                Value between 0 and 1 or None.
                Whether you want to have a sample of your dataset.

        Returns:
            list :
                Returns the dataset in the form:
                [train_data, train_masks_data, valid_data, valid_masks_data]
        """

        # Get a list of IDs
        train_ids = list(map(lambda img: img.split(".")[0], self.train_files))

        # Each id has 16 images but well...
        if sample_size:
            rnd = np.random.choice(train_ids, int(len(train_ids) * sample_size))
            train_ids = rnd.ravel()

        if validation_size:
            ids_train_split, ids_valid_split = train_test_split(train_ids, test_size=validation_size)
        else:
            ids_train_split = train_ids
            ids_valid_split = []

        train_ret = []
        train_masks_ret = []
        valid_ret = []
        valid_masks_ret = []

        for id in ids_train_split:
            train_ret.append(self.get_image_files(id))
            train_masks_ret.append(self.get_image_files(id, get_mask=True))

        for id in ids_valid_split:
            valid_ret.append(self.get_image_files(id))
            valid_masks_ret.append(self.get_image_files(id, get_mask=True))

        return [np.array(train_ret).ravel(), np.array(train_masks_ret).ravel(),
                np.array(valid_ret).ravel(), np.array(valid_masks_ret).ravel()]

    def get_test_files(self, sample_size):
        test_files = self.test_files

        if sample_size:
            rnd = np.random.choice(self.test_files, int(len(self.test_files) * sample_size))
            test_files = rnd.ravel()

        ret = [None] * len(test_files)
        for i, file in enumerate(test_files):
            ret[i] = self.test_data + "/" + file

        return np.array(ret)
