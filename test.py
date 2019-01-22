import os
from data.fetcher import DatasetFetcher
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

if __name__ == '__main__':
    print('!!')
    validation_size = 0.2
    sample_size = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    destination_path = os.path.join(script_dir, '../../../MATLAB/CNNTrainingCrop64/')
    prefix = ""
    datasets_path = [destination_path + "TrainingData" + prefix, destination_path + "TestingData" + prefix,
                     destination_path + "CompositeLabel2"]
    train_data = datasets_path[0]
    test_data = datasets_path[1]
    train_masks_data = datasets_path[2]
    train_files = sorted(os.listdir(train_data))
    test_files = sorted(os.listdir(test_data))
    train_masks_files = sorted(os.listdir(train_masks_data))

    train_ids = list(map(lambda img: img.split(".")[0], train_files))

    if sample_size:
        rnd = np.random.choice(train_ids, int(len(train_ids) * sample_size))
        train_ids = rnd.ravel()
    if validation_size:
        ids_train_split, ids_valid_split = train_test_split(train_ids, test_size=validation_size)
    else:
        ids_train_split = train_ids
        ids_valid_split = []

    def get_image_files(image_id, test_file=False, get_mask=False):
        if get_mask:
            if image_id + ".tif" in train_masks_files:
                return train_masks_data + "/" + image_id + ".tif"
            elif image_id + ".png" in train_masks_files:
                return train_masks_data + "/" + image_id + ".png"
            else:
                raise Exception("No mask with this ID found")
        elif test_file:
            if image_id + ".tif" in test_files:
                return test_data + "/" + image_id + ".tif"
        else:
            if image_id + ".tif" in train_files:
                return train_data + "/" + image_id + ".tif"
        raise Exception("No image with this ID found")


    train_ret = []
    train_masks_ret = []
    valid_ret = []
    valid_masks_ret = []

    for id in ids_train_split:
        train_ret.append(get_image_files(id))
        train_masks_ret.append(get_image_files(id, get_mask=True))

    for id in ids_valid_split:
        valid_ret.append(get_image_files(id))
        valid_masks_ret.append(get_image_files(id, get_mask=True))

    img = Image.open(train_ret[0])
    print(img.size)