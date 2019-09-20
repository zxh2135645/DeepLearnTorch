import nn.classifier
import nn.unet_origin as unet_origin
import nn.unet as unet
import torch.optim as optim
import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import img.augmentation as aug
from data.fetcher import DatasetFetcher
import nn.classifier
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback

import os
from multiprocessing import cpu_count

from data.dataset import TrainImageDataset, TestImageDataset
from img.augmentation import random_shift_scale_rotate
import multiprocessing

def main():
    # Clear log dir first
    helpers.clear_logs_folder()

    # Hyperparameters
    input_img_resize = (16, 16) # The resize size of the input images of the neural net
    output_img_resize = (16, 16) # The resize size of the output images of the neural net
    batch_size = 100
    epochs = 50 # 100
    threshold = 1.5  # mask is 1 background and 2 infarct
    validation_size = 0.1
    sample_size = None

    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Fetch the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.fetch_dataset()

    # Get the path to the files for the neural net
    # We don't want to split train/valid for KFold crossval
    X_train, y_train, X_valid, y_valid, z_train, z_valid= ds_fetcher.get_train_files(sample_size=sample_size,
                                                                    validation_size=validation_size)
    full_x_test = ds_fetcher.get_test_files(sample_size)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_fetcher.get_image_size(X_train[0])
    # The image kept its aspect ratio so we need to recalculate the img size for the nn
    # Seems not necessary
    # img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize)

    # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz_' + helpers.get_model_timestamp()))
    tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs_' + helpers.get_model_timestamp()))
    model_saver_cb = ModelSaverCallback(
        os.path.join(script_dir, '../output/models/model_' + helpers.get_model_timestamp()), verbose=True)

    # Testing callbacks
    pred_thresh = 0.5
    pred_saver_cb = PredictionsSaverCallback(os.path.join(script_dir, '../output/submit_'+ helpers.get_model_timestamp() + '.csv.gz'),
                                             origin_img_size, pred_thresh)

    # Define our neural net architecture
    net = unet.UNet1024((1, *input_img_resize))
    classifier = nn.classifier.InfarctClassifier(net, epochs)

    img_aug = random_shift_scale_rotate  # Image augmentation with shift, scaling and rotation
    train_ds = TrainImageDataset(X_train, y_train, z_train, input_img_resize, X_transform=img_aug)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, z_valid, input_img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    # Train the classifier
    classifier.train(train_loader, valid_loader, epochs, callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])

    test_ds = TestImageDataset(full_x_test, input_img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])
    pred_saver_cb.close_saver()


if __name__ == '__main__':
    # Trying majority voting
    main()
    torch.cuda.empty_cache()
