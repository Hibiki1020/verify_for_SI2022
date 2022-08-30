from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import vgg_network_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import make_datalist_mod



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='../pyyaml/train_config.yaml',
        help='Training configuration file'
    )

    FLAGS, unparsed = parser.parse_known_args()

    print("Load YAML file")

    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    save_top_path = CFG["save_top_path"]
    yaml_path = save_top_path + "/train_config.yaml"
    shutil.copy(FLAGS.train_cfg, yaml_path)

    train_sequence = CFG["train"]
    valid_sequence = CFG["valid"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]

    multiGPU = int(CFG["multiGPU"])

    img_size = int(CFG["hyperparameters"]["img_size"])
    resize = int(CFG["hyperparameters"]["resize"])
    num_classes = int(CFG["hyperparameters"]["num_classes"])
    deg_threshold = float(CFG["hyperparameters"]["deg_threshold"])
    batch_size = int(CFG["hyperparameters"]["batch_size"])
    num_epochs = int(CFG["hyperparameters"]["num_epochs"])
    optimizer_name = str(CFG["hyperparameters"]["optimizer_name"])
    lr = float(CFG["hyperparameters"]["lr"])
    alpha = float(CFG["hyperparameters"]["alpha"])
    num_workers = int(CFG["hyperparameters"]["num_workers"])
    save_step = int(CFG["hyperparameters"]["save_step"])
    mean_element = float(CFG["hyperparameters"]["mean_element"])
    std_element = float(CFG["hyperparameters"]["std_element"])
    do_white_makeup = bool(CFG["hyperparameters"]["do_white_makeup"])
    do_white_makeup_from_back = bool(CFG["hyperparameters"]["do_white_makeup_from_back"])
    whiteup_frame = int(CFG["hyperparameters"]["whiteup_frame"])


    print("Train sequence: %s" % train_sequence)
    train_dataset = dataset_mod.ClassOriginalDataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequence, csv_name),
        transform = data_transform_mod.DataTransform(
            img_size,
            resize,
            mean_element,
            std_element,
        ),
        phase = "train",
        index_dict_path = index_csv_path,
        dim_fc_out = num_classes,
    )

    print("Valid sequence: %s" % valid_sequence)
    valid_dataset = dataset_mod.ClassOriginalDataset(
        data_list = make_datalist_mod.makeMultiDataList(valid_sequence, csv_name),
        transform = data_transform_mod.DataTransform(
            img_size,
            resize,
            mean_element,
            std_element,
        ),
        phase = "valid",
        index_dict_path = index_csv_path,
        dim_fc_out = num_classes,
    )