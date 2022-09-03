import sys, codecs

from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random
import csv
import cv2
import PIL.Image as PILIMAGE

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
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from common import dataset_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import network_mod

from collections import OrderedDict

class FrameInfer:
    def __init__(self, CFG):
        super(FrameInfer, self).__init__()
        self.CFG = CFG

        self.infer_dataset = self.CFG["infer_dataset"]
        self.csv_name = self.CFG["csv_name"]

        self.radian_weights_top_directory = self.CFG["radian_weights_top_directory"]
        self.radian_weights_name = self.CFG["radian_weights_file_name"]
        self.radian_weights_path = os.path.join(self.radian_weights_top_directory, self.radian_weights_name)


        self.infer_log_top_directory = self.CFG["infer_log_top_directory"]
        self.radian_infer_log_file_name = self.CFG["radian_infer_log_file_name"]

        self.resize = int(self.CFG["resize"])
        self.mean_element = float(self.CFG["mean_element"])
        self.std_element = float(self.CFG["std_element"])
        self.dropout_rate = float(self.CFG["dropout_rate"])

        self.transform = data_transform_mod.DataTransform(
            self.resize, self.mean_element, self.std_element
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.net_radian = self.getNetwork(self.radian_weights_path)

        print("Load Test Dataset")

        self.test_dataset = dataset_mod.Originaldataset(
            data_list = make_datalist_mod.makeMultiDataList(self.infer_dataset, self.csv_name),
            transform = data_transform_mod.DataTransform(
                self.resize,
                self.mean_element,
                self.std_element
            ),
            phase = "train",
        )

    def getNetwork(self, weights_path):
        net = network_mod.Network(self.resize, self.dropout_rate, use_pretrained_vgg=False)
        net.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        net.to(self.device)
        net.eval()

        #load weights trained on multiple GPU device
        if torch.cuda.is_available:
            state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            state_dict = new_state_dict
            print("Load .pth file")
        else:
            state_dict = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Load to CPU")

        net.load_state_dict(state_dict, strict=False)
        return net

    def spin(self):
        result_csv_degree = [] 
        result_csv_radian = []

        infer_count = 0

        diff_total_roll = 0.0
        diff_total_pitch = 0.0

        for img_rad, label_rad in self.test_dataset:
            start_clock = time.time()

            img_rad = img_rad.unsqueeze(dim=0)
            img_rad = img_rad.to(self.device)

            rad_result = self.net_radian(img_rad).to('cpu').detach().numpy().copy()
            rad_roll = rad_result[0][0]/3.141592*180.0
            rad_pitch = rad_result[0][1]/3.141592*180.0

            rad_diff_roll = abs(rad_roll - label_rad[0]).to('cpu').detach().numpy().copy()
            rad_diff_pitch = abs(rad_pitch - label_rad[1]).to('cpu').detach().numpy().copy()

            diff_total_roll += rad_diff_roll
            diff_total_pitch += rad_diff_pitch

            label_rad = label_rad.to('cpu').detach().numpy().copy()

            #print(label_deg, label_rad)

            print("Infer Count: ", infer_count)
            print("--------------------Rad--------------------")
            print("Infered Roll  :" + str(rad_roll) +  "[rad]")
            print("GT Roll       :" + str(label_rad[0]) + "[rad]")
            print("Diff Roll     :" + str(rad_diff_roll) + "[rad]")
            print("Infered Pitch :" + str(rad_pitch) + "[rad]")
            print("GT Pitch      :" + str(label_rad[1]) + "[rad]")
            print("Diff Pitch    :" + str(rad_diff_pitch) + "[rad]")
            print("-------------------------------------------")

            tmp_result_rad = [infer_count, rad_roll, rad_pitch, label_rad[0], label_rad[1], rad_diff_roll, rad_diff_pitch]

            result_csv_radian.append(tmp_result_rad)

            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")
            print("\n\n")

            infer_count += 1

        print("Inference Test Has Done....")
        print("Average of Error of Roll : " + str(diff_total_roll/float(infer_count)) + " [deg]")
        print("Average of Error of Pitch: " + str(diff_total_pitch/float(infer_count)) + " [deg]")

        return result_csv_radian

    def save_csv(self, result_csv_radian):
        result_csv_path_radian = os.path.join(self.infer_log_top_directory, self.radian_infer_log_file_name)

        csv_file = open(result_csv_path_radian, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv_radian:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data in Radian")
        print(result_csv_path_radian)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./frame_infer.py")

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../pyyaml/infer_config.yaml',
        help='Frame Infer config file'
    )

    FLAGS, unparsed = parser.parse_known_args()

    try:
        print("Opening Infer config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Failed to open config file %s", FLAGS.config)
        quit()

    frame_infer = FrameInfer(CFG)
    result_csv_radian = frame_infer.spin()
    frame_infer.save_csv(result_csv_radian)