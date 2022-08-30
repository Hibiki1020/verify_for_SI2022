
import sys, codecs
from tkinter import W
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
sys.dont_write_bytecode = True

from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random

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
import torch.nn.functional as nn_functional
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from common import dataset_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import network_mod

class Trainer:
    def __init__(self,
        save_top_path,
        multiGPU,
        img_size,
        mean_element,
        std_element,
        batch_size,
        num_epochs,
        optimizer_name,
        loss_function,
        lr,
        alpha,
        net_radian,
        train_dataset,
        valid_dataset,
        num_workers,
        save_step):

        self.save_top_path = save_top_path
        self.multiGPU = multiGPU
        self.img_size = img_size
        self.mean_element = mean_element
        self.std_element = std_element
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name
        self.criterion_name = loss_function
        self.loss_function = loss_function
        self.lr = lr
        self.alpha = alpha
        self.net_radian = net_radian
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_workers = num_workers
        self.save_step = save_step


        if self.multiGPU == 0:
                self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.setRandomCondition()
        self.dataloaders_dict = self.getDataloaders(self.train_dataset, self.valid_dataset, batch_size)

        self.net_radian = self.getNetwork(self.net_radian)

        self.optimizer_rad = self.getOptimizer(self.optimizer_name, self.lr, self.net_radian)
        self.criterion = self.getCriterion(self.criterion_name)

    def setRandomCondition(self, keep_reproducibility=False, seed=123456789):
        if keep_reproducibility:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataloaders(self, train_dataset, valid_dataset, batch_size):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = self.num_workers,
            #pin_memory =True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = self.num_workers,
            #pin_memory = True
        )

        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}

        return dataloaders_dict

    def getNetwork(self, net):
        print("Loading Network")
        net = net.to(self.device)

        if self.multiGPU == 1 and self.device == "cuda":
            net = nn.DataParallel(net)
            cudnn.benchmark = True
            print("Training on multiGPU Device")
        else:
            cudnn.benchmark = True
            print("Training on Single GPU Device")

        return net

    def getOptimizer(self, optimizer_name, lr, net):

        if optimizer_name == "SGD":
            optimizer = optim.SGD(net.parameters() ,lr = lr, momentum=0.9, 
            weight_decay=0.0)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=0.0)
        elif optimizer_name == "RAdam":
            optimizer = optim.RAdam(net.parameters(), lr = lr, weight_decay=0.0)

        print("optimizer: {}".format(optimizer_name))
        return optimizer

    def getCriterion(self, criterion_name):
        if criterion_name == "MSE":
            criterion = nn.MSELoss()
        elif criterion_name == "L1":
            criterion = nn.L1Loss()
        elif criterion_name == "SmoothL1":
            criterion = nn.SmoothL1Loss()
        elif criterion_name == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif criterion_name == "BCEWithLogits":
            criterion = nn.BCEWithLogitsLoss()
        elif criterion_name == "BCE":
            criterion = nn.BCELoss()
        elif criterion_name == "CrossEntropy2d":
            criterion = nn.CrossEntropyLoss2d()
        elif criterion_name == "NLLLoss2d":
            criterion = nn.NLLLoss2d()
        elif criterion_name == "NLL":
            criterion = nn.NLLLoss()
        elif criterion_name == "KLDiv":
            criterion = nn.KLDivLoss()
        elif criterion_name == "PoissonNLLLoss":
            criterion = nn.PoissonNLLLoss()
        elif criterion_name == "L1Loss":
            criterion = nn.L1Loss()
        elif criterion_name == "MSELoss":
            criterion = nn.MSELoss()
        elif criterion_name == "SmoothL1Loss":
            criterion = nn.SmoothL1Loss()
        elif criterion_name == "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss()
        elif criterion_name == "MultiLabelMarginLoss":
            criterion = nn.MultiLabelMarginLoss()
        elif criterion_name == "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss()
        elif criterion_name == "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss()

        return criterion

    def process(self):
        start_clock = time.time()
        
        #Loss recorder
        writer = SummaryWriter(log_dir = self.save_top_path + "/log")

        record_train_loss_deg = []
        record_valid_loss_deg = []

        record_train_loss_rad = []
        record_valid_loss_rad = []

        for epoch in range(self.num_epochs):
            print("--------------------------------")
            print("Epoch: {}/{}".format(epoch+1, self.num_epochs))

            for phase in ["train", "valid"]:
                if phase == "train":
                    self.net_degree.train()
                    self.net_radian.train()
                elif phase == "valid":
                    self.net_degree.eval()
                    self.net_radian.eval()
                
                #Data Load
                epoch_loss_degree = 0.0
                epoch_loss_radian = 0.0

                for img_rad, label_rad in tqdm(self.dataloaders_dict[phase]):
                    self.optimizer_rad.zero_grad()

                    img_rad = img_rad.to(self.device)
                    label_rad = label_rad.to(self.device)

                    with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                        outputs_rad = self.net_radian(img_rad)
                        loss_rad = self.computeLoss(outputs_rad, label_rad)

                        if self.device == 'cpu':
                            l2norm_rad = torch.tensor(0., requires_grad = True).cpu()
                        else:
                            l2norm_rad = torch.tensor(0., requires_grad = True).cuda()

                        for w in self.net_radian.parameters():
                            l2norm_rad = l2norm_rad + torch.norm(w)**2

                        loss_rad = loss_rad + self.alpha*l2norm_rad

                        if phase == "train":
                            loss_rad.backward()
                            self.optimizer_rad.step()

                        ## add loss
                        epoch_loss_radian += loss_rad.item() * label_rad.size(0)

                ## average loss

                epoch_loss_radian = epoch_loss_radian / len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss_radian))

                if(epoch%self.save_step == 0 and epoch > 0 and epoch != self.num_epochs and phase == "valid"):
                    self.saveWeight_Interval(epoch)

                ## record
                if phase == "train":
                    record_train_loss_rad.append(epoch_loss_radian)
                    writer.add_scalar("Loss/train", epoch_loss_radian, epoch)
                    
                else:
                    record_valid_loss_rad.append(epoch_loss_radian)
                    writer.add_scalar("Loss/train", epoch_loss_radian, epoch)

            if record_train_loss_rad and record_valid_loss_rad:
                writer.add_scalars("Loss/train_and_val", {"train": record_train_loss_rad[-1], "valid": record_valid_loss_rad[-1]}, epoch)

        writer.close()
        self.saveParam()

        self.saveGraph(record_train_loss_rad, record_valid_loss_rad)


    def computeLoss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss

    def saveWeight_Interval(self, epoch):
        save_path_rad = self.save_top_path + "/weights" + "_" + str(epoch) + "_rad.pth"
        torch.save(self.net_radian.state_dict(), save_path_rad)
        print("Saved Weight in Epoch: {}".format(epoch))

    def saveParam(self):
        save_path_rad = self.save_top_path + "/param_rad.pth"
        torch.save(self.net_radian.state_dict(), save_path_rad)
        print("Saved Param")

    def saveGraph(self, record_loss_train, record_loss_val):
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig(self.save_top_path + "/train_log.jpg")
        plt.show()

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

    multiGPU = int(CFG["multiGPU"])

    img_size = int(CFG["hyperparameters"]["img_size"])
    resize = int(CFG["hyperparameters"]["resize"])
    batch_size = int(CFG["hyperparameters"]["batch_size"])
    num_epochs = int(CFG["hyperparameters"]["num_epochs"])
    optimizer_name = str(CFG["hyperparameters"]["optimizer_name"])
    loss_function = str(CFG["hyperparameters"]["loss_function"])
    lr = float(CFG["hyperparameters"]["lr"])
    alpha = float(CFG["hyperparameters"]["alpha"])
    num_workers = int(CFG["hyperparameters"]["num_workers"])
    save_step = int(CFG["hyperparameters"]["save_step"])
    mean_element = float(CFG["hyperparameters"]["mean_element"])
    std_element = float(CFG["hyperparameters"]["std_element"])
    dropout_rate = float(CFG["hyperparameters"]["dropout_rate"])

    print("Load Train Dataset")

    train_dataset = dataset_mod.Originaldataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequence, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element
        ),
        phase = "train",
    )

    print("Load Valid Dataset")

    valid_dataset = dataset_mod.Originaldataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequence, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element
        ),
        phase = "train",
    )

    print("Load Network")
    net_radian = network_mod.Network(resize, dropout_rate)

    trainer = Trainer(
        save_top_path,
        multiGPU,
        img_size,
        mean_element,
        std_element,
        batch_size,
        num_epochs,
        optimizer_name,
        loss_function,
        lr,
        alpha,
        net_radian,
        train_dataset,
        valid_dataset,
        num_workers,
        save_step
    )

    trainer.process()

