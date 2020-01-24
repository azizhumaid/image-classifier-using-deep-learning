import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter
import argparse
import json
import terminal_training

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--arch', dest = "arch", type=str, default='vgg16', help='available architecture: densenet, vgg', required=True)
parser.add_argument('--learning_rate', dest = "learing_rate", type=float, default=0.001, help='learning rate')    
parser.add_argument('--hidden_layer', dest = "hidden_layer", type=int, default=512, help='hidden_layer')
parser.add_argument('--epochs', dest = "epochs", type=int, default=8, help='number of epochs')
parser.add_argument('data_dir', type=str,  action="store", default="./flowers", help='directory')
parser.add_argument('--save_dir' , dest = "save_directory", type=str, default="checkpoint.pth", help='saved model')
parser.add_argument('--dropout' , dest = "dropout", type=str, action = "store", default = 0.5)

args = parser.parse_args()

directory = args.data_dir
save_dir = args.save_directory
learning_rate = args.learing_rate
structure = args.arch
dropout = args.dropout
hidden_layer = args.hidden_layer
gpu = args.gpu
epochs = args.epochs



train_loaders, valid_loaders, test_loaders = terminal_training.data_loders(directory)
train_datasets, valid_datasets, test_datasets = terminal_training.image_datasets()
model, optimizer, criterion = terminal_training.create_model(structure,dropout,hidden_layer,learning_rate)
terminal_training.start_training(model, optimizer, criterion, epochs, 8, train_loaders,valid_loaders)
terminal_training.save_Checkpoint(model,save_dir, structure,train_datasets ,epochs ,learning_rate,dropout,hidden_layer)
print("Finished Training!")

