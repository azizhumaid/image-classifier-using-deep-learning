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



parser = argparse.ArgumentParser(description='Predict.py')


parser.add_argument('image', default='/home/workspace/ImageClassifier/flowers/train/1/image_06734.jpg', nargs='?', action="store", type = str)
parser.add_argument('checkpoint', default='checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('data_dir', type=str,  action="store", default="./flowers", help='directory')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
parser.add_argument('--names', dest="names", action="store", default='cat_to_name.json')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--arch', dest = "defaultarch", type=str, default='vgg16', help='available architecture: densenet, vgg')
args = parser.parse_args()

directory = args.data_dir
checkpoint = args.checkpoint
top_k = args.top_k
names = args.names
image = args.image
gpu = args.gpu



with open('cat_to_name.json', 'r') as json_file:
   cat_to_name = json.load(json_file)
model=terminal_training.load_checkpoint(checkpoint)
probabilities = terminal_training.predict(image, model, top_k)
labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])
i=0
while i < top_k:
   print("{} with a probability of {}".format(labels[i], probability[i]))
   i += 1
print("finished predicting")
     