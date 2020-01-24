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
from PIL import Image


arch = {"vgg16":25088,
        "densenet121":1024
        }

def image_datasets():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    valid_transforms = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    test_transforms = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform= training_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform= valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform= test_transforms)
    

    return  train_datasets, valid_datasets, test_datasets
    
    
def data_loders(path):
    train_datasets, valid_datasets, test_datasets = image_datasets()
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders

    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    return train_loaders, valid_loaders, test_loaders


def create_model(structure='vgg16',dropout=0.5, hidden_layer = 512, learing_rate = 0.001):

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Please use only vgg16 or densenet121 ")


    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(arch[structure], hidden_layer),
                      nn.ReLU(),
                      nn.Dropout(p=dropout),
                      nn.Linear(hidden_layer, 102),
                      nn.LogSoftmax(dim=1))

    model.classifier = classifier
    model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learing_rate)
       
    model.cuda()
    model = model.to('cuda')
    
    return model, criterion, optimizer

def start_training(model, criterion, optimizer, epochs = 8, print_every=5, train_loaders=0,valid_loaders=0):
    steps = 0
    running_loss = 0
    for e in range(epochs):
        for ii, (images, labels) in enumerate(train_loaders):
          images= images.cuda()
          labels = labels.cuda()
          steps+=1
          log_ps = model(images)
          loss = criterion(log_ps, labels)
        
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        
          running_loss += loss.item()
          if steps % print_every == 0:

            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for ii, (images, labels) in enumerate(valid_loaders):
                    images, labels = images.cuda(), labels.cuda()
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_loaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(valid_loaders):.3f}")
            running_loss = 0
            model.train()

            
def save_Checkpoint(model,save_directory,arch,train_datasets,epochs,learning_rate,dropout, hidden_layer):
    model.class_to_idx = train_datasets.class_to_idx
    torch.save({'structure' :arch,
                'hidden_layer':hidden_layer,
                'dropout':dropout,
                'learning_rate':learning_rate,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_directory)
    
def load_checkpoint(filepath ='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(filepath)
    structure = checkpoint['structure']
    hidden_layer = checkpoint['hidden_layer']
    dropout = checkpoint['dropout']
    learning_rate=checkpoint['learning_rate']
    model = models.vgg16(pretrained=True)
    
    model,_,_ = create_model(structure,dropout, hidden_layer, learning_rate)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    
    return model

def process_image(image_path= '/home/workspace/ImageClassifier/flowers/train/1/image_06734.jpg'):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image_path)
   
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil)
    
    return img_tensor



def predict(image_path = '/home/workspace/ImageClassifier/flowers/train/1/image_06734.jpg', model=0, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    model = model.to('cuda')
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    
    
    
    with torch.no_grad():
        image.cuda()
        image = image.to('cuda')
        output = model.forward(image)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
