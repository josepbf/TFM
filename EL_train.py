import pycocotools
print(pycocotools.__version__)

import torch  
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler 

import numpy as np
import pandas as pd

import random
import time

from EL_models import Model
from EL_dataset import PVDefectsDStrain, get_transform, collate_fn
from EL_train_utils import train_one_epoch
from EL_utils import compute_confusion_matrix_training
from EL_optim import Optimizer
from EL_validation import evaluate_engine

import argparse

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

parser = argparse.ArgumentParser(description="EL training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Reproducibility and device configuration
parser.add_argument("-D", "--deterministic", action='store_true', help="deterministic (slower)")
parser.add_argument("-W", "--num_workers", type=int, default=1, help="number of workers in data loader")
parser.add_argument("-P", "--pin_memory", action='store_true', help="pin memory in data loader")
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Saving
parser.add_argument("-E", "--save_model_epochs", type=int, default=0, help="save the model every this number of epochs")
parser.add_argument("-O", "--save_optim", action='store_true', help="save the optimizer state at the end")
#parser.add_argument("name", help="Name to be saved")

# Training param
parser.add_argument("-e", "--num_epochs", type=int, default=1, help="number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")

# Optimizer hyper-param
parser.add_argument("--optim_name", type=str, default='Adam')
parser.add_argument("--optim_default", type=int, default=0, help="0 is No default, 1 is default")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-r ", "--rho", type=float, default=0.9)
parser.add_argument("-w", "--weight_decay", type=float, default=0, help="weight decay (L2 regularization)")
parser.add_argument("-m ", "--momentum", type=float, default=0.9, help="momentum of the learning")
parser.add_argument("--dampening", type=float, default=0)
parser.add_argument("--nesterov", type=int, default=0, help="0 is off, 1 is on")

# Model
parser.add_argument("--model_name", type=str, default='FasterRCNN_ResNet-50-FPN')
parser.add_argument("--trainable_backbone_layers", type=int, default=3)

args = parser.parse_args()
config = vars(args)
print(config)

#%%
# Reproducibility
seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if config['deterministic']:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Device configuration
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ' + str(device))

num_workers = config['num_workers']
pin_memory = config['pin_memory']

# Optimizer hyper-parameters 
optim_name = config['optim_name']
optim_default = config['optim_default']
if optim_default == 0:
    optim_default = False
if optim_default == 1:
    optim_default = True
learning_rate = config['learning_rate']
rho = config['rho']
weight_decay = config['weight_decay']
momentum = config['momentum']
dampening = config['dampening']
nesterov = config['nesterov']
if nesterov == 0:
    nesterov = False
if nesterov == 1:
    nesterov = True

# Hyper-parameters 
num_epochs = config['num_epochs']
batch_size = config['batch_size']
#label_smoothing = config['label_smoothing']

# Model name
model_name = config['model_name']
trainable_backbone_layers = config['trainable_backbone_layers']
net_instance = Model(model_name=model_name, trainable_backbone_layers=trainable_backbone_layers)
net = net_instance.get_model()

net = net.to(device)
if seed != 0:
    torch.manual_seed(seed)

# Create the custom dataset
dataset_train = PVDefectsDStrain(get_transform(train=True)) #TODO train, validation, calibration, tests

# Create a DataLoader for batching and shuffling the data
trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn = collate_fn)

# Loss function 
net_params = [p for p in net.parameters() if p.requires_grad]
optimizer_instance = Optimizer(net_params,
                optim_name,
                optim_default,
                learning_rate,
                rho,
                weight_decay, # L2 penalty
                momentum,
                dampening,
                nesterov)   
optimizer = optimizer_instance.get_optim()

# Loss function
criterion = nn.CrossEntropyLoss() #TODO generalize

# Training
epoch = 0
iteration = 0
while epoch != epoch+num_epochs:
    # train for one epoch, printing every 10 iterations

    print("Starting training num." + str(epoch))
    # train for one epoch, printing every 10 iterations
    iteration = len(trainloader)*epoch
    train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=1, iteration=iteration)
    
    if epoch % 10 == 0 or epoch == 0:
        print("Starting evaluation num. " + str(epoch))
        
        # evaluate on the train dataset
        print("Starting train data evaluation...")
        evaluate_engine(net,trainloader,device, epoch, isValData = False)
        
        # evaluate on the test dataset
        #print("Starting validation data evaluation...")
        #evaluate_engine(model,validation_data_loader,device, epoch, isValData = True)

        print("Computing confusion matrix training...")
        #compute_confusion_matrix_training(epoch, iou_threshold = 0.5)

        print("Computing confusion matrix validation...")
        #compute_confusion_matrix_validation(epoch, iou_threshold = 0.5)

        print("Saving the model...")
        #model_name = str("./models_faster_1/model_epoch_" + str(epoch) + ".pth")
        #torch.save(model, model_name)
    epoch += 1
        
print("That's it!")