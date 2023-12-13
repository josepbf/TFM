import torch  
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler 

import numpy as np
import pandas as pd

import random
import time
from datetime import datetime

from EL_models import Model
from EL_dataset import PVDefectsDS, get_transform, collate_fn
from EL_train_utils import train_one_epoch
from EL_utils import int_to_boolean
from EL_optim import Optimizer
from EL_validation import evaluate_engine, loss_one_epoch_val, compute_confusion_matrix 
from EL_metrics import Writer

import argparse

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

parser = argparse.ArgumentParser(description="EL training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Reproducibility and device configuration
parser.add_argument("-D", "--deterministic", action='store_true', help="deterministic (slower)")
parser.add_argument("-W", "--num_workers", type=int, default=4, help="number of workers in data loader")
parser.add_argument("-P", "--pin_memory", action='store_true', help="pin memory in data loader")
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Saving
parser.add_argument("-E", "--save_model_epochs", type=int, default=0, help="save the model every this number of epochs")
parser.add_argument("-O", "--save_optim", action='store_true', help="save the optimizer state at the end")
#parser.add_argument("name", help="Name to be saved")

# Training param
parser.add_argument("-e", "--num_epochs", type=int, default=3, help="number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")

# Optimizer hyper-param
parser.add_argument("--optim_name", type=str, default='Adam')
parser.add_argument("--optim_default", type=int, default=0, help="0 is No default, 1 is default")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-r ", "--rho", type=float, default=0.9)
parser.add_argument("-w", "--weight_decay", type=float, default=0, help="weight decay (L2 regularization)")
parser.add_argument("-m ", "--momentum", type=float, default=0.9, help="momentum of the learning")
parser.add_argument("--dampening", type=float, default=0)
parser.add_argument("--nesterov", type=int, default=0, help="0 is off, 1 is on")

# Data augmentation
parser.add_argument("--gaussian_blur", type=int, default=0, help="gaussian_blur")
parser.add_argument("--color_jitter", type=int, default=0, help="color_jitter")
parser.add_argument("--horizontal_flip", type=int, default=0, help="horizontal_flip")
parser.add_argument("--vertical_flip", type=int, default=0, help="vertical_flip")
parser.add_argument("--adjust_sharpness", type=int, default=0, help="adjust_sharpness")
parser.add_argument("--random_gamma", type=int, default=0, help="random_gamma")
parser.add_argument("--gaussian_noise", type=int, default=0, help="gaussian_noise")
parser.add_argument("--random_erasing", type=int, default=0, help="random_erasing")

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
optim_default = int_to_boolean(config['optim_default'])
learning_rate = config['learning_rate']
rho = config['rho']
weight_decay = config['weight_decay']
momentum = config['momentum']
dampening = config['dampening']
nesterov = int_to_boolean(config['nesterov'])

# Hyper-parameters 
num_epochs = config['num_epochs']
batch_size = config['batch_size']
#label_smoothing = config['label_smoothing']

# Data Augmentation
gaussian_blur = int_to_boolean(config['gaussian_blur'])
color_jitter = int_to_boolean(config['color_jitter'])
horizontal_flip = int_to_boolean(config['horizontal_flip'])
vertical_flip = int_to_boolean(config['vertical_flip'])
adjust_sharpness = int_to_boolean(config['adjust_sharpness'])
random_gamma = int_to_boolean(config['random_gamma'])
gaussian_noise = int_to_boolean(config['gaussian_noise'])
random_erasing = int_to_boolean(config['random_erasing'])

# Model name
model_name = config['model_name']
trainable_backbone_layers = config['trainable_backbone_layers']
net_instance = Model(model_name=model_name, trainable_backbone_layers=trainable_backbone_layers)
net = net_instance.get_model()

net = net.to(device)
if seed != 0:
    torch.manual_seed(seed)

# Dataset
dataset_train = PVDefectsDS(get_transform(train=True), train_val_test = 0)
dataset_train_no_augmentation = PVDefectsDS(get_transform(train=False), train_val_test = 0)
dataset_validation = PVDefectsDS(get_transform(train=False), train_val_test = 1)
dataset_test = PVDefectsDS(get_transform(train=False), train_val_test = 2)

trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn = collate_fn)
trainloader_no_augmentation = DataLoader(dataset_train_no_augmentation, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)
validationloader = DataLoader(dataset_validation, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)
testloader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)

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

# Metrics
writer_training = Writer('',0, config)
writer_validation = Writer('',1, config)

# Training
epoch = 0
iteration = 0
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

while epoch != num_epochs:
    # train for one epoch, printing every 10 iterations

    print("Starting training num." + str(epoch))
    # train for one epoch, printing every 10 iterations
    iteration = len(trainloader)*epoch
    train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=1, iteration=iteration, writer = writer_training)
    loss_one_epoch_val(net, optimizer, validationloader, device, epoch, print_freq=1, iteration=iteration, writer = writer_validation)

    foldername_to_save_outputs = str("./runs/run_outputs_" + dt_string + "/epoch_" + str(epoch))
    
    if epoch == 0 or epoch == 1:
        print("Starting evaluation num. " + str(epoch))
        
        # evaluate on the train dataset
        print("Starting train data evaluation...")
        evaluate_engine(net,trainloader_no_augmentation,device, epoch, writer_training, foldername_to_save_outputs)
        
        # evaluate on the validation dataset
        print("Starting validation data evaluation...")
        evaluate_engine(net,validationloader,device, epoch, writer_validation, foldername_to_save_outputs)

        print("Computing confusion matrix training...")
        compute_confusion_matrix(epoch, writer_training, foldername_to_save_outputs, dataset_train_no_augmentation, iou_threshold = 0.5)

        print("Computing confusion matrix validation...")
        compute_confusion_matrix(epoch, writer_validation, foldername_to_save_outputs, dataset_validation, iou_threshold = 0.5)

        print("Saving the model...")
        #net_instance.save_model(net, epoch)
        
    epoch += 1
        
print("That's it!")