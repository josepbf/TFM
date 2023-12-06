import torch 
from torch import Tensor
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.transforms import GaussianBlur
from torchvision.transforms import RandomEqualize
from torchvision.transforms import ColorJitter
from torchvision.transforms import RandomAdjustSharpness
from torchvision.transforms import RandomErasing
import scipy.io

import PIL
from PIL import Image

import numpy as np
import pandas

import math
import random

import os

from ast import literal_eval as make_tuple

# TODO: Check this: https://pytorch.org/vision/main/transforms.html

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class OurRandomErasing(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    erasing = RandomErasing(self.prob)
    image = erasing(image)
    return image, target

class OurGaussianNoise(object):
    def __init__(self, prob):
        self.minstd = 0.0
        self.maxstd = 0.1
        self.mean = 0
        self.prob = prob
    def __call__(self, image, target):
      if random.random() < self.prob:
        std = np.random.uniform(self.minstd, self.maxstd)
        image = image + torch.randn(image.size) * std + self.mean
      return image, target

class OurRandomGamma(object):
    def __init__(self, prob):
        self.prob = prob
        mingamma = 2/3
        maxgamma = 3/2
        self.minloggamma = np.log(mingamma)
        self.maxloggamma = np.log(maxgamma)
    def __call__(self, image, target):
      if random.random() < self.prob:
        gamma = np.exp(np.random.uniform(self.minloggamma, self.maxloggamma))
        image = TF.adjust_gamma(image, gamma=gamma)
      return image, target

class OurColorJitter(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            jitter = ColorJitter(brightness=0.25, contrast=0.25)
            image = jitter(image)

        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    # 0: xmin, 1: ymin, 2: xmax, 3: ymax
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]

        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

        return image, target

class OurGaussianBlur(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            blurfilter = GaussianBlur(kernel_size=5, sigma=(0.01,1.0))
            image = blurfilter(image)
        return image, target

class OurRandomAdjustSharpness(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        randomadjustsharpnessfilter = RandomAdjustSharpness(sharpness_factor=1.5, p=self.prob)
        image = randomadjustsharpnessfilter(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_transform(train):
    transforms = []
    if train:
      transforms.append(OurGaussianBlur(0.125))
      transforms.append(OurColorJitter(0.125))
      transforms.append(OurRandomAdjustSharpness(0.5))
      transforms.append(OurRandomGamma(0.5))
      #transforms.append(OurGaussianNoise(0.5))
      #transforms.append(OurRandomErasing(0.5))
    transforms.append(ToTensor())
    if train:
      transforms.append(RandomHorizontalFlip(0.5))
      transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

class PVDefectsDS(torch.utils.data.Dataset):
    def __init__(self, transforms, train_val_test):
        self.transforms = transforms
        self.labels = list()
        self.imgs_names = list()
        self.train_val_test = train_val_test
        
        self.dataset_path = "/zhome/de/6/201864/Downloads/PVDefectsDS/"
        annotationsPath = self.dataset_path + "namesDSVersionH_CorrAug2023.xlsx"
        
        self.masks_path = self.dataset_path + 'MasksVersionH-CorrAugust2023/'
        print("Loading dataset...")

        for i in range(12):
          if i%2 == 0:
            self.labels.append(pandas.read_excel(annotationsPath,i)[:-2])
          else:
            self.imgs_names.append(pandas.read_excel(annotationsPath,i)[:-2])
        
        self.imgs_names = pandas.concat(self.imgs_names, ignore_index=True)
        self.labels = pandas.concat(self.labels, ignore_index=True)

        split_train_val_test_csv = pandas.read_csv(self.dataset_path + 'split_train_val_test.csv')

        self.split_imgs_name = []
        if train_val_test == 0:
          # Train
          for i in range(len(split_train_val_test_csv)):
            if split_train_val_test_csv.at[i,'train_val_test_split'] == 0:
              self.split_imgs_name.append(split_train_val_test_csv.at[i,"name_image"])
        elif train_val_test == 1:
          # Validation
          for i in range(len(split_train_val_test_csv)):
            if split_train_val_test_csv.at[i,'train_val_test_split'] == 1:
              self.split_imgs_name.append(split_train_val_test_csv.at[i,"name_image"])
        elif train_val_test == 2:
          # Test
          for i in range(len(split_train_val_test_csv)):
            if split_train_val_test_csv.at[i,'train_val_test_split'] == 2:
              self.split_imgs_name.append(split_train_val_test_csv.at[i,"name_image"])
        
        # Create hash name table
        columns = ['image_name', 'hash']
        self.hash_names_table = pandas.DataFrame(columns=columns)
        for i in range(len(self.split_imgs_name)):
          hash_image_name = str(hash(self.split_imgs_name[i]))
          hash_image_name = hash_image_name[1:18]
          new_row = {'image_name': self.split_imgs_name[i], 'hash': hash_image_name}
          self.hash_names_table = pandas.concat([self.hash_names_table, pandas.DataFrame([new_row])], ignore_index=True)
          



    def __getitem__(self, idx):
        # define paths
        self.dataset_path = "/zhome/de/6/201864/Downloads/PVDefectsDS/"
        img_path = self.dataset_path + "/CellsImages/CellsGS/" + str(self.split_imgs_name[idx][:5]) + "_" + str(self.split_imgs_name[idx][5:12]) + "GS" + str(self.split_imgs_name[idx][12:]) + ".png"
        row_index = self.imgs_names[self.imgs_names["namesAllCells"] == self.split_imgs_name[idx]].index[0]
        number_of_labels = int(self.imgs_names["nbDefAllCellsVH"].values[row_index])
        if number_of_labels != 0:
          row = self.labels.loc[self.labels["namesCellsWF"] == self.split_imgs_name[idx]]
          if row['nbCAVH'].values[0] > 0:
            img_class = torch.tensor(1, dtype=torch.uint8)
          elif row['nbCBVH'].values[0] > 0:
            img_class = torch.tensor(1, dtype=torch.uint8)
          elif row['nbCCVH'].values[0] > 0:
            img_class = torch.tensor(2, dtype=torch.uint8)
          elif row['nbFFVH'].values[0] > 0:
            img_class = torch.tensor(3, dtype=torch.uint8)
          else:
            print("Image not labeled correctly")
            print(self.imgs_names["namesAllCells"])
        else:
          img_class = torch.tensor(0, dtype=torch.uint8)

        # load images
        img = Image.open(img_path)
        original_size = img.size[::-1]
        img = torchvision.transforms.functional.resize(img, (300,300))

        image_name = self.split_imgs_name[idx]
        image_id = str(hash(image_name))
        image_id = image_id[1:18]
        image_id = int(image_id)
        image_id = torch.tensor(image_id, dtype=torch.int64)

        if img_class != 0:

          mask_data = scipy.io.loadmat(self.masks_path + "GT_" + str(self.split_imgs_name[idx]) + ".mat")
          number_of_boxes = len(mask_data['GTLabelVH'])
          masks = mask_data['GTMaskVH']
          bbox = []
          labels = []
          areas = []

          mask = masks
          if number_of_boxes > 1:
            for i in range(number_of_boxes):
              mask = masks[:,:,i]
              xmin = math.trunc(min(np.where(mask != 0)[1]) * (300 / original_size[0])) 
              xmax = math.trunc(max(np.where(mask != 0)[1]) * (300 / original_size[0]))
              ymin = math.trunc(min(np.where(mask != 0)[0]) * (300 / original_size[1]))
              ymax = math.trunc(max(np.where(mask != 0)[0]) * (300 / original_size[1]))

              area = (xmax-xmin)*(ymax-ymin)
              bbox.append((xmin, ymin, xmax, ymax))
              areas.append(area)
              labels.append(img_class)
          else:
            xmin = math.trunc(min(np.where(mask != 0)[1]) * (300 / original_size[0]))
            xmax = math.trunc(max(np.where(mask != 0)[1]) * (300 / original_size[0]))
            ymin = math.trunc(min(np.where(mask != 0)[0]) * (300 / original_size[1]))
            ymax = math.trunc(max(np.where(mask != 0)[0]) * (300 / original_size[1]))

            area = (xmax-xmin)*(ymax-ymin)
            bbox.append((xmin, ymin, xmax, ymax))
            areas.append(area)
            labels.append(img_class)

          bbox = torch.tensor(bbox, dtype=torch.float)
          areas = torch.tensor(areas, dtype=torch.int64)
          labels = torch.tensor(labels, dtype=torch.int64)
          iscrowd = torch.zeros((number_of_boxes,), dtype=torch.int64) # suppose all instances are not crowd

          target = {}
          target['boxes'] = bbox
          target['labels'] = labels
          target['area'] = areas
          target['image_id'] = image_id
          target["iscrowd"] = iscrowd

        else:
          target = {}
          bbox = []
          bbox.append((0, 0, 300, 300))
          bbox = torch.tensor(bbox, dtype=torch.float)
          target['boxes'] = bbox
          labels = []
          labels.append(0)
          labels = torch.tensor(labels, dtype=torch.int64)
          target['labels'] = labels
          area = 300*300
          areas = []
          areas.append(area)
          areas = torch.tensor(areas, dtype=torch.int64)
          target['area'] = areas
          target['image_id'] = torch.tensor(image_id, dtype=torch.int64)
          iscrowd = torch.zeros((1,), dtype=torch.int64) # suppose all instances are not crowd
          target["iscrowd"] = iscrowd

        if self.transforms is not None:
          img, target = self.transforms(img,target)

        return img, target

    def __len__(self):
      return len(self.split_imgs_name)

    def get_hash_names(self):
      return self.hash_names_table
    
    def get_imgs_names(self):
      return self.imgs_names

    def get_labels(self):
      return self.labels

