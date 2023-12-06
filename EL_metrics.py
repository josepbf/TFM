import wandb
from datetime import datetime
import torch
import numpy

class Writer:
   def __init__(self, name_experiment, train_or_val, config):
      self.name_experiment = name_experiment
      self.train_or_val = train_or_val # 0 for train, 1 for val
      if len(self.name_experiment) == 0:
         self.name_experiment = 'finding_bugs'

      if self.train_or_val == 0:
         folder_name = 'training'
      else:
         folder_name = 'validation'

      wandb.init(project=name_experiment, config = config)


   def store_metric(self, metric_name, scalar, step):
      wandb.log({metric_name: scalar}, step=step)
   
   def get_train_or_val(self):
      return self.train_or_val

   def store_matrix(self, matrix_name, matrix, step):
      wandb.log({matrix_name: matrix.numpy()}, step=step)

   def close_writer(self):
      wandb.finish()
      