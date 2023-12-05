import wandb
from datetime import datetime
import torch
import numpy

class Writer:
   def __init__(self, name_experiment, train_or_val):
      self.name_experiment = name_experiment
      self.train_or_val = train_or_val # 0 for train, 1 for val
      if len(self.name_experiment) == 0:
         self.name_experiment = 'defaultname'

      if self.train_or_val == 0:
         folder_name = 'training'
      else:
         folder_name = 'validation'
      
      now = datetime.now()
      dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

      run_name = f'{name_experiment}_{dt_string}_{folder_name}'
      wandb.init(name=run_name, project=name_experiment)


   def store_metric(metric_name, scalar, step):
      wandb.log({metric_name: scalar}, step=step)
   
   def get_train_or_val():
      return self.train_or_val

   def store_matrix(matrix_name, matrix, step):
      wandb.log({matrix_name: matrix.numpy()}, step=step)

   def close_writer():
      wandb.finish()
      