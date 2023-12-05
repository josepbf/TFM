from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

      self.writer = SummaryWriter('runs/' + name_experiment + '_' + dt_string + '/' + folder_name)

   def store_metric(metric_name, scalar, step):
      self.writer.add_scalar(metric_name, scalar, step)
   
   def get_train_or_val():
      return self.train_or_val

   def store_matrix(matrix_name, matrix, step):
      self.writer.add_image(matrix_name, matrix, step)

   def close_writer():
      self.writer.close()
      