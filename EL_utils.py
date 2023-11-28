import json
import time
import datetime
import numpy as np

import collections
from collections import defaultdict, deque

import os
from os.path import isfile, join
from os import listdir

import torch.distributed as dist

def compute_confusion_matrix_training(epoch, isValData = False, iou_threshold = 0.5):
  # Read all the names of outputs
  jsonPath = './FullTrainingDataset/Annotations'
  outputsPath = str('./outputs_faster_train_1/faster/outputs_epoch_' + str(epoch) + '_train/')
  targetsPath = './FullTrainingDataset/Annotations'

  jsonNames = [f for f in listdir(jsonPath) if isfile(join(jsonPath, f))]
  outputNames = [f for f in listdir(outputsPath) if isfile(join(outputsPath, f))]
  jsonNames.sort()
  outputNames.sort()

  score_thresholds = [0.5,0.7,0.9]
  for score_threshold in score_thresholds:
    confusion_matrix = torch.zeros(4,4)
    true_positive = 0
    positive = 0
    true = 0
    # totes les outputs
    for out_ind in range(len(outputNames)):

      # Read the output
      out = torch.load(str(outputsPath + '/' + outputNames[out_ind]))

      # Read the targets
      targetName = outputNames[out_ind]
      targetName = targetName.replace("pt","json")
      bbox_path = os.path.join("./FullTrainingDataset/Annotations", targetName)

      tree = open(bbox_path,)
      root = y = json.load(tree)
      objects = root["annotations"]
      bbox = []
      labels = []
      for obj in objects:
        if obj["label"] == "Leukocyte" or obj["label"] == "Parasite":
          xmin = int(obj["posX"])
          ymin = int(obj["posY"])
          xmax = xmin + int(obj["width"])
          ymax = ymin + int(obj["height"])
          bbox.append((xmin, ymin, xmax, ymax))

        if obj["label"] == "Leukocyte":
          labels.append(1)
        if obj["label"] == "Parasite":
          if obj["disease"] != "+malaria mature trophozoite plasmodium spp":
            labels.append(2)
          elif obj["disease"] == "+malaria mature trophozoite plasmodium spp":
            labels.append(3)
        
      ground_truth_boxes = torch.tensor(bbox, dtype=torch.float)
      ground_truth_labels = torch.tensor(labels, dtype=torch.int64)

      # Compute the confusion matrix for that image

      # 1 Discard detections with IoU greater or equal than 0.5 Keep the highest score
      detected_boxes = out['boxes']
      detected_scores = out['scores']
      detected_labels = out['labels']
      indices = torchvision.ops.batched_nms(boxes = detected_boxes, scores = detected_scores, idxs = detected_labels, iou_threshold = iou_threshold)

      boxes = []
      scores = []
      labels = []
      for indice in indices:
        boxes.append(detected_boxes[indice])
        scores.append(detected_scores[indice])
        labels.append(detected_labels[indice])

      detected_boxes_prefilter = boxes
      detected_scores_prefilter = scores
      detected_labels_prefilter = labels

      # 2 Only detections with a score greater or equal than 0.5 are considered. Anything that’s under this value is discarded.
      # SCORE THRESHOLD IS A PARAM
      boxes = []
      scores = []
      labels = []
      for i in range(len(indices)):
        if detected_scores_prefilter[i] > score_threshold:
          boxes.append(detected_boxes_prefilter[i])
          scores.append(detected_scores_prefilter[i])
          labels.append(detected_labels_prefilter[i])

      detected_boxes = boxes
      detected_scores = scores
      detected_labels = labels

      # 3 For each ground-truth box, the algorithm generates the IoU (Intersection over Union) with every detected box. 
      # A match is found if both boxes have an IoU greater or equal than 0.5.
      # IoU IS A PARAM
      boxes = []
      for detected_box in detected_boxes:
        boxes.append(torch.trunc(detected_box))

      detected_boxes = boxes

      boxes = []
      for detected_box in detected_boxes:
        xmin = detected_box[0].item()
        ymin = detected_box[1].item()
        xmax = detected_box[2].item()
        ymax = detected_box[3].item()
        boxes.append((xmin, ymin, xmax, ymax))

      detected_boxes = torch.tensor(boxes, dtype=torch.float)

      # no detections
      if len(detected_boxes) == 0:
        for m in range(len(ground_truth_labels)):
          confusion_matrix[ground_truth_labels[m],0] += 1

      # no ground truth
      if len(ground_truth_boxes) == 0:
        for m in range(len(detected_labels)):
          confusion_matrix[0,detected_labels[m]] += 1

      # detection and ground truth
      if len(ground_truth_boxes) != 0 and len(detected_boxes) != 0:
        iou = torchvision.ops.box_iou(boxes1 = ground_truth_boxes, boxes2 = detected_boxes)

        # 4 The list of matches is pruned to remove duplicates (ground-truth boxes that match with more than one detection box 
        # or vice versa). If there are duplicates, the best match (greater IoU) is always selected.
        detection_results = np.zeros(len(detected_boxes))
        ground_truth_results = np.zeros(len(ground_truth_boxes))
        
        for n in range(len(iou)):
          for m in range(len(iou[0])):
            if iou[n,m] >= iou_threshold:
              index_detection = torch.argmax(input = iou[n], dim=None, keepdim=False) # Aquesta es l'index de la detecció
              index_ground_truth = n # Aquest es l'index del ground truth
              
              # Només comptar ground_truth_results quan la detecció en questió encara no te resultat assignat
              if detection_results[index_detection] == 0:
                ground_truth_results[index_ground_truth] += 1
              detection_results[index_detection] += 1
              
              if detection_results[index_detection] == 1 and ground_truth_results[index_ground_truth] == 1:
                confusion_matrix[ground_truth_labels[index_ground_truth],detected_labels[index_detection]] += 1

        # 5 The confusion matrix is updated to reflect the resulting matches between ground-truth and detections.

        # 6 Objects that are part of the ground-truth but weren’t detected are counted in the last column of the matrix
        # (in the row corresponding to the ground-truth class). Objects that were detected but aren’t part of the confusion 
        # matrix are counted in the last row of the matrix (in the column corresponding to the detected class).
        for index_detection in range(len(detection_results)):
          if detection_results[index_detection] == 0:
            confusion_matrix[0,detected_labels[index_detection]] += 1

        for index_ground_truth in range(len(ground_truth_results)):
          if ground_truth_results[index_ground_truth] == 0:
            confusion_matrix[ground_truth_labels[index_ground_truth],0] += 1
            
      if out_ind == len(outputNames)-1 :
        print("Confunsion Matrix completed:")
        print("Score = " + str(score_threshold))
        print(confusion_matrix)

        true_positive = confusion_matrix[1,1] + confusion_matrix[2,2] + confusion_matrix[3,3]
        positive = true_positive + confusion_matrix[1,2] +confusion_matrix[1,3] + confusion_matrix[2,1] + confusion_matrix[2,3] +confusion_matrix[3,1] + confusion_matrix[3,2] +confusion_matrix[0,1]+confusion_matrix[0,2]+confusion_matrix[0,3]
        true = true_positive + confusion_matrix[1,2] +confusion_matrix[1,3] + confusion_matrix[2,1] + confusion_matrix[2,3] +confusion_matrix[3,1] + confusion_matrix[3,2] + confusion_matrix[1,0]+confusion_matrix[2,0]+confusion_matrix[3,0]

        if positive>0 and true>0:
          if score_threshold == 0.5:
            score_name = '50'
            precision_50 = true_positive / positive
            recall_50 = true_positive / true
            fmeasure_50 = 2*(precision_50*recall_50)/(precision_50+recall_50)
            print("Precision (score = 50): " + str(precision_50))
            print("Recall (score = 50): " + str(recall_50))
            print("Fmeasure (score = 50): " + str(fmeasure_50))
            writer_training.add_scalar('ConfusionMatrix/precision_0.5_score', precision_50, epoch)
            writer_training.add_scalar('ConfusionMatrix/recall_0.5_score', recall_50, epoch)
            writer_training.add_scalar('ConfusionMatrix/fmeasure_0.5_score', fmeasure_50, epoch)
          if score_threshold == 0.7:
            score_name = '70'
            precision_70 = true_positive / positive
            recall_70 = true_positive / true
            fmeasure_70 = 2*(precision_70*recall_70)/(precision_70+recall_70)
            print("Precision (score = 70): " + str(precision_70))
            print("Recall (score = 70): " + str(recall_70))
            print("Fmeasure (score = 70): " + str(fmeasure_70))
            writer_training.add_scalar('ConfusionMatrix/precision_0.7_score', precision_70, epoch)
            writer_training.add_scalar('ConfusionMatrix/recall_0.7_score', recall_70, epoch)
            writer_training.add_scalar('ConfusionMatrix/fmeasure_0.7_score', fmeasure_70, epoch)
          if score_threshold == 0.9:
            score_name = '90'        
            precision_90 = true_positive / positive
            recall_90 = true_positive / true
            fmeasure_90 = 2*(precision_90*recall_90)/(precision_90+recall_90)  
            print("Precision (score = 90): " + str(precision_90))
            print("Recall (score = 90): " + str(recall_90))
            print("Fmeasure (score = 90): " + str(fmeasure_90))
            writer_training.add_scalar('ConfusionMatrix/precision_0.9_score', precision_90, epoch)
            writer_training.add_scalar('ConfusionMatrix/recall_0.9_score', recall_90, epoch)
            writer_training.add_scalar('ConfusionMatrix/fmeasure_0.9_score', fmeasure_90, epoch)
        # Store confusion matrix image
        matrix_name = str('./confusions_matrix_faster_train_1/faster/training/confusion_matrix_epoch_' + str(epoch) + '_score_' + score_name + '.pt')
        torch.save(confusion_matrix, matrix_name)    


def compute_confusion_matrix_validation(epoch, isValData = False, iou_threshold = 0.5):
  # Read all the names of outputs
  jsonPath = './FullValidationDataset/Annotations'
  outputsPath = str('./outputs_faster_train_1/faster/outputs_epoch_' + str(epoch) + '_val/')
  targetsPath = './FullValidationDataset/Annotations'

  jsonNames = [f for f in listdir(jsonPath) if isfile(join(jsonPath, f))]
  outputNames = [f for f in listdir(outputsPath) if isfile(join(outputsPath, f))]
  jsonNames.sort()
  outputNames.sort()

  score_thresholds = [0.5,0.7,0.9]
  for score_threshold in score_thresholds:
    confusion_matrix = torch.zeros(4,4)
    true_positive = 0
    positive = 0
    true = 0
    # totes les outputs
    for out_ind in range(len(outputNames)):

      # Read the output
      out = torch.load(str(outputsPath + '/' + outputNames[out_ind]))

      # Read the targets
      targetName = outputNames[out_ind]
      targetName = targetName.replace("pt","json")
      bbox_path = os.path.join("./FullValidationDataset/Annotations", targetName)

      tree = open(bbox_path,)
      root = y = json.load(tree)
      objects = root["annotations"]
      bbox = []
      labels = []
      for obj in objects:
        if obj["label"] == "Leukocyte" or obj["label"] == "Parasite":
          xmin = int(obj["posX"])
          ymin = int(obj["posY"])
          xmax = xmin + int(obj["width"])
          ymax = ymin + int(obj["height"])
          bbox.append((xmin, ymin, xmax, ymax))

        if obj["label"] == "Leukocyte":
          labels.append(1)
        if obj["label"] == "Parasite":
          if obj["disease"] != "+malaria mature trophozoite plasmodium spp":
            labels.append(2)
          elif obj["disease"] == "+malaria mature trophozoite plasmodium spp":
            labels.append(3)
        
      ground_truth_boxes = torch.tensor(bbox, dtype=torch.float)
      ground_truth_labels = torch.tensor(labels, dtype=torch.int64)

      # Compute the confusion matrix for that image

      # 1 Discard detections with IoU greater or equal than 0.5 Keep the highest score
      detected_boxes = out['boxes']
      detected_scores = out['scores']
      detected_labels = out['labels']
      indices = torchvision.ops.batched_nms(boxes = detected_boxes, scores = detected_scores, idxs = detected_labels, iou_threshold = iou_threshold)

      boxes = []
      scores = []
      labels = []
      for indice in indices:
        boxes.append(detected_boxes[indice])
        scores.append(detected_scores[indice])
        labels.append(detected_labels[indice])

      detected_boxes_prefilter = boxes
      detected_scores_prefilter = scores
      detected_labels_prefilter = labels

      # 2 Only detections with a score greater or equal than 0.5 are considered. Anything that’s under this value is discarded.
      # SCORE THRESHOLD IS A PARAM
      boxes = []
      scores = []
      labels = []
      for i in range(len(indices)):
        if detected_scores_prefilter[i] > score_threshold:
          boxes.append(detected_boxes_prefilter[i])
          scores.append(detected_scores_prefilter[i])
          labels.append(detected_labels_prefilter[i])

      detected_boxes = boxes
      detected_scores = scores
      detected_labels = labels

      # 3 For each ground-truth box, the algorithm generates the IoU (Intersection over Union) with every detected box. 
      # A match is found if both boxes have an IoU greater or equal than 0.5.
      # IoU IS A PARAM
      boxes = []
      for detected_box in detected_boxes:
        boxes.append(torch.trunc(detected_box))

      detected_boxes = boxes

      boxes = []
      for detected_box in detected_boxes:
        xmin = detected_box[0].item()
        ymin = detected_box[1].item()
        xmax = detected_box[2].item()
        ymax = detected_box[3].item()
        boxes.append((xmin, ymin, xmax, ymax))

      detected_boxes = torch.tensor(boxes, dtype=torch.float)

      # no detections
      if len(detected_boxes) == 0:
        for m in range(len(ground_truth_labels)):
          confusion_matrix[ground_truth_labels[m],0] += 1

      # no ground truth
      if len(ground_truth_boxes) == 0:
        for m in range(len(detected_labels)):
          confusion_matrix[0,detected_labels[m]] += 1

      # detection and ground truth
      if len(ground_truth_boxes) != 0 and len(detected_boxes) != 0:
        iou = torchvision.ops.box_iou(boxes1 = ground_truth_boxes, boxes2 = detected_boxes)

        # 4 The list of matches is pruned to remove duplicates (ground-truth boxes that match with more than one detection box 
        # or vice versa). If there are duplicates, the best match (greater IoU) is always selected.
        detection_results = np.zeros(len(detected_boxes))
        ground_truth_results = np.zeros(len(ground_truth_boxes))
        
        for n in range(len(iou)):
          for m in range(len(iou[0])):
            if iou[n,m] >= iou_threshold:
              index_detection = torch.argmax(input = iou[n], dim=None, keepdim=False) # Aquesta es l'index de la detecció
              index_ground_truth = n # Aquest es l'index del ground truth
              
              # Només comptar ground_truth_results quan la detecció en questió encara no te resultat assignat
              if detection_results[index_detection] == 0:
                ground_truth_results[index_ground_truth] += 1
              detection_results[index_detection] += 1
              
              if detection_results[index_detection] == 1 and ground_truth_results[index_ground_truth] == 1:
                confusion_matrix[ground_truth_labels[index_ground_truth],detected_labels[index_detection]] += 1

        # 5 The confusion matrix is updated to reflect the resulting matches between ground-truth and detections.

        # 6 Objects that are part of the ground-truth but weren’t detected are counted in the last column of the matrix
        # (in the row corresponding to the ground-truth class). Objects that were detected but aren’t part of the confusion 
        # matrix are counted in the last row of the matrix (in the column corresponding to the detected class).
        for index_detection in range(len(detection_results)):
          if detection_results[index_detection] == 0:
            confusion_matrix[0,detected_labels[index_detection]] += 1

        for index_ground_truth in range(len(ground_truth_results)):
          if ground_truth_results[index_ground_truth] == 0:
            confusion_matrix[ground_truth_labels[index_ground_truth],0] += 1
            
      if out_ind == len(outputNames)-1 :
        print("Confunsion Matrix completed:")
        print("Score = " + str(score_threshold))
        print(confusion_matrix)

        true_positive = confusion_matrix[1,1] + confusion_matrix[2,2] + confusion_matrix[3,3]
        positive = true_positive + confusion_matrix[1,2] +confusion_matrix[1,3] + confusion_matrix[2,1] + confusion_matrix[2,3] +confusion_matrix[3,1] + confusion_matrix[3,2] +confusion_matrix[0,1]+confusion_matrix[0,2]+confusion_matrix[0,3]
        true = true_positive + confusion_matrix[1,2] +confusion_matrix[1,3] + confusion_matrix[2,1] + confusion_matrix[2,3] +confusion_matrix[3,1] + confusion_matrix[3,2] + confusion_matrix[1,0]+confusion_matrix[2,0]+confusion_matrix[3,0]

        if positive>0 and true>0:
          if score_threshold == 0.5:
            score_name = '50'
            precision_50 = true_positive / positive
            recall_50 = true_positive / true
            fmeasure_50 = 2*(precision_50*recall_50)/(precision_50+recall_50)
            print("Precision (score = 50): " + str(precision_50))
            print("Recall (score = 50): " + str(recall_50))
            print("Fmeasure (score = 50): " + str(fmeasure_50))
            writer_validation.add_scalar('ConfusionMatrix/precision_0.5_score', precision_50, epoch)
            writer_validation.add_scalar('ConfusionMatrix/recall_0.5_score', recall_50, epoch)
            writer_validation.add_scalar('ConfusionMatrix/fmeasure_0.5_score', fmeasure_50, epoch)
          if score_threshold == 0.7:
            score_name = '70'
            precision_70 = true_positive / positive
            recall_70 = true_positive / true
            fmeasure_70 = 2*(precision_70*recall_70)/(precision_70+recall_70)
            print("Precision (score = 70): " + str(precision_70))
            print("Recall (score = 70): " + str(recall_70))
            print("Fmeasure (score = 70): " + str(fmeasure_70))
            writer_validation.add_scalar('ConfusionMatrix/precision_0.7_score', precision_70, epoch)
            writer_validation.add_scalar('ConfusionMatrix/recall_0.7_score', recall_70, epoch)
            writer_validation.add_scalar('ConfusionMatrix/fmeasure_0.7_score', fmeasure_70, epoch)
          if score_threshold == 0.9:
            score_name = '90'        
            precision_90 = true_positive / positive
            recall_90 = true_positive / true
            fmeasure_90 = 2*(precision_90*recall_90)/(precision_90+recall_90)  
            print("Precision (score = 90): " + str(precision_90))
            print("Recall (score = 90): " + str(recall_90))
            print("Fmeasure (score = 90): " + str(fmeasure_90))
            writer_validation.add_scalar('ConfusionMatrix/precision_0.9_score', precision_90, epoch)
            writer_validation.add_scalar('ConfusionMatrix/recall_0.9_score', recall_90, epoch)
            writer_validation.add_scalar('ConfusionMatrix/fmeasure_0.9_score', fmeasure_90, epoch)
        # Store confusion matrix image
        matrix_name = str('./confusions_matrix_faster_train_1/faster/validation/confusion_matrix_epoch_' + str(epoch) + '_score_' + score_name + '.pt')
        torch.save(confusion_matrix, matrix_name)    

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count != 0:
            return self.total / self.count
        else:
            return 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, epoch, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict



def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True