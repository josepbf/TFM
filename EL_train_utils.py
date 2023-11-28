import math
import sys
import torch

from EL_utils import MetricLogger, SmoothedValue, reduce_dict

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, iteration):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, epoch, header):

        iteration = iteration + 1
        print("Iteration: " + str(iteration))
        images = list(image.to(device) for image in images)
        print(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        loss = losses_reduced.item()
        loss_classifier = loss_dict_reduced['loss_classifier'].item()
        loss_box_reg = loss_dict_reduced['loss_box_reg'].item()
        loss_objectness = loss_dict_reduced['loss_objectness'].item()
        loss_rpn_box_reg = loss_dict_reduced['loss_rpn_box_reg'].item()

        #writer_training.add_scalar('Loss/loss', loss, iteration)
        #writer_training.add_scalar('Loss/loss_classifier', loss_classifier, iteration)
        #writer_training.add_scalar('Loss/loss_box_reg', loss_box_reg, iteration)
        #writer_training.add_scalar('Loss/loss_objectness', loss_objectness, iteration)
        #writer_training.add_scalar('Loss/loss_rpn_box_reg', loss_rpn_box_reg, iteration)
        
        learning_rate = optimizer.param_groups[0]["lr"]
        #writer_training.add_scalar('LearningRate/lr', learning_rate, iteration)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        #metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        #metric_logger.update(lr=optimizer.param_groups[0]["lr"])