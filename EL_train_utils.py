import math
import sys
import torch

from EL_utils import MetricLogger, SmoothedValue, reduce_dict

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, iteration, writer):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, epoch, header):

        iteration = iteration + 1
        #print("Iteration: " + str(iteration))
        images = list(image.to(device) for image in images)
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

        writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss'), loss, str('iteration_' + writer.get_folder_name()), iteration)
        writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_classifier'), loss_classifier, str('iteration_' + writer.get_folder_name()), iteration)
        writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_box_reg'), loss_box_reg, str('iteration_' + writer.get_folder_name()), iteration)
        writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_objectness'), loss_objectness, str('iteration_' + writer.get_folder_name()), iteration)
        writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_rpn_box_reg'), loss_rpn_box_reg, str('iteration_' + writer.get_folder_name()), iteration)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        #metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        #metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def fastrcnn_loss_custom(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels, self.label_smoothing)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()