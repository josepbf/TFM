# Source: https://pytorch.org/vision/stable/models.html#object-detection
import torchvision
import torch
from datetime import datetime
import torch.nn.functional as F

class Model:
    def __init__(self, 
                model_name,
                trainable_backbone_layers, 
                num_classes = 4, 
                pretrained = True, 
                pretrained_backbone = True,
                label_smoothing = 0.1):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.trainable_backbone_layers = trainable_backbone_layers
        self.pretrained_backbone = pretrained_backbone
        self.label_smoothing = label_smoothing

        if self.model_name == 'FasterRCNN_ResNet-50-FPN':
            # Faster R-CNN model with a ResNet-50-FPN backbone from the 
            # Faster R-CNN: Towards Real-Time Object Detection with Region 
            # Proposal Networks paper.
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes, 
                pretrained_backbone=pretrained_backbone, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

        elif self.model_name == 'FasterRCNN_ResNet-50-FPN_v2':    
            # Constructs an improved Faster R-CNN model with a ResNet-50-FPN 
            # backbone from Benchmarking Detection Transfer Learning with Vision 
            # Transformers paper.
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                num_classes=num_classes, 
                pretrained_backbone=pretrained_backbone, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

        elif self.model_name == 'FasterRCNN_MobileNetV3-Large':
            # Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large 
            # FPN backbone.
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=pretrained, 
                num_classes=num_classes, 
                pretrained_backbone=pretrained_backbone, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 6
        
        elif self.model_name == 'RetinaNet_ResNet-50-FPN':
            # Constructs a RetinaNet model with a ResNet-50-FPN backbone.
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(
                weights= 'DEFAULT',
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

        elif self.model_name == 'RetinaNet_ResNet-50-FPN v2':
            # Constructs an improved RetinaNet model with a ResNet-50-FPN backbone.
            self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights= 'DEFAULT',
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

        elif self.model_name == 'FCOS':
            # Constructs a FCOS model with a ResNet-50-FPN backbone.
            self.model = torchvision.models.detection.fcos_resnet50_fpn(
                weights= 'DEFAULT',
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5
                
        elif self.model_name == 'SSD':
             # The SSD300 model is based on the SSD: Single Shot MultiBox Detector paper.
            self.model = torchvision.models.detection.ssd300_vgg16(
                weights= 'DEFAULT',
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5
        
    def get_model(self):
        return self.model

    def load_model(self, name):
        self.model = torch.load("./models_saved/" + str(name) + ".pth")

    def save_model(self, net, epoch):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        name_to_save = str("./" + self.model_name + "_epoch_" + str(epoch) + dt_string + ".pth")
        torch.save(net, name_to_save)

    def add_label_smoothing(self):
        self.model.roi_heads.fastrcnn_loss = fastrcnn_loss_custom

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

        return classification_loss, box_loss
