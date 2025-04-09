import torch
import torch.nn as nn

from external.yolov5.utils.loss import ComputeLoss
from external.yolov5.models.yolo import Model
from external.yolov5.models.common import AutoShape


class YoloV5Loss(nn.Module):
    def __init__(self, model, hyp=None):
        super(YoloV5Loss, self).__init__()
        # Initialize the original ComputeLoss with the model
        self.compute_loss = ComputeLoss(model)
        
        # Store hyperparameters (optional, can be passed or hardcoded)
        if hyp is None:
            # Default hyperparameters (mimicking hyp.scratch.yaml)
            self.hyp = {
                'box': 0.05,    # box loss gain
                'cls': 0.5,     # cls loss gain
                'obj': 1.0,     # obj loss gain
                'fl_gamma': 0.0 # focal loss gamma (0 = disabled)
            }
        else:
            self.hyp = hyp

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output (list/tuple of tensors from YOLO layers)
            targets: Ground truth in YOLO format [batch_idx, class, x, y, w, h]
        Returns:
            total_loss: Scalar tensor representing the total loss
        """
        # Use the original ComputeLoss to calculate the loss
        loss, loss_items = self.compute_loss(predictions, targets)
        
        return loss  # Return total loss (can return loss_items if needed)
    

def load_yolov5_model(model_path="examples/fedavg_yolo/yolo_weights/yolov5su.pt", num_classes=80, pretrained=True):
    """
    Load a YOLOv5 model for federated learning.
    
    Args:
        model_type (str): The YOLOv5 model variant ('yolov5s', 'yolov5m', etc.)
        config_path (str): Path to custom YAML config (optional)
        num_classes (int): Number of classes to detect
        pretrained (bool): Whether to load pretrained weights
    
    Returns:
        torch.nn.Module: The YOLOv5 model
    """
    # Load model config
    model = Model(nc=num_classes)
 
    # Load pretrained weights if requested
    if pretrained:
        ckpt = torch.load(model_path, map_location='cpu')
        
        # Load compatible weights (excluding the detection heads if num_classes differs)
        state_dict = ckpt['model'].float().state_dict()
        exclude_layers = []
        
        # Skip detection head params if num_classes differs from pretrained model
        if ckpt['model'].nc != num_classes:
            exclude_layers = [f'model.{x}' for x in range(10, 21) if x % 3 == 0]  # Output layers
        
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if 
                     k in model_state_dict and not any(x in k for x in exclude_layers)}
        model.load_state_dict(state_dict, strict=False)
    
    # Apply AutoShape wrapper for preprocessing
    model = AutoShape(model)
    
    return model

