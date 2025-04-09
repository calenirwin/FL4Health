import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.config import narrow_dict_type

from examples.fedavg_yolo.yolo_utils import load_yolov5_model, YoloV5Loss
from fl4health.utils.load_data import get_mscoco_dataloader


class YOLOv5MSCOCOClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        # Extract configuration parameters
        batch_size = narrow_dict_type(config, "batch_size", int)
        
        # Load dataset with YOLOv5 formatting
        train_loader, val_loader = get_mscoco_dataloader(
            data_path=self.data_path,
            batch_size=batch_size,
        )
        return train_loader, val_loader


    def get_criterion(self) -> _Loss:
        return YoloV5Loss()

    def get_optimizer(self, config: Config) -> Optimizer:
        lr = narrow_dict_type(config, "learning_rate", float, 0.01)
        weight_decay = narrow_dict_type(config, "weight_decay", float, 0.0005)
        momentum = narrow_dict_type(config, "momentum", float, 0.937)
        
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    def get_model(self, config: Config) -> nn.Module:
        # Load YOLOv5s model
        model_path = narrow_dict_type(config, "model_path", str)
        num_classes = narrow_dict_type(config, "nc", int)
        
        # Initialize YOLOv5s model
        model = load_yolov5_model(
            model_path=model_path,
            num_classes=num_classes
        ).to(self.device)
        
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Federated Learning Client")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    
    client = YOLOv5MSCOCOClient(data_path, device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
    