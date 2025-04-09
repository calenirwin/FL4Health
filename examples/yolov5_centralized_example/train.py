import os
import torch
from ultralytics import YOLO
from flwr.common.typing import Config
from torch.utils.data import DataLoader

from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import get_mscoco_dataloader


def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
    batch_size = narrow_dict_type(config, "batch_size", int)
    data_path = narrow_dict_type(config, "data_path", str)
    train_loader, val_loader, _ = get_mscoco_dataloader(data_path, batch_size)
    return train_loader, val_loader

def train_yolov5(config_path, epochs=100, batch_size=16, img_size=640, weights='yolov5s.pt'):
    """
    Train YOLOv5 model with the given configuration
    """
    # Load YOLOv5 model
    model = YOLO(weights)
    
    # Train the model
    results = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    return model, results


# Example usage:
if __name__ == "__main__":
    SAVE = True

    config_path = "examples/yolov5_centralized_example/coco.yaml"
    save_path = "examples/yolov5_centralized_example/yolov5_coco.pt"


    model, results = train_yolov5(
        config_path=config_path,
        epochs=5,
        batch_size=16,
        img_size=640,
        weights='yolov5s.pt'  # Use 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', or 'yolov5x.pt'
    )

    if SAVE:
        model.save(save_path)
    
    # Evaluate model on validation set
    val_results = model.val()
    print(f"Validation results: {val_results}")
    
    # Perform inference on a test image
    test_img = '/projects/federated_learning/Hitachi/MSCOCO2017/images/val2017/000000000139.jpg'
    if os.path.exists(test_img):
        results = model.predict(test_img, save=True, conf=0.25)
        print(f"Prediction results saved to: {results[0].save_dir}")
