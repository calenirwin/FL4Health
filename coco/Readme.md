# Federated Learning for COCO Object Detection with FL4Health

This repository demonstrates federated learning (FL) for object detection on the COCO 2017 dataset using the [FL4Health](https://github.com/VectorInstitute/FL4Health) framework. 
The example uses a **non-IID client split** based on object categories.
## Setup
1. **Install Dependencies**:
   ```bash
   pip install torch torchvision pycocotools Pillow fl4health
Prepare COCO Dataset:
Download COCO 2017 Test Images and annotations.
Place images in data/coco/test2017 and annotations in data/coco/annotations.

Usage
Split Data for Clients:

bash
Copy
python split_coco_clients.py \
  --annotations_path data/coco/annotations/instances_val2017.json \
  --output_dir client_splits \
  --num_clients 10
Run Federated Training:

bash
Copy
python federated_coco.py --config configs/coco_config.yaml
Key Files
split_coco_clients.py: Splits COCO data into non-IID client subsets.

coco_dataset.py: Custom dataset loader for client-specific COCO data.

federated_coco.py: Federated training script using FL4Health.

configs/coco_config.yaml: Configuration for paths, clients, and hyperparameters.

Implementation Details
Non-IID Split: Clients receive data dominated by specific object categories.

Object Detection: Uses Faster R-CNN with PyTorch.

Dataset Class: CocoClientDataset filters COCO images per client and converts annotations to PyTorch format.

References
COCO Dataset

FL4Health Framework
