from torchvision.datasets import CocoDetection
from pathlib import Path
import json

class CocoClientDataset(CocoDetection):
    def __init__(self, root: str, annFile: str, client_split_file: str, transform=None):
        # Load client-specific image IDs
        with open(client_split_file, "r") as f:
            client_data = json.load(f)
        self.image_ids = client_data["image_ids"]
        
        # Initialize COCO dataset
        super().__init__(root, annFile, transform=transform)
    
    def __getitem__(self, index):
        # Override to filter images by client's IDs
        img_id = self.image_ids[index]
        img = self.coco.loadImgs(img_id)[0]
        image_path = Path(self.root) / img["file_name"]
        image = Image.open(image_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        if self.transform:
            image = self.transform(image)
        
        # Format annotations for object detection
        target = {
            "boxes": [ann["bbox"] for ann in annotations],
            "labels": [ann["category_id"] for ann in annotations],
            "image_id": img_id,
        }
        return image, target