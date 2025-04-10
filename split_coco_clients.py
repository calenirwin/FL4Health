import json
import numpy as np
from pathlib import Path
import argparse

def split_coco_clients(annotations_path: str, output_dir: str, num_clients: int) -> None:
    # Load COCO annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)
    
    # Extract image IDs and their categories (non-IID simulation)
    image_id_to_categories = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_id_to_categories:
            image_id_to_categories[image_id] = []
        image_id_to_categories[image_id].append(ann["category_id"])
    
    # Assign images to clients based on dominant category (simulate non-IID)
    clients = {i: [] for i in range(num_clients)}
    for img_id, cats in image_id_to_categories.items():
        dominant_cat = max(set(cats), key=cats.count)
        client_id = dominant_cat % num_clients  # Simple assignment
        clients[client_id].append(img_id)
    
    # Save client splits
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for client_id, img_ids in clients.items():
        with open(output_dir / f"client_{client_id}.json", "w") as f:
            json.dump({"client_id": client_id, "image_ids": img_ids}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=10)
    args = parser.parse_args()
    split_coco_clients(args.annotations_path, args.output_dir, args.num_clients)