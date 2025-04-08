
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
from pycocotools.coco import COCO


def convert_coco_to_yolo(coco_json_path, images_dir, output_label_dir, output_image_dir, image_size=(640, 640)):
    coco = COCO(coco_json_path)

    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    cats = coco.loadCats(coco.getCatIds())
    cat2label = {cat['id']: idx for idx, cat in enumerate(cats)}

    for img_id in tqdm(img_ids, desc="Converting COCO to YOLO format"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(images_dir, file_name)

        # Copy image to output directory and resize
        image = Image.open(img_path).convert("RGB")
        image = image.resize(image_size)
        image.save(os.path.join(output_image_dir, file_name))

        width, height = image_size
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        yolo_labels = []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height
            class_id = cat2label[ann['category_id']]
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        label_file = os.path.join(output_label_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_labels))


def prepare_yolo_dataset(
    coco_root="/projects/federated_learning/Hitachi/MSCOCO2017",
    yolo_output_root="data/coco_yolo",
    image_size=(640, 640)
):
    # Set paths
    paths = {
        "train_img_dir": f"{coco_root}/images/train2017",
        "val_img_dir": f"{coco_root}/images/val2017",
        "train_ann": f"{coco_root}/annotations/instances_train2017.json",
        "val_ann": f"{coco_root}/annotations/instances_val2017.json",
        "yolo_train_img": f"{yolo_output_root}/images/train2017",
        "yolo_val_img": f"{yolo_output_root}/images/val2017",
        "yolo_train_lbl": f"{yolo_output_root}/labels/train2017",
        "yolo_val_lbl": f"{yolo_output_root}/labels/val2017",
    }

    convert_coco_to_yolo(paths["train_ann"], paths["train_img_dir"], paths["yolo_train_lbl"], paths["yolo_train_img"], image_size)
    convert_coco_to_yolo(paths["val_ann"], paths["val_img_dir"], paths["yolo_val_lbl"], paths["yolo_val_img"], image_size)

    print("✅ YOLOv5 dataset prepared!")


if __name__ == "__main__":
    prepare_yolo_dataset()