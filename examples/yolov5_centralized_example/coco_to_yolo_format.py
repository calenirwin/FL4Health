import os
from pathlib import Path
from pycocotools.coco import COCO


def convert_coco_to_yolo(coco_instance, label_dir):
        """Convert COCO annotations to YOLOv5 format"""
        # Get all categories
        cats = coco_instance.loadCats(coco_instance.getCatIds())
        cat_ids = [cat['id'] for cat in cats]
        
        # Map category ids to 0-indexed values for YOLO
        cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        
        # Process all images
        img_ids = coco_instance.getImgIds()
        for img_id in img_ids:
            # Get image info
            img_info = coco_instance.loadImgs(img_id)[0]
            img_width, img_height = img_info['width'], img_info['height']
            
            # Get annotations for this image
            ann_ids = coco_instance.getAnnIds(imgIds=img_id)
            anns = coco_instance.loadAnns(ann_ids)
            
            # Create YOLO label file
            label_file = os.path.join(label_dir, Path(img_info['file_name']).stem + '.txt')
            
            with open(label_file, 'w') as f:
                for ann in anns:
                    # Skip annotations without segmentation or with empty bbox
                    if 'bbox' not in ann or len(ann['bbox']) != 4:
                        continue
                    
                    # Get category index for YOLO format
                    if ann['category_id'] not in cat_id_to_idx:
                        continue
                    cat_idx = cat_id_to_idx[ann['category_id']]
                    
                    # COCO bbox format: [x_min, y_min, width, height]
                    # YOLO format: [x_center, y_center, width, height] (normalized)
                    x_min, y_min, width, height = ann['bbox']
                    
                    # Convert to YOLO format
                    x_center = (x_min + width / 2) / img_width
                    y_center = (y_min + height / 2) / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    # Write to file: class_idx x_center y_center width height
                    f.write(f"{cat_idx} {x_center} {y_center} {norm_width} {norm_height}")
                    f.write("\n")


def convert_annotations(coco_labels_path='/projects/federated_learning/Hitachi/MSCOCO2017/annotations',
                        yolo_labels_path='/projects/federated_learning/Hitachi/COCO_YOLO_Labels/labels'):
    """
    Creates a YAML configuration file for YOLOv5 training on MSCOCO
    """


    # Create output directories if they don't exist
    yolo_labels_train = f'{yolo_labels_path}/train2017'
    yolo_labels_val = f'{yolo_labels_path}/val2017'
          
    # Convert annotations
    coco_train = COCO(f'{coco_labels_path}/instances_train2017.json')
    coco_val = COCO(f'{coco_labels_path}/instances_val2017.json')
    
    print("Converting training annotations to YOLO format...")
    convert_coco_to_yolo(coco_train, yolo_labels_train)
    
    print("Converting validation annotations to YOLO format...")
    convert_coco_to_yolo(coco_val, yolo_labels_val)
    

if __name__ == "__main__":
    convert_annotations()