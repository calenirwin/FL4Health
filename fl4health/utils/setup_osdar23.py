import os
import yaml
from pathlib import Path
import json
import random
from PIL import Image

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
BASE_DIR = Path("OSDAR23")
CLASSES = [
    'person', 'crowd', 'train', 'wagons', 'bicycle',
    'group of bicycles', 'motorcycle', 'road vehicle',
    'animal', 'group of animals', 'wheelchair',
    'drag shoe', 'track', 'transition', 'switch',
    'catenary pole', 'signal pole', 'signal',
    'signal bridge', 'buffer stop', 'flame', 'smoke'
]

# -------------------------------------------------------------------
# 2. Create Directory Structure
# -------------------------------------------------------------------
def create_dirs():
    """Create YOLO directory structure"""
    for split in ['train', 'val', 'test']:
        (BASE_DIR / 'osdar23_yolo' / 'images' / split).mkdir(parents=True, exist_ok=True)
        (BASE_DIR / 'osdar23_yolo' / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print("Created directory structure")

# -------------------------------------------------------------------
# 3. Create Class File
# -------------------------------------------------------------------
def create_class_file():
    """Generate class definition file"""
    with open(BASE_DIR / "osdar23_classes.yml", 'w') as f:
        yaml.dump({'names': CLASSES}, f)
    print("Created class file")

# -------------------------------------------------------------------
# 4. Annotation Conversion (WITH FALLBACK LOGIC)
# -------------------------------------------------------------------
def convert_annotations():
    """Convert annotations with comprehensive validation"""
    # Path configuration
    json_path = BASE_DIR / "4_station_pedestrian_bridge_4.1_labels.json"
    image_dir = BASE_DIR / "rgb_center"
    output_dir = BASE_DIR / "osdar23_yolo"
    camera_stream = "rgb_center"

    # Validate paths
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Load class mapping
    with open(BASE_DIR / "osdar23_classes.yml") as f:
        class_mapping = {name: idx for idx, name in enumerate(yaml.safe_load(f)['names'])}
    print(f"Loaded class mapping: {class_mapping}")

    # Load annotations
    with open(json_path) as f:
        data = json.load(f)
    frames = list(data['openlabel']['frames'].items())
    print(f"Found {len(frames)} frames in dataset")

    # Split frames
    random.shuffle(frames)
    train_end = int(len(frames) * 0.7)
    val_end = train_end + int(len(frames) * 0.2)
    splits = {
        'train': frames[:train_end],
        'val': frames[train_end:val_end],
        'test': frames[val_end:]
    }

    # Conversion metrics
    conversion_stats = {
        'total_frames': 0,
        'frames_with_labels': 0,
        'total_objects': 0,
        'valid_objects': 0,
        'fallback_used': 0
    }

    # Process splits
    for split_name, split_frames in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_frames)} frames)")
        
        for frame_id, frame_data in split_frames:
            conversion_stats['total_frames'] += 1
            try:
                # Get image metadata
                img_info = frame_data['frame_properties']['streams'][camera_stream]
                img_name = Path(img_info['uri']).name
                src_img = image_dir / img_name
                
                if not src_img.exists():
                    print(f"Missing image: {src_img}")
                    continue

                # Create symlink
                dest_img = output_dir / 'images' / split_name / img_name
                dest_img.parent.mkdir(exist_ok=True)
                if not dest_img.exists():
                    os.symlink(src_img.resolve(), dest_img)

                # Prepare label file
                label_file = output_dir / 'labels' / split_name / f"{src_img.stem}.txt"
                with Image.open(src_img) as img:
                    img_w, img_h = img.size

                # Process objects
                valid_objects = []
                objects = frame_data.get('objects', {})
                conversion_stats['total_objects'] += len(objects)
                
                for obj_id, obj in objects.items():
                    obj_data = obj.get('object_data', {})
                    obj_type = None
                    
                    # Attempt 1: Standard type field
                    obj_type = obj_data.get('type')
                    
                    # Attempt 2: Extract from bbox name
                    if not obj_type:
                        for bbox in obj_data.get('bbox', []):
                            if '__bbox__' in bbox.get('name', ''):
                                obj_type = bbox['name'].split('__')[-1]
                                conversion_stats['fallback_used'] += 1
                                break
                    
                    # Attempt 3: Use object ID as last resort
                    if not obj_type:
                        obj_type = obj_id
                        conversion_stats['fallback_used'] += 1
                        print(f"Using object ID as class: {obj_id}")

                    if not obj_type:
                        print(f"Unidentifiable object: {json.dumps(obj, indent=2)}")
                        continue
                        
                    if obj_type not in class_mapping:
                        print(f"Unknown class '{obj_type}' in frame {frame_id}")
                        continue

                    # Extract valid bboxes
                    bboxes = [
                        b for b in obj_data.get('bbox', [])
                        if b.get('coordinate_system') == camera_stream
                        and len(b.get('val', [])) == 4
                    ]
                    
                    for bbox in bboxes:
                        try:
                            x_min, y_min, width, height = map(float, bbox['val'])
                            valid_objects.append({
                                'class': class_mapping[obj_type],
                                'x_center': (x_min + width/2) / img_w,
                                'y_center': (y_min + height/2) / img_h,
                                'width': width / img_w,
                                'height': height / img_h
                            })
                            conversion_stats['valid_objects'] += 1
                        except (TypeError, ValueError) as e:
                            print(f"Invalid bbox values in {obj_id}: {e}")

                # Write labels
                if valid_objects:
                    conversion_stats['frames_with_labels'] += 1
                    with open(label_file, 'w') as f:
                        for obj in valid_objects:
                            f.write(
                                f"{obj['class']} "
                                f"{obj['x_center']:.6f} {obj['y_center']:.6f} "
                                f"{obj['width']:.6f} {obj['height']:.6f}\n"
                            )
                else:
                    print(f"No valid objects in frame {frame_id}")
                    label_file.unlink(missing_ok=True)

            except KeyError as e:
                print(f"Key error in frame {frame_id}: {str(e)}")
            except Exception as e:
                print(f"Error processing frame {frame_id}: {str(e)}")

    # Print conversion statistics
    print("\nConversion Statistics:")
    print(f"- Total frames processed: {conversion_stats['total_frames']}")
    print(f"- Frames with labels: {conversion_stats['frames_with_labels']}")
    print(f"- Total objects found: {conversion_stats['total_objects']}")
    print(f"- Valid objects converted: {conversion_stats['valid_objects']}")
    print(f"- Fallback class usages: {conversion_stats['fallback_used']}")

# -------------------------------------------------------------------
# 5. Create YOLO Config
# -------------------------------------------------------------------
def create_yolo_config():
    """Generate dataset configuration file"""
    config = f"""
path: {BASE_DIR.resolve()}/osdar23_yolo
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
    """
    with open(BASE_DIR / "osdar23_yolo" / "dataset.yaml", 'w') as f:
        f.write(config.strip())
    print("Created dataset configuration")

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting OSDAR23 Conversion")
    create_dirs()
    create_class_file()
    convert_annotations()
    create_yolo_config()
    print("Conversion completed successfully")