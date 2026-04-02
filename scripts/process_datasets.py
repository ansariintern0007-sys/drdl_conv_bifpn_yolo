#!/usr/bin/env python3
"""
Unified Dataset Processing Pipeline for Weld Defect Detection.

Processes 4 heterogeneous datasets into a single unified YOLO-format dataset:
  - RT.v23 (YOLO txt, 5 classes including Wormhole)
  - weld_detection.v1i (YOLO txt, 5 classes)
  - weld_detection.v2i (YOLO txt, 5 classes)
  - SWRD (Labelme JSON + TIF images, Chinese labels)

Output: Unified 5-class YOLO dataset at 940x940 resolution.
  0: Porosity
  1: Crack
  2: Lack_of_Fusion
  3: Lack_of_Penetration
  4: Slag_Inclusion
"""

import os
import sys
import json
import glob
import shutil
import random
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path("/media/aid-pc/My1TB_2/Swin Yolo Model")
OUTPUT_DIR = ROOT_DIR / "dataset"
IMG_SIZE = 640  # Target resolution

# Final unified class schema
UNIFIED_CLASSES = {
    0: "Porosity",
    1: "Crack",
    2: "Lack_of_Fusion",
    3: "Lack_of_Penetration",
    4: "Slag_Inclusion",
}
NUM_CLASSES = len(UNIFIED_CLASSES)

# ============================================================
# CLASS MAPPING PER DATASET
# ============================================================

# RT.v23: ['Clustered Porosity', 'Lack of Fusion', 'Lack of Penetration', 'Single Porosity', 'Wormhole']
RT_V23_MAP = {
    0: 0,     # Clustered Porosity -> Porosity
    1: 2,     # Lack of Fusion -> Lack_of_Fusion
    2: 3,     # Lack of Penetration -> Lack_of_Penetration
    3: 0,     # Single Porosity -> Porosity
    4: None,  # Wormhole -> DROP
}

# v1i & v2i: ['Crack', 'Lack_Of_Fusion', 'Lack_Of_Penetration', 'Porosity', 'Slag_Inclusion']
V1I_V2I_MAP = {
    0: 1,  # Crack -> Crack
    1: 2,  # Lack_Of_Fusion -> Lack_of_Fusion
    2: 3,  # Lack_Of_Penetration -> Lack_of_Penetration
    3: 0,  # Porosity -> Porosity
    4: 4,  # Slag_Inclusion -> Slag_Inclusion
}

# SWRD Chinese label mapping
SWRD_CHINESE_MAP = {
    "气孔": 0,          # Gas Hole / Porosity -> Porosity
    "裂纹": 1,          # Crack -> Crack
    "未熔合": 2,        # Lack of Fusion -> Lack_of_Fusion
    "未焊透": 3,        # Lack of Penetration -> Lack_of_Penetration
    "夹渣": 4,          # Slag Inclusion -> Slag_Inclusion
    # DROP these classes:
    "伪缺陷": None,     # Pseudo-defect
    "焊缝": None,       # Weld Seam
    "咬边": None,       # Undercut
    "夹钨": None,       # Tungsten Inclusion
    "焊瘤": None,       # Weld Bead / Overlap
    "内凹": None,       # Root Concavity
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def letterbox_resize(img, target_size=IMG_SIZE, fill_value=114):
    """Resize image to target_size x target_size with letterbox padding.
    
    Returns: (resized_image, scale, pad_x, pad_y)
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    
    # Create padded image
    canvas = np.full((target_size, target_size, 3) if len(img.shape) == 3 else (target_size, target_size),
                      fill_value, dtype=np.uint8)
    
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    if len(img.shape) == 3:
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = resized
    else:
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return canvas, scale, pad_x, pad_y


def transform_yolo_bbox(bbox, orig_w, orig_h, scale, pad_x, pad_y, target_size=IMG_SIZE):
    """Transform YOLO bbox from original image coords to letterboxed image coords.
    
    Input bbox: [class_id, x_center, y_center, width, height] (normalized to orig image)
    Output bbox: [class_id, x_center, y_center, width, height] (normalized to target_size)
    """
    cls_id, xc, yc, bw, bh = bbox
    
    # Convert from normalized to absolute original coords
    abs_xc = xc * orig_w
    abs_yc = yc * orig_h
    abs_bw = bw * orig_w
    abs_bh = bh * orig_h
    
    # Apply scale and padding
    new_xc = abs_xc * scale + pad_x
    new_yc = abs_yc * scale + pad_y
    new_bw = abs_bw * scale
    new_bh = abs_bh * scale
    
    # Normalize to target_size
    norm_xc = new_xc / target_size
    norm_yc = new_yc / target_size
    norm_bw = new_bw / target_size
    norm_bh = new_bh / target_size
    
    # Clamp to [0, 1]
    norm_xc = max(0.0, min(1.0, norm_xc))
    norm_yc = max(0.0, min(1.0, norm_yc))
    norm_bw = max(0.001, min(1.0, norm_bw))
    norm_bh = max(0.001, min(1.0, norm_bh))
    
    return [cls_id, norm_xc, norm_yc, norm_bw, norm_bh]


def polygon_to_bbox(points):
    """Convert polygon points to bounding box [x_min, y_min, x_max, y_max] in absolute coords."""
    pts = np.array(points, dtype=np.float64)
    x_min = pts[:, 0].min()
    y_min = pts[:, 1].min()
    x_max = pts[:, 0].max()
    y_max = pts[:, 1].max()
    return x_min, y_min, x_max, y_max


def convert_tif_to_png(tif_path, output_path):
    """Convert TIF image to PNG using min-max normalization for 16-bit images."""
    img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    if img.dtype == np.uint16:
        # Min-max normalization for 16-bit
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure 3-channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(str(output_path), img)
    return img


def get_image_hash(img_path):
    """Get hash of image file for deduplication."""
    hasher = hashlib.md5()
    with open(img_path, 'rb') as f:
        buf = f.read(65536)  # Read first 64KB for fast hashing
        hasher.update(buf)
    return hasher.hexdigest()


# ============================================================
# DATASET PROCESSORS
# ============================================================

def process_yolo_dataset(dataset_dir, class_map, dataset_name, staging_dir):
    """Process a YOLO-format dataset (RT.v23, v1i, v2i).
    
    Returns list of (image_path, label_lines, dataset_name) tuples.
    """
    results = []
    
    # Find images and labels
    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"
    
    if not img_dir.exists():
        print(f"  WARNING: {img_dir} not found, skipping.")
        return results
    
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
    
    skipped = 0
    kept = 0
    dropped_annotations = 0
    
    for img_file in tqdm(image_files, desc=f"  Processing {dataset_name}"):
        # Find matching label
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if not lbl_file.exists():
            skipped += 1
            continue
        
        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            skipped += 1
            continue
        orig_h, orig_w = img.shape[:2]
        
        # Letterbox resize
        resized_img, scale, pad_x, pad_y = letterbox_resize(img)
        
        # Read and remap labels
        new_labels = []
        with open(lbl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                old_cls = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Remap class
                new_cls = class_map.get(old_cls, None)
                if new_cls is None:
                    dropped_annotations += 1
                    continue
                
                # Transform bbox to letterboxed coords
                new_bbox = transform_yolo_bbox([new_cls, xc, yc, bw, bh],
                                                orig_w, orig_h, scale, pad_x, pad_y)
                new_labels.append(new_bbox)
        
        if len(new_labels) == 0:
            # Skip images with no valid annotations after filtering
            skipped += 1
            continue
        
        # Save resized image
        out_stem = f"{dataset_name}_{img_file.stem}"
        out_img_path = staging_dir / "images" / f"{out_stem}.jpg"
        cv2.imwrite(str(out_img_path), resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Save labels
        out_lbl_path = staging_dir / "labels" / f"{out_stem}.txt"
        with open(out_lbl_path, 'w') as f:
            for bbox in new_labels:
                f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        results.append((out_img_path, out_lbl_path, dataset_name))
        kept += 1
    
    print(f"  {dataset_name}: kept={kept}, skipped={skipped}, dropped_annotations={dropped_annotations}")
    return results


def process_swrd_dataset(swrd_dir, staging_dir):
    """Process SWRD dataset (Labelme JSON + TIF images).
    
    Returns list of (image_path, label_lines, dataset_name) tuples.
    """
    results = []
    dataset_name = "SWRD"
    
    crop_data = swrd_dir / "crop_weld_data"
    img_base = crop_data / "crop_weld_images"
    json_base = crop_data / "crop_weld_jsons"
    
    # Collect all TIF images and JSON files
    tif_files = []
    for weld_type in ['L', 'T']:
        type_img_dir = img_base / weld_type
        if type_img_dir.exists():
            for subdir in sorted(type_img_dir.iterdir()):
                if subdir.is_dir():
                    for tif_file in subdir.glob("*.tif"):
                        json_file = json_base / weld_type / subdir.name / (tif_file.stem + ".json")
                        if json_file.exists():
                            tif_files.append((tif_file, json_file, weld_type, subdir.name))
    
    skipped = 0
    kept = 0
    dropped_annotations = 0
    conversion_errors = 0
    
    for tif_file, json_file, weld_type, sub_id in tqdm(tif_files, desc="  Processing SWRD"):
        # Convert TIF to temp PNG
        temp_png = staging_dir / "temp" / f"{tif_file.stem}.png"
        img = convert_tif_to_png(tif_file, temp_png)
        if img is None:
            conversion_errors += 1
            continue
        
        orig_h, orig_w = img.shape[:2]
        
        # Read JSON annotations
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue
        
        shapes = data.get("shapes", [])
        
        # Convert polygon annotations to YOLO bboxes
        new_labels = []
        for shape in shapes:
            label = shape.get("label", "")
            points = shape.get("points", [])
            
            if not points or len(points) < 3:
                continue
            
            # Map Chinese label to unified class
            new_cls = SWRD_CHINESE_MAP.get(label, None)
            if new_cls is None:
                dropped_annotations += 1
                continue
            
            # Convert polygon to bbox
            x_min, y_min, x_max, y_max = polygon_to_bbox(points)
            
            # Validate bbox
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            
            # Convert to YOLO format (normalized to original image)
            xc = (x_min + x_max) / 2.0 / orig_w
            yc = (y_min + y_max) / 2.0 / orig_h
            bw = bbox_w / orig_w
            bh = bbox_h / orig_h
            
            new_labels.append([new_cls, xc, yc, bw, bh])
        
        if len(new_labels) == 0:
            skipped += 1
            continue
        
        # Letterbox resize
        resized_img, scale, pad_x, pad_y = letterbox_resize(img)
        
        # Transform all bboxes
        transformed_labels = []
        for bbox in new_labels:
            new_bbox = transform_yolo_bbox(bbox, orig_w, orig_h, scale, pad_x, pad_y)
            transformed_labels.append(new_bbox)
        
        # Save
        out_stem = f"{dataset_name}_{weld_type}{sub_id}_{tif_file.stem}"
        out_img_path = staging_dir / "images" / f"{out_stem}.jpg"
        cv2.imwrite(str(out_img_path), resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        out_lbl_path = staging_dir / "labels" / f"{out_stem}.txt"
        with open(out_lbl_path, 'w') as f:
            for bbox in transformed_labels:
                f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        results.append((out_img_path, out_lbl_path, dataset_name))
        kept += 1
        
        # Clean up temp
        if temp_png.exists():
            temp_png.unlink()
    
    print(f"  SWRD: kept={kept}, skipped={skipped}, dropped_annotations={dropped_annotations}, conversion_errors={conversion_errors}")
    return results


# ============================================================
# DATASET SPLIT
# ============================================================

def split_dataset(all_samples, output_dir):
    """Split samples into train/val/test with stratified sampling.
    
    Prevents data leakage by splitting at the image level (not annotation level).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Group by source dataset for stratified split
    dataset_groups = defaultdict(list)
    for img_path, lbl_path, ds_name in all_samples:
        dataset_groups[ds_name].append((img_path, lbl_path))
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for ds_name, samples in dataset_groups.items():
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Rest goes to test
        
        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])
    
    # Shuffle within each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
    }
    
    for split_name, samples in splits.items():
        for img_src, lbl_src in tqdm(samples, desc=f"  Moving to {split_name}"):
            img_dst = output_dir / "images" / split_name / Path(img_src).name
            lbl_dst = output_dir / "labels" / split_name / Path(lbl_src).name
            shutil.move(str(img_src), str(img_dst))
            shutil.move(str(lbl_src), str(lbl_dst))
    
    return {k: len(v) for k, v in splits.items()}


# ============================================================
# DATA.YAML GENERATION
# ============================================================

def generate_data_yaml(output_dir, split_counts):
    """Generate YOLO-compatible data.yaml."""
    yaml_content = f"""# Unified Weld Defect Detection Dataset
# Generated by process_datasets.py
# Total: {sum(split_counts.values())} images ({split_counts['train']} train / {split_counts['val']} val / {split_counts['test']} test)

path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: {NUM_CLASSES}
names: {list(UNIFIED_CLASSES.values())}
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n  data.yaml written to {yaml_path}")


# ============================================================
# VERIFICATION
# ============================================================

def verify_dataset(output_dir):
    """Verify dataset integrity."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    issues = []
    class_counts = Counter()
    total_annotations = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        
        if not img_dir.exists():
            issues.append(f"Missing directory: {img_dir}")
            continue
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        
        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in labels}
        
        # Check for mismatched pairs
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        
        if missing_labels:
            issues.append(f"[{split}] {len(missing_labels)} images missing labels")
        if missing_images:
            issues.append(f"[{split}] {len(missing_images)} labels missing images")
        
        # Check label contents
        for lbl_file in labels:
            with open(lbl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"[{split}] {lbl_file.name}:{line_num} has {len(parts)} fields (expected 5)")
                        continue
                    
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id >= NUM_CLASSES:
                        issues.append(f"[{split}] {lbl_file.name}:{line_num} invalid class_id={cls_id}")
                        continue
                    
                    coords = [float(p) for p in parts[1:]]
                    for i, c in enumerate(coords):
                        if c < 0 or c > 1.001:
                            issues.append(f"[{split}] {lbl_file.name}:{line_num} coord[{i}]={c} out of [0,1]")
                    
                    class_counts[cls_id] += 1
                    total_annotations += 1
        
        print(f"  {split}: {len(images)} images, {len(labels)} labels")
    
    print(f"\n  Total annotations: {total_annotations}")
    print(f"\n  Class distribution:")
    for cls_id in sorted(class_counts.keys()):
        name = UNIFIED_CLASSES.get(cls_id, "UNKNOWN")
        count = class_counts[cls_id]
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        print(f"    {cls_id}: {name:25s} = {count:6d} ({pct:.1f}%)")
    
    if issues:
        print(f"\n  ⚠️  {len(issues)} issues found:")
        for issue in issues[:20]:
            print(f"    - {issue}")
        return False
    else:
        print(f"\n  ✅ Dataset verification PASSED — no issues found.")
        return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Weld Defect Dataset Processing")
    parser.add_argument('--verify-only', action='store_true', help="Only verify existing dataset")
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.verify_only:
        verify_dataset(output_dir)
        return
    
    print("=" * 60)
    print("WELD DEFECT DETECTION — DATASET PROCESSING PIPELINE")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Classes: {NUM_CLASSES} — {list(UNIFIED_CLASSES.values())}")
    print(f"  Split: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print()
    
    # Clean output directory
    # Clean output directory
    if output_dir.exists():
        print("  Removing existing output directory...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create staging directory
    staging_dir = output_dir / "_staging"
    (staging_dir / "images").mkdir(parents=True, exist_ok=True)
    (staging_dir / "labels").mkdir(parents=True, exist_ok=True)
    (staging_dir / "temp").mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # --- Process RT.v23 ---
    print("\n[1/4] Processing RT.v23...")
    rt_dir = ROOT_DIR / "RT.v23-final.yolov11(RENAME LATER)"
    samples = process_yolo_dataset(rt_dir, RT_V23_MAP, "RTv23", staging_dir)
    all_samples.extend(samples)
    
    # --- Process v1i ---
    print("\n[2/4] Processing weld_detection.v1i...")
    v1i_dir = ROOT_DIR / "weld_detection.v1i.yolov11(DONE)"
    samples = process_yolo_dataset(v1i_dir, V1I_V2I_MAP, "v1i", staging_dir)
    all_samples.extend(samples)
    
    # --- Process v2i ---
    print("\n[3/4] Processing weld_detection.v2i...")
    v2i_dir = ROOT_DIR / "weld_detection.v2i.yolov11(DONE)"
    samples = process_yolo_dataset(v2i_dir, V1I_V2I_MAP, "v2i", staging_dir)
    all_samples.extend(samples)
    
    # --- Process SWRD ---
    print("\n[4/4] Processing SWRD...")
    swrd_dir = ROOT_DIR / "SWRD_Data"
    samples = process_swrd_dataset(swrd_dir, staging_dir)
    all_samples.extend(samples)
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL VALID SAMPLES: {len(all_samples)}")
    print(f"{'=' * 60}")
    
    # --- Split dataset ---
    print("\nSplitting dataset (70/20/10)...")
    split_counts = split_dataset(all_samples, output_dir)
    print(f"  Train: {split_counts['train']}")
    print(f"  Val:   {split_counts['val']}")
    print(f"  Test:  {split_counts['test']}")
    
    # Cleanup staging
    staging_images = staging_dir / "images"
    staging_labels = staging_dir / "labels"
    staging_temp = staging_dir / "temp"
    for d in [staging_images, staging_labels, staging_temp, staging_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    
    # --- Generate data.yaml ---
    print("\nGenerating data.yaml...")
    generate_data_yaml(output_dir, split_counts)
    
    # --- Verify ---
    verify_dataset(output_dir)
    
    print(f"\n✅ Dataset processing complete!")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
