import os
import json
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# =========================
# Path configuration
# =========================
json_path = "coco/annotations/instances_val2017_filter.json"
src_image_dir = "coco/val2017"

dst_image_dir = "coco/image"
dst_mask_dir = "coco/mask_fine"
dst_visual_dir = "coco/visual_fine"

# Blue overlay with 55% opacity
overlay_alpha = int(255 * 0.55)

# =========================
# Create directories
# =========================
os.makedirs(dst_image_dir, exist_ok=True)
os.makedirs(dst_mask_dir, exist_ok=True)
os.makedirs(dst_visual_dir, exist_ok=True)

# =========================
# Read JSON
# =========================
with open(json_path, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# =========================
# Build indexes
# =========================
# image_id -> image_info
image_info_dict = {img["id"]: img for img in images}

# image_id -> annotations
anns_by_image = defaultdict(list)
for ann in annotations:
    anns_by_image[ann["image_id"]].append(ann)

# =========================
# Statistics: annotation level
# =========================
rle_ann_count = 0
polygon_ann_count = 0
invalid_seg_count = 0

# Image level: if an image contains at least one RLE annotation, mark it as an RLE image
rle_image_ids = set()

for ann in annotations:
    seg = ann.get("segmentation", None)

    if isinstance(seg, dict):
        rle_ann_count += 1
        rle_image_ids.add(ann["image_id"])
    elif isinstance(seg, list):
        polygon_ann_count += 1
    else:
        invalid_seg_count += 1

# =========================
# Statistics: image level
# =========================
all_image_ids = set(anns_by_image.keys())
non_rle_image_ids = all_image_ids - rle_image_ids

print("\n========== Dataset Statistics ==========")
print("[Annotation-level]")
print(f"Total annotations: {len(annotations)}")
print(f"Polygon annotations: {polygon_ann_count}")
print(f"RLE annotations: {rle_ann_count}")
print(f"Invalid/None segmentation annotations: {invalid_seg_count}")

if len(annotations) > 0:
    print(f"Polygon annotation ratio: {polygon_ann_count / len(annotations):.4f}")
    print(f"RLE annotation ratio: {rle_ann_count / len(annotations):.4f}")

print("\n[Image-level]")
print(f"Total images with annotations: {len(all_image_ids)}")
print(f"Images containing RLE: {len(rle_image_ids)}")
print(f"Images without RLE (usable): {len(non_rle_image_ids)}")

if len(all_image_ids) > 0:
    print(f"RLE image ratio: {len(rle_image_ids) / len(all_image_ids):.4f}")
    print(f"Non-RLE image ratio: {len(non_rle_image_ids) / len(all_image_ids):.4f}")

print("========================================\n")

# =========================
# Utility functions
# =========================
def build_mask_for_image(image_info, ann_list):
    """
    Build a binary mask for the entire image:
    - White (255): annotation regions
    - Black (0): background

    Note:
    This function assumes that images containing RLE annotations
    have already been filtered out externally, so it only handles polygons.
    """
    width = image_info["width"]
    height = image_info["height"]

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for ann in ann_list:
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        if isinstance(seg, list):
            for poly in seg:
                if not poly or len(poly) < 6:
                    continue
                polygon = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                draw.polygon(polygon, fill=255)

    return mask_img


def create_visual_overlay(original_img, mask_img, alpha=140):
    """
    Overlay the mask regions on the original image with semi-transparent blue
    """
    original_rgba = original_img.convert("RGBA")
    width, height = original_rgba.size

    overlay = Image.new("RGBA", (width, height), (0, 0, 255, 0))

    mask_arr = np.array(mask_img, dtype=np.uint8)
    overlay_arr = np.array(overlay, dtype=np.uint8)

    overlay_arr[mask_arr > 0] = [0, 0, 255, alpha]

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    visual = Image.alpha_composite(original_rgba, overlay)
    return visual


# =========================
# Main process
# =========================
all_image_ids = list(anns_by_image.keys())

print(f"Start processing {len(all_image_ids)} images with annotations...")

copy_warn_count = 0
mask_warn_count = 0
visual_warn_count = 0
missing_src_count = 0
skipped_rle_image_count = 0
processed_image_count = 0

for image_id in tqdm(all_image_ids, desc="Processing images"):
    if image_id not in image_info_dict:
        print(f"Warning: image_id {image_id} not found in images")
        continue

    # =========================
    # First check whether the image contains RLE annotations
    # If it contains any RLE annotation, skip the entire image directly
    # =========================
    if image_id in rle_image_ids:
        skipped_rle_image_count += 1
        continue

    image_info = image_info_dict[image_id]
    file_name = image_info["file_name"]

    src_path = os.path.join(src_image_dir, file_name)
    dst_img_path = os.path.join(dst_image_dir, file_name)

    if not os.path.exists(src_path):
        print(f"Warning: source image not found: {src_path}")
        missing_src_count += 1
        continue

    # 1. Copy the original image
    if os.path.exists(dst_img_path):
        print(f"[Warning] Image already exists, skip copy: {dst_img_path}")
        copy_warn_count += 1
    else:
        shutil.copy2(src_path, dst_img_path)

    # Read the original image
    original_img = Image.open(src_path).convert("RGB")

    # 2. Generate the mask
    mask_img = build_mask_for_image(image_info, anns_by_image[image_id])

    stem = os.path.splitext(file_name)[0]
    mask_save_path = os.path.join(dst_mask_dir, f"{stem}_mask_fine.png")

    if os.path.exists(mask_save_path):
        print(f"[Warning] Mask already exists, skip save: {mask_save_path}")
        mask_warn_count += 1
    else:
        mask_img.save(mask_save_path)

    # 3. Generate the visualization image
    visual_img = create_visual_overlay(original_img, mask_img, alpha=overlay_alpha)
    visual_save_path = os.path.join(dst_visual_dir, f"{stem}_visual_fine.png")

    if os.path.exists(visual_save_path):
        print(f"[Warning] Visual already exists, skip save: {visual_save_path}")
        visual_warn_count += 1
    else:
        visual_img.save(visual_save_path)

    processed_image_count += 1

# =========================
# Final statistics
# =========================
print("\nDone.")
print("========== Processing Summary ==========")
print(f"Processed images: {processed_image_count}")
print(f"Skipped images containing RLE: {skipped_rle_image_count}")
print(f"Image copy warnings: {copy_warn_count}")
print(f"Mask save warnings: {mask_warn_count}")
print(f"Visual save warnings: {visual_warn_count}")
print(f"Missing source images: {missing_src_count}")
print("========================================")