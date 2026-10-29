import os
import json
import random
from typing import Dict, List, Any
from PIL import Image
from tqdm import tqdm

# Category pool for object insertion
ITEMS: List[str] = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def _sample_template(templates: List[str]) -> str:
    """Randomly select one sentence template from several similar expressions."""
    return random.choice(templates)


# =========================
# Foreground editing: 0 / 1 / 2 / 3 / 4
# Background editing: 0 / 5 / 6
# =========================

def prompt_0_no_prompt() -> str:
    """
    0 — No prompt
    """
    return ""


def prompt_1_object_replacement() -> str:
    """
    1 — Object Replacement
    """
    templates = [
        "Replace the masked object with another object naturally.",
        "Substitute the object in the inpainting region with a different object.",
        "Change the masked object into another object while keeping the result realistic."
    ]
    return _sample_template(templates)


def prompt_2_object_removal() -> str:
    """
    2 — Object Removal (Context-Aware Fill)
    """
    templates = [
        "Remove the content in the masked region and fill it naturally to match the surroundings.",
        "Erase the object in the inpainting area and complete the region with context-aware content.",
        "Delete the masked content and seamlessly reconstruct the area based on the nearby scene."
    ]
    return _sample_template(templates)


def prompt_3_attribute_modification() -> str:
    """
    3 — Attribute / Appearance Modification
    """
    templates = [
        "Modify the appearance in the masked region by changing its color, material, or style in a natural way.",
        "Adjust the masked content to have a different color, texture, or visual style while keeping it realistic.",
        "Change the visual attributes in the inpainting area, such as making it black or giving it a different material or style."
    ]
    return _sample_template(templates)


def prompt_4_object_insertion_foreground(item_name: str = None) -> str:
    """
    4 — Object Insertion (Adding New Object to Empty Region)
    """
    if item_name is None:
        item_name = random.choice(ITEMS)

    templates = [
        f"Add a {item_name} inside the inpainting region naturally.",
        f"Insert a {item_name} into the masked area in a realistic way.",
        f"Naturally place a {item_name} within the selected region."
    ]
    return _sample_template(templates)


def prompt_5_background_replacement() -> str:
    """
    5 — Background Replacement
    """
    templates = [
        "Replace the background in the masked region with a different background naturally.",
        "Change the inpainting background area to another background while keeping the scene coherent.",
        "Substitute the masked background with a new background that blends naturally with the image."
    ]
    return _sample_template(templates)


def prompt_6_object_insertion_background(item_name: str = None) -> str:
    """
    6 — Object Insertion
    """
    if item_name is None:
        item_name = random.choice(ITEMS)

    templates = [
        f"Add a {item_name} naturally in the background region.",
        f"Insert a {item_name} into the masked background area in a realistic way.",
        f"Naturally place a {item_name} in the background within the selected region."
    ]
    return _sample_template(templates)


# =========================
# Find the original image directly by image_id and read its area.
# This does not depend on the images field.
# =========================

def build_image_info_map_from_annotations(
    annotations: List[Dict[str, Any]],
    image_root: str
) -> Dict[int, Dict[str, Any]]:
    """
    Find the width, height, and area of each image.

    Returns:
    {
        image_id: {
            "path": ...,
            "width": ...,
            "height": ...,
            "image_area": ...
        }
    }
    """
    image_info_map: Dict[int, Dict[str, Any]] = {}

    unique_image_ids = {ann["image_id"] for ann in annotations}

    for image_id in tqdm(unique_image_ids, desc="Building image_info_map"):
        image_path = os.path.join(image_root, f"{image_id:012d}.jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Original image not found: {image_path}")

        with Image.open(image_path) as img:
            width, height = img.size

        image_info_map[image_id] = {
            "path": image_path,
            "width": width,
            "height": height,
            "image_area": width * height
        }

    return image_info_map


# =========================
# Generate foreground / background prompts according to each annotation.
# =========================

def generate_inpainting_prompt(
    annotation: Dict[str, Any],
    image_info_map: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Input a single annotation and output foreground_prompt and background_prompt.

    Logic:
    1. mask_area_ratio = annotation["area"] / image_area
    2. Foreground:
       - If mask_area_ratio >= 0.5:
         randomly select one mode from [0, 1, 2, 3, 4]
       - Otherwise:
         randomly select one mode from [0, 1, 2, 3]
    3. Background:
       - If mask_area_ratio < 0.5:
         randomly select one mode from [0, 5, 6]
       - Otherwise:
         randomly select one mode from [0, 5]
    """
    image_id = annotation["image_id"]

    if image_id not in image_info_map:
        raise KeyError(f"image_id={image_id} does not exist in image_info_map.")

    image_info = image_info_map[image_id]
    image_area = image_info["image_area"]
    mask_area = float(annotation.get("area", 0.0))

    if image_area <= 0:
        raise ValueError(f"Invalid image_area for image_id={image_id}: {image_area}")

    mask_area_ratio = mask_area / image_area

    # -------- Foreground prompt mode --------
    if mask_area_ratio >= 0.5:
        foreground_mode = random.choice([0, 1, 2, 3, 4])
    else:
        foreground_mode = random.choice([0, 1, 2, 3])

    if foreground_mode == 0:
        foreground_prompt = prompt_0_no_prompt()
    elif foreground_mode == 1:
        foreground_prompt = prompt_1_object_replacement()
    elif foreground_mode == 2:
        foreground_prompt = prompt_2_object_removal()
    elif foreground_mode == 3:
        foreground_prompt = prompt_3_attribute_modification()
    elif foreground_mode == 4:
        foreground_prompt = prompt_4_object_insertion_foreground()
    else:
        raise ValueError(f"Invalid foreground_mode: {foreground_mode}")

    # -------- Background prompt mode --------
    if mask_area_ratio < 0.5:
        background_mode = random.choice([0, 5, 6])
    else:
        background_mode = random.choice([0, 5])

    if background_mode == 0:
        background_prompt = prompt_0_no_prompt()
    elif background_mode == 5:
        background_prompt = prompt_5_background_replacement()
    elif background_mode == 6:
        background_prompt = prompt_6_object_insertion_background()
    else:
        raise ValueError(f"Invalid background_mode: {background_mode}")

    return {
        "foreground_mode": foreground_mode,
        "foreground_prompt": foreground_prompt,
        "background_mode": background_mode,
        "background_prompt": background_prompt,
        "mask_area_ratio": mask_area_ratio
    }


# =========================
# Batch process COCO json.
# =========================

def augment_coco_annotations_with_prompts(
    input_json_path: str,
    output_json_path: str,
    image_root: str = "coco/image"
) -> Dict[str, Any]:
    """
    Read a COCO json file and process only the annotations:
    - For each annotation, find the original image according to image_id.
    - Read the original image size.
    - Compute mask_area_ratio = annotation["area"] / original image area.
    - Generate foreground / background prompts.

    Finally, write the new fields back into each annotation:
    - foreground_mode
    - foreground_prompt
    - background_mode
    - background_prompt
    - mask_area_ratio
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    if "annotations" not in coco_data:
        raise ValueError("The input json does not contain an 'annotations' field.")

    annotations = coco_data["annotations"]

    image_info_map = build_image_info_map_from_annotations(
        annotations=annotations,
        image_root=image_root
    )

    new_annotations = []
    for ann in tqdm(annotations, desc="Generating prompts for annotations"):
        prompt_info = generate_inpainting_prompt(ann, image_info_map)

        ann_new = dict(ann)
        ann_new.update(prompt_info)
        new_annotations.append(ann_new)

    coco_data["annotations"] = new_annotations

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    return coco_data


# =========================
# Example
# =========================

if __name__ == "__main__":
    input_json = "coco/annotations/instances_val2017_filter.json"
    output_json = "coco/annotations/instances_val2017_filter_prompt.json"
    image_root = "coco/image"

    augment_coco_annotations_with_prompts(
        input_json_path=input_json,
        output_json_path=output_json,
        image_root=image_root
    )

    print(f"Done. Saved to: {output_json}")