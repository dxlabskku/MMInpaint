import json
import statistics
from tqdm import tqdm

json_path = "coco/annotations/instances_val2017.json"
save_path = "coco/annotations/instances_val2017_filter.json"

with open(json_path, "r") as f:
    data = json.load(f)

# image_id -> image_info
image_info_dict = {
    img["id"]: (img["width"], img["height"])
    for img in data["images"]
}

ratio_list = []
area_list = []

# New: save the filtered annotations
filtered_annotations = []
kept_image_ids = set()

# Statistics
total_count = 0
filtered_count = 0
seg_filtered_count = 0

for ann in tqdm(data["annotations"], desc="Filtering annotations"):
    total_count += 1

    image_id = ann["image_id"]

    if image_id not in image_info_dict:
        continue

    width, height = image_info_dict[image_id]
    image_area = width * height

    if image_area == 0:
        continue

    ann_area = ann["area"]
    ratio = ann_area / image_area

    # ===== Segmentation filtering =====
    seg = ann.get("segmentation", None)

    if isinstance(seg, list):
        if len(seg) > 3:
            seg_filtered_count += 1
            continue

    # ===== Original filtering rules =====
    if ratio < 0.05 or ratio > 0.7 or ann_area < 500:
        continue

    # Keep the annotation
    filtered_count += 1
    filtered_annotations.append(ann)   # Do not modify the original structure
    kept_image_ids.add(image_id)

    ratio_list.append(ratio)
    area_list.append(ann_area)


# ===== Filter images: keep only images with annotations =====
filtered_images = [
    img for img in data["images"] if img["id"] in kept_image_ids
]


# ===== Construct a new COCO JSON file =====
new_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": data["categories"]
}


# ===== Save =====
with open(save_path, "w") as f:
    json.dump(new_data, f)

print(f"\nSaved cleaned dataset to: {save_path}")


# ===== Statistics function =====
def print_stats(name, data_list):
    if len(data_list) == 0:
        print(f"{name}: No data")
        return

    print(f"\n=== {name} Statistics ===")
    print(f"Count: {len(data_list)}")
    print(f"Max: {max(data_list):.6f}")
    print(f"Min: {min(data_list):.6f}")
    print(f"Mean: {sum(data_list) / len(data_list):.6f}")
    print(f"Median: {statistics.median(data_list):.6f}")


# ===== Output =====
print("\n===== Filtering Summary =====")
print(f"Total annotations: {total_count}")
print(f"Removed by segmentation (>3): {seg_filtered_count}")
print(f"Remaining after filtering: {filtered_count}")
print(f"Removed total: {total_count - filtered_count}")
print(f"Keep ratio: {filtered_count / total_count:.4f}")

print_stats("Ratio (Filtered)", ratio_list)
print_stats("Area (Filtered)", area_list)