import json
import os
from collections import defaultdict

def has_sufficient_y_overlap(box1, box2, threshold=0.4):
    """Returns True if Y-overlap is at least threshold fraction of smaller height"""
    overlap = get_y_overlap(box1, box2)
    min_height = min(box1['h'], box2['h'])
    return overlap >= threshold * min_height

def boxes_overlap_adjusted(box1, box2):
    adjusted_box2 = {
        "x": box2["x"],
        "y": box2["y"] + 10,
        "w": box2["w"],
        "h": box2["h"] - 10
    }
    return (
        box1["x"] < adjusted_box2["x"] + adjusted_box2["w"] and
        box1["x"] + box1["w"] > adjusted_box2["x"] and
        box1["y"] < adjusted_box2["y"] + adjusted_box2["h"] and
        box1["y"] + box1["h"] > adjusted_box2["y"]
    )

def is_x_overlap(box1, box2):
    return not (box1["x"] + box1["w"] <= box2["x"] or box2["x"] + box2["w"] <= box1["x"])

def get_y_overlap(box1, box2):
    y_top = max(box1['y'], box2['y'])
    y_bottom = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    return max(0, y_bottom - y_top)

def select_best_neighbor(region, candidates):
    selected_neighbors = []
    while candidates:
        candidate = candidates.pop()
        overlapping_group = [candidate] + [r for r in candidates if is_x_overlap(r["bounding_box"], candidate["bounding_box"])]
        for r in overlapping_group[1:]:
            candidates.remove(r)
        best_neighbor = max(overlapping_group, key=lambda r: get_y_overlap(r["bounding_box"], region["bounding_box"]))
        selected_neighbors.append(best_neighbor)
    return selected_neighbors

def has_sufficient_y_overlap(box1, box2, threshold=0.5):
    overlap = get_y_overlap(box1, box2)
    min_height = min(box1["h"], box2["h"])
    return min_height > 0 and (overlap / min_height) >= threshold

def assign_orders_based_on_neighbors(union_data):
    for image_entry in union_data:
        regions = image_entry["regions"]

        for region in regions:
            if region.get("order", 0) == 0:
                box1 = region["bounding_box"]
                x1 = box1["x"]

                L = [
                    r for r in regions
                    if r != region and has_sufficient_y_overlap(r["bounding_box"], box1, threshold=0.5)
                ]

                L1 = [r for r in L if r["bounding_box"]["x"] < x1 and r.get("order", 0) > 0]
                L2 = [r for r in L if r["bounding_box"]["x"] > x1 and r.get("order", 0) > 0]

                excluded_L1 = [r for r in L1 if is_x_overlap(r["bounding_box"], box1)]
                excluded_L2 = [r for r in L2 if is_x_overlap(r["bounding_box"], box1)]
                L1_filtered = [r for r in L1 if r not in excluded_L1]
                L2_filtered = [r for r in L2 if r not in excluded_L2]

                selected_L1 = select_best_neighbor(region, L1_filtered)
                selected_L2 = select_best_neighbor(region, L2_filtered)

                if selected_L1:
                    selected_L1.sort(key=lambda r: abs(r["bounding_box"]["x"] + r["bounding_box"]["w"] - x1))
                if selected_L2:
                    selected_L2.sort(key=lambda r: abs(r["bounding_box"]["x"] - (x1 + box1["w"])))

                chosen_neighbor = None
                if selected_L1 and selected_L2:
                    dist_left = abs((selected_L1[0]['bounding_box']['x'] + selected_L1[0]['bounding_box']['w']) - x1)
                    dist_right = abs(selected_L2[0]['bounding_box']['x'] - (x1 + box1["w"]))
                    chosen_neighbor = selected_L1[0] if dist_left < dist_right else selected_L2[0]
                elif selected_L1:
                    chosen_neighbor = selected_L1[0]
                elif selected_L2:
                    chosen_neighbor = selected_L2[0]

                if chosen_neighbor:
                    region["order"] = chosen_neighbor["order"]
                    region["line"] = chosen_neighbor.get("line", -1)

def yx_sort(regions):
    return sorted(regions, key=lambda r: (r["bounding_box"]["y"], r["bounding_box"]["x"]))

def order_regions_like_2json(ref_regions, target_regions):
    ref_sorted = yx_sort(ref_regions)
    target_sorted = yx_sort(target_regions)
    for i, region in enumerate(target_sorted):
        region["order"] = ref_sorted[i % len(ref_sorted)]["order"]
        region["line"] = ref_sorted[i % len(ref_sorted)].get("line", -1)
    return target_sorted

def remove_y_overlaps(regions):
    non_overlapping = []
    for i, r1 in enumerate(regions):
        y1, h1 = r1["bounding_box"]["y"], r1["bounding_box"]["h"]
        y2 = y1 + h1
        overlaps = False
        for j, r2 in enumerate(regions):
            if i == j:
                continue
            ry1, rh1 = r2["bounding_box"]["y"], r2["bounding_box"]["h"]
            ry2 = ry1 + rh1
            if not (ry2 < y1 or ry1 > y2):
                overlaps = True
                break
        if not overlaps:
            non_overlapping.append(r1)
    return non_overlapping

def remove_smaller_overlapping_regions(union_data):
    def get_intersection_area(b1, b2):
        x_left = max(b1["x"], b2["x"])
        y_top = max(b1["y"], b2["y"])
        x_right = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
        y_bottom = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
        if x_right <= x_left or y_bottom <= y_top:
            return 0
        return (x_right - x_left) * (y_bottom - y_top)

    for image_entry in union_data:
        regions = image_entry["regions"]
        keep_flags = [True] * len(regions)
        for i in range(len(regions)):
            if not keep_flags[i]:
                continue
            box1 = regions[i]["bounding_box"]
            area1 = box1["w"] * box1["h"]
            for j in range(i + 1, len(regions)):
                if not keep_flags[j]:
                    continue
                box2 = regions[j]["bounding_box"]
                area2 = box2["w"] * box2["h"]
                inter_area = get_intersection_area(box1, box2)
                if inter_area == 0:
                    continue
                if area1 < area2:
                    smaller_idx = i
                else:
                    smaller_idx = j
                if inter_area / min(area1, area2) > 0.5:
                    keep_flags[smaller_idx] = False
                    if smaller_idx == i:
                        break
        image_entry["regions"] = [r for k, r in enumerate(regions) if keep_flags[k]]

def get_vertical_overlap_fraction(box1, box2):
    y_top = max(box1["y"], box2["y"])
    y_bottom = min(box1["y"] + box1["h"], box2["y"] + box2["h"])
    overlap = max(0, y_bottom - y_top)
    min_height = min(box1["h"], box2["h"])
    if min_height == 0:
        return 0
    return overlap / min_height

def resolve_duplicate_orders(union_data):
    def y_overlap_fraction(box1, box2):
        y_top = max(box1['y'], box2['y'])
        y_bottom = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
        overlap_height = max(0, y_bottom - y_top)
        min_height = min(box1['h'], box2['h'])
        return overlap_height / min_height if min_height > 0 else 0

    for image_entry in union_data:
        regions = image_entry["regions"]
        regions.sort(key=lambda r: r.get("order", 0))

        new_regions = []
        i = 0
        current_order = 0

        while i < len(regions):
            base_order = regions[i]["order"]
            duplicate_group = [regions[i]]
            i += 1
            while i < len(regions) and regions[i]["order"] == base_order:
                duplicate_group.append(regions[i])
                i += 1

            if len(duplicate_group) == 1:
                duplicate_group[0]["order"] = current_order
                new_regions.append(duplicate_group[0])
                current_order += 1
            else:
                lines = []
                used = set()

                for idx, r1 in enumerate(duplicate_group):
                    if idx in used:
                        continue
                    line = [r1]
                    used.add(idx)
                    for jdx in range(idx + 1, len(duplicate_group)):
                        if jdx in used:
                            continue
                        if y_overlap_fraction(r1["bounding_box"], duplicate_group[jdx]["bounding_box"]) >= 0.4:
                            line.append(duplicate_group[jdx])
                            used.add(jdx)
                    lines.append(line)

                lines.sort(key=lambda line: min(r["bounding_box"]["y"] for r in line))

                for line in lines:
                    line.sort(key=lambda r: r["bounding_box"]["x"])
                    for r in line:
                        r["order"] = current_order
                        new_regions.append(r)
                        current_order += 1

        image_entry["regions"] = new_regions

def merge_v4(data1, data2):
    map1 = {entry["image_name"]: entry["regions"] for entry in data1}
    map2 = {entry["image_name"]: entry["regions"] for entry in data2}

    result = []
    overlap_1, overlap_2 = set(), set()
    total_boxes_file1, total_boxes_file2 = 0, 0
    multiple_1_to_2 = defaultdict(list)
    multiple_2_to_1 = defaultdict(list)
    used_2_indices = defaultdict(set)
    non_overlap_1 = defaultdict(list)
    non_overlap_2 = defaultdict(list)
    skipped_large_width_2 = defaultdict(list)

    for img_name in map1:
        regions1 = map1[img_name]
        regions2 = map2.get(img_name, [])
        total_boxes_file1 += len(regions1)
        total_boxes_file2 += len(regions2)

        merged_regions = []
        invalid_indices_2 = set()

        for idx1, reg1 in enumerate(regions1):
            box1 = reg1["bounding_box"]
            found = False
            for idx2, reg2 in enumerate(regions2):
                box2 = reg2["bounding_box"]

                if boxes_overlap_adjusted(box1, box2) and has_sufficient_y_overlap(box1, box2, threshold=0.4):
                    if box2["w"] > 1.3 * box1["w"]:
                        skipped_large_width_2[img_name].append((idx2, reg2.get("order", -1)))
                        invalid_indices_2.add(idx2)
                        continue

                    overlap_1.add(idx1)
                    overlap_2.add(idx2)
                    multiple_1_to_2[(img_name, idx1)].append((img_name, idx2))
                    multiple_2_to_1[(img_name, idx2)].append((img_name, idx1))
                    found = True

                    w1, h1 = box1["w"], box1["h"]
                    w2, h2 = box2["w"], box2["h"]
                    select_r1 = False
                    if (w1 >= 0.95 * w2 and w1 <= 3.5 * w2 and 0.9 * h2 <= h1 <= 1.5 * h2):
                        select_r1 = True
                    elif (h1 > h2 and h1 <= 1.4 * h2 and 0.7 * w2 <= w1 <= 1.3 * w2):
                        select_r1 = True

                    if select_r1:
                        merged = reg1.copy()
                        merged["bounding_box"] = box1
                    else:
                        merged = {
                            "order": reg1.get("order"),
                            "line": reg1.get("line"),
                            "bounding_box": box2,
                            "text": reg2.get("text", "")
                        }
                        for key in reg1:
                            if key not in merged:
                                merged[key] = reg1[key]
                    merged_regions.append(merged)
                    used_2_indices[img_name].add(idx2)
                    break

            if not found:
                merged_regions.append(reg1)
                non_overlap_1[img_name].append((idx1, reg1.get("order", -1)))

        for idx2, reg2 in enumerate(regions2):
            if idx2 not in used_2_indices[img_name] and idx2 not in invalid_indices_2:
                new_region = {
                    "order": 0,
                    "line": reg2.get("line", -1),
                    "bounding_box": reg2.get("bounding_box"),
                    "text": reg2.get("text", "")
                }
                merged_regions.append(new_region)
                non_overlap_2[img_name].append((idx2, reg2.get("order", -1)))

        merged_regions.sort(key=lambda r: r["order"])
        result.append({"image_name": img_name, "regions": merged_regions})

    assign_orders_based_on_neighbors(result)
    remove_smaller_overlapping_regions(result)
    resolve_duplicate_orders(result)

    stats_lines = []
    stats_lines.append("Integrated 1 and 2")
    return result

# if __name__ == "__main__":
#     file1 = "1.json"
#     file2 = "2.json"
#     output_file = "union.json"
#     summary_file = "overlap_report_summary.txt"

#     if not all(os.path.exists(f) for f in [file1, file2]):
#         print("One or more input files not found.")
#         exit(1)

#     union_data, count1, count2, overlap1, overlap2, stats_lines = merge_all_regions_with_stats(file1, file2)

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(union_data, f, indent=2, ensure_ascii=False)