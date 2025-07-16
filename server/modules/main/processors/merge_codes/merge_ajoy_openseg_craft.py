import json
import os
from collections import defaultdict


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

def assign_orders_based_on_neighbors(union_data):
    for image_entry in union_data:
        regions = image_entry["regions"]
        for r1 in regions:
            if r1.get("order", -1) != 0:
                continue

            box1 = r1["bounding_box"]
            y1 = box1["y"]
            y1_min, y1_max = y1, y1 + box1["h"]
            x1 = box1["x"]

            L = [r for r in regions if r != r1 and not (
                r["bounding_box"]["y"] + r["bounding_box"]["h"] < y1_min or
                r["bounding_box"]["y"] > y1_max)]

            L1 = [r for r in L if r["bounding_box"]["x"] < x1 and r.get("order", 0) > 0]
            L2 = [r for r in L if r["bounding_box"]["x"] > x1 and r.get("order", 0) > 0]

            if L1:
                R2 = max(L1, key=lambda r: r["bounding_box"]["x"])
                r1["order"] = R2["order"]
                r1["line"] = R2.get("line", -1)
            elif L2:
                R3 = min(L2, key=lambda r: r["bounding_box"]["x"])
                r1["order"] = R3["order"]
                r1["line"] = R3.get("line", -1)
            else:
                U = [r for r in regions if r["bounding_box"]["y"] < y1 and r.get("order", 0) > 0]
                if U:
                    R4 = max(U, key=lambda r: r["bounding_box"]["y"])
                    y4_min, y4_max = R4["bounding_box"]["y"], R4["bounding_box"]["y"] + R4["bounding_box"]["h"]
                    U1 = [r for r in regions if r != R4 and not (
                        r["bounding_box"]["y"] + r["bounding_box"]["h"] < y4_min or
                        r["bounding_box"]["y"] > y4_max)]
                    U1_valid = [r for r in U1 if r.get("order", 0) > 0]
                    if U1_valid:
                        R5 = max(U1_valid, key=lambda r: r["order"])
                        r1["order"] = R5["order"]
                        r1["line"] = R5.get("line", -1) + 1
                    else:
                        r1["order"] = R4["order"]
                        r1["line"] = R4.get("line", -1) + 1
                else:
                    U3 = [r for r in regions if r["bounding_box"]["y"] > y1 and r.get("order", 0) > 0]
                    if U3:
                        R6 = min(U3, key=lambda r: r["bounding_box"]["y"])
                        y6_min, y6_max = R6["bounding_box"]["y"], R6["bounding_box"]["y"] + R6["bounding_box"]["h"]
                        U4 = [r for r in regions if r != R6 and not (
                            r["bounding_box"]["y"] + r["bounding_box"]["h"] < y6_min or
                            r["bounding_box"]["y"] > y6_max)]
                        U4_valid = [r for r in U4 if r.get("order", 0) > 0]
                        if U4_valid:
                            R7 = min(U4_valid, key=lambda r: r["order"])
                            r1["order"] = R7["order"]
                            r1["line"] = R7.get("line", -1) - 1
                        else:
                            r1["order"] = R6["order"]
                            r1["line"] = R6.get("line", -1) - 1

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

def resolve_duplicate_orders(union_data):
    for image_entry in union_data:
        regions = image_entry["regions"]
        regions.sort(key=lambda r: r.get("order", 0))
        new_regions = []
        i = 0
        current_order = 0
        while i < len(regions):
            base_order = regions[i]["order"]
            group = [regions[i]]
            i += 1
            while i < len(regions) and regions[i]["order"] == base_order:
                group.append(regions[i])
                i += 1
            if len(group) == 1:
                group[0]["order"] = current_order
                new_regions.append(group[0])
                current_order += 1
            else:
                subgroups = []
                used = set()
                for idx, r1 in enumerate(group):
                    if idx in used:
                        continue
                    y1, h1 = r1["bounding_box"]["y"], r1["bounding_box"]["h"]
                    y2 = y1 + h1
                    subgroup = [r1]
                    used.add(idx)
                    for jdx in range(idx + 1, len(group)):
                        if jdx in used:
                            continue
                        ry1 = group[jdx]["bounding_box"]["y"]
                        ry2 = ry1 + group[jdx]["bounding_box"]["h"]
                        if not (ry2 < y1 or ry1 > y2):
                            subgroup.append(group[jdx])
                            used.add(jdx)
                    subgroups.append(sorted(subgroup, key=lambda r: r["bounding_box"]["x"]))
                subgroups.sort(key=lambda g: g[0]["bounding_box"]["y"])
                for subgroup in subgroups:
                    for r in subgroup:
                        r["order"] = current_order
                        new_regions.append(r)
                        current_order += 1
        image_entry["regions"] = new_regions

def integrate_3json(union_data, map3, map1, map2):
    added_from_3 = 0

    for image_entry in union_data:
        img_name = image_entry["image_name"]
        merged_regions = image_entry["regions"]
        regions3 = map3.get(img_name, [])
        if not regions3:
            continue

        # No order reassignment from 2.json
        ordered_3 = regions3
        # filtered_3 = remove_y_overlaps(ordered_3)
        filtered_3 = ordered_3
        final_3 = []

        for r3 in filtered_3:
            box3 = r3["bounding_box"]
            has_overlap = any(
                boxes_overlap_adjusted(box3, r["bounding_box"]) for r in merged_regions
            )

            if not has_overlap:
                # Force order=0 for all newly added 3.json regions
                r3["order"] = 0
                final_3.append(r3)

        added_from_3 += len(final_3)
        merged_regions.extend(final_3)
        merged_regions.sort(key=lambda r: r["order"])
        image_entry["regions"] = merged_regions

    return added_from_3

# def merge_all_regions_with_stats(file1, file2, file3):
def merge_all_regions(data1, data2, data3):
    # openseg, ajoy, craft
    # with open(file1, "r", encoding="utf-8") as f1, \
    #      open(file2, "r", encoding="utf-8") as f2, \
    #      open(file3, "r", encoding="utf-8") as f3:
    #     data1 = json.load(f1)
    #     data2 = json.load(f2)
    #     data3 = json.load(f3)

    map1 = {entry["image_name"]: entry["regions"] for entry in data1}
    map2 = {entry["image_name"]: entry["regions"] for entry in data2}
    map3 = {entry["image_name"]: entry["regions"] for entry in data3}

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
                if boxes_overlap_adjusted(box1, box2):
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
                    if (w1 >=0.95*w2 and w1 <= 3.5 * w2 and 0.4 * h2 <= h1 <= 1.5 * h2):
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

    count3 = integrate_3json(result, map3, map1, map2)
    assign_orders_based_on_neighbors(result)
    remove_smaller_overlapping_regions(result)
    resolve_duplicate_orders(result)

    stats_lines = []
    stats_lines.append("✅ Integrated 3.json after merging 1 and 2")

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)
    return result

# if __name__ == "__main__":
#     file1 = "openseg.json"
#     file2 = "ajoy.json"
#     file3 = "craft.json"
#     output_file = "union.json"
#     summary_file = "overlap_report_summary.txt"

#     if not all(os.path.exists(f) for f in [file1, file2, file3]):
#         print("❌ One or more input files not found.")
#         exit(1)

#     union_data = merge_all_regions_with_stats(file1, file2, file3)

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(union_data, f, indent=2, ensure_ascii=False)

    # total_union = sum(len(e["regions"]) for e in union_data)

    # summary = [
    #     f"✅ Full union saved in '{output_file}'",
    #     f"📦 Total boxes in {file1}: {count1}",
    #     f"📦 Total boxes in {file2}: {count2}",
    #     f"🔢 Overlapping regions in {file1}: {overlap1}",
    #     f"🔢 Non-overlapping regions in {file1}: {count1 - overlap1}",
    #     f"🔢 Overlapping regions in {file2}: {overlap2}",
    #     f"🔢 Non-overlapping regions in {file2}: {count2 - overlap2}",
    #     f"📦 Total regions in union.json: {total_union}",
    #     f"📥 Regions added from 3.json: {count3}",
    #     ""
    # ] + stats_lines

    # with open(summary_file, "w", encoding="utf-8") as f:
    #     f.write("\n".join(summary))

    # print("\n".join(summary))
