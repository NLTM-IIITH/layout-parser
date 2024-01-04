import cv2


# Convert list of layout parser blocks 'lp_blocks' into a list of bounding boxes of format [x1, y1, x2, y2]
def process_blocks_to_bboxes(lp_blocks):
    bboxes = []
    blocks = lp_blocks.to_dict()["blocks"]
    for blk in blocks:
        box = [int(blk["x_1"]), int(blk["y_1"]), int(blk["x_2"]), int(blk["y_2"])]
        bboxes.append(box)
    return bboxes


# Mask 'image' with white rectangles denoted by the list of 'boxes'
def mask_image(image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), -1)
    return image
