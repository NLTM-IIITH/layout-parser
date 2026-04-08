import base64
import os
import time
from os.path import join
from typing import Any

import aiohttp
from fastapi import HTTPException

from ..factory import register
from ..models import LayoutImageResponse, ModelChoice, Region

LIPIKAR_URL = "https://lipikar.cse.iitd.ac.in/api-direct/detection/infer"


def logtime(t: float, msg: str) -> None:
    print(f"[{int(time.time() - t)}s]\t {msg}")


LANGUAGE_ALIASES = {
    "ur": "urdu",
    "urd": "urdu",
    "hi": "hindi",
    "te": "telugu",
    "tel": "telugu",
    "ta": "tamil",
    "kn": "kannada",
    "ml": "malayalam",
    "bn": "bengali",
    "gu": "gujarati",
    "mr": "marathi",
    "pa": "punjabi",
    "or": "odia",
    "od": "odia",
    "as": "assamese",
    "kon": "konkani",
}


def normalize_language(language: str) -> str:
    if not language:
        return "english"
    lang = language.strip().lower()
    return LANGUAGE_ALIASES.get(lang, lang)


def choose_model(language: str) -> str:
    lang = normalize_language(language)
    # focusformer_line_level is only for urdu
    if lang == "urdu":
        return "focusformer_line_level"
    return "focusformer"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_images(folder_path: str) -> list[str]:
    valid = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    return sorted(
        [img for img in os.listdir(folder_path) if img.lower().endswith(valid)]
    )


def find_detections_list(obj: Any) -> list | None:
    """Recursively find the first list containing dicts with 'text_bbox'."""
    if isinstance(obj, dict):
        for v in obj.values():
            result = find_detections_list(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        if obj and all(isinstance(item, dict) and "text_bbox" in item for item in obj):
            return obj
        for item in obj:
            result = find_detections_list(item)
            if result is not None:
                return result
    return None


def extract_detection_list(result_item: Any) -> list[dict]:
    """Extract word detections from Lipikar response."""
    if not isinstance(result_item, dict):
        return []

    candidates = [
        result_item.get("output"),
        result_item.get("result"),
        result_item.get("detections"),
    ]

    for cand in candidates:
        if cand is None:
            continue

        if isinstance(cand, list):
            if cand and all(isinstance(item, dict) and "text_bbox" in item for item in cand):
                return cand
            for elem in cand:
                if isinstance(elem, dict):
                    bboxes = elem.get("bboxes")
                    if isinstance(bboxes, list) and bboxes:
                        return bboxes

        elif isinstance(cand, dict):
            bboxes = cand.get("bboxes")
            if isinstance(bboxes, list):
                return bboxes

            dets = cand.get("detections")
            if isinstance(dets, dict):
                bboxes = dets.get("bboxes")
                if isinstance(bboxes, list):
                    return bboxes

    found = find_detections_list(result_item)
    if found:
        return found

    return []


def convert_detections_to_regions(detections: list[dict]) -> list[Region]:
    """Convert Lipikar detections into Region objects."""
    parsed = []

    for det in detections:
        if not isinstance(det, dict):
            continue

        text_bbox = det.get("text_bbox")
        if isinstance(text_bbox, dict):
            try:
                x_min = int(text_bbox["x_min"])
                y_min = int(text_bbox["y_min"])
                x_max = int(text_bbox["x_max"])
                y_max = int(text_bbox["y_max"])

                if x_max <= x_min or y_max <= y_min:
                    continue

                parsed.append({
                    "x1": x_min,
                    "y1": y_min,
                    "x2": x_max,
                    "y2": y_max,
                    "line": int(text_bbox.get("line_index", 0)),
                    "word_index": int(text_bbox.get("word_index", 0)),
                })
                continue
            except Exception as e:
                print(f"Skipping invalid text_bbox: {e}")
                continue

        bbox = det.get("bounding_box")
        if isinstance(bbox, dict):
            try:
                x = int(bbox["x"])
                y = int(bbox["y"])
                w = int(bbox["w"])
                h = int(bbox["h"])

                if w <= 0 or h <= 0:
                    continue

                parsed.append({
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "line": int(det.get("line", 0)),
                    "word_index": int(det.get("order", 0)),
                })
            except Exception as e:
                print(f"Skipping invalid bounding_box: {e}")

    if not parsed:
        return []

    unique_lines = sorted(set(p["line"] for p in parsed))
    line_map = {old: idx + 1 for idx, old in enumerate(unique_lines)}

    parsed.sort(key=lambda r: (line_map[r["line"]], r["word_index"], r["x1"], r["y1"]))

    regions = []
    for item in parsed:
        try:
            region = Region.from_xyxy((item["x1"], item["y1"], item["x2"], item["y2"]))
            # optional metadata, if your Region model supports these attrs
            if hasattr(region, "line"):
                region.line = line_map[item["line"]]
            if hasattr(region, "order"):
                region.order = item["word_index"]
            regions.append(region)
        except Exception as e:
            print(f"Skipping invalid region conversion: {e}")

    return regions


async def call_lipikar_with_payload(payload: dict) -> dict:
    """Send request to Lipikar and return parsed JSON."""
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            LIPIKAR_URL,
            headers=headers,
            json=payload,
            timeout=300,
        ) as response:
            raw_text = await response.text()

            if response.status != 200:
                print(f"Lipikar error {response.status}: {raw_text}")
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Lipikar API error: {raw_text}",
                )

            try:
                return await response.json()
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"Lipikar returned non-JSON: {raw_text}",
                )


async def call_lipikar(
    folder_path: str,
    language: str,
    version: str = "1",
) -> list[LayoutImageResponse]:
    images = get_images(folder_path)
    if not images:
        return []

    normalized_language = normalize_language(language)
    model_id = choose_model(normalized_language)

    payload = {
        "config": {
            "allowPadding": True,
            "language": normalized_language,
            "modelId": model_id,
            "version": version,
        },
        "image": [{"imageContent": encode_image(join(folder_path, img))} for img in images],
    }

    print(
        f"Payload prepared: modelId={model_id}, "
        f"language={normalized_language}, version={version}"
    )

    try:
        ret = await call_lipikar_with_payload(payload)
    except HTTPException as e:
        # Fallback for Urdu
        if normalized_language == "urdu" and e.status_code == 400:
            print(
                "Urdu request with 'focusformer_line_level' failed. "
                "Retrying with 'focusformer'..."
            )
            payload["config"]["modelId"] = "focusformer"
            ret = await call_lipikar_with_payload(payload)
        else:
            raise

    if isinstance(ret, dict):
        results = ret.get("results", ret)
    else:
        results = ret

    if isinstance(results, dict):
        results = [results]
    elif not isinstance(results, list):
        raise HTTPException(500, f"Unexpected response format: {ret}")

    final_output: list[LayoutImageResponse] = []

    for idx, img in enumerate(images):
        result_item = results[idx] if idx < len(results) else {}
        detections = extract_detection_list(result_item)
        regions = convert_detections_to_regions(detections)

        print(f"{img}: detections={len(detections)}, regions={len(regions)}")

        final_output.append(
            LayoutImageResponse(
                image_name=img,
                regions=regions,
            )
        )

    return final_output


class LipikarProcessor:
    async def __call__(self, folder_path: str, **kwargs: Any):
        try:
            t = time.time()
            language = kwargs.get("language", "english")
            version = kwargs.get("version", "1")
            model = choose_model(language)

            logtime(t, f"Running Lipikar model: {model}")

            results = await call_lipikar(
                folder_path=folder_path,
                language=language,
                version=version,
            )

            logtime(t, "Lipikar completed")
            return results

        except HTTPException:
            raise
        except Exception as e:
            print(e)
            raise HTTPException(500, f"Lipikar error: {e}")


register(ModelChoice.focusformer)(LipikarProcessor())