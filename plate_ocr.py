"""Detect Russian license plates with YOLO and read them with OCR."""

# Enable postponed evaluation of annotations for cleaner type hints.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Save structured OCR results as JSON.
import json
# Clean and validate OCR text with regular expressions.
import re
# Work with filesystem paths safely.
from pathlib import Path

# OpenCV is used for image preprocessing and perspective correction.
import cv2
# NumPy is used for array math and box manipulation.
import numpy as np
# Tesseract runs the OCR stage.
import pytesseract
# Ultralytics loads the YOLO model and performs detection.
from ultralytics import YOLO

# Allowed Latin letters for Russian license plates.
LATIN_PLATE_LETTERS = "ABEKMHOPCTYX"

# Convert plate letters from Latin OCR output to visually matching Cyrillic letters.
LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    }
)


# Build the CLI for the script.
def parse_args() -> argparse.Namespace:
    # Create the top-level argument parser.
    parser = argparse.ArgumentParser(
        description="Detect a license plate with YOLO and read it with OCR."
    )
    # Path to the trained YOLO checkpoint.
    parser.add_argument("--model", default="exp.pt", help="Path to YOLO weights.")
    # Path to the input image.
    parser.add_argument("--source", default="image.png", help="Path to input image.")
    # Directory where all OCR outputs will be saved.
    parser.add_argument(
        "--output-dir",
        default="runs/ocr",
        help="Directory for OCR outputs.",
    )
    # Minimum YOLO confidence for the detection stage.
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    # Extra margin around the detected bounding box.
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Relative padding added around the detected box.",
    )
    # Return the parsed CLI arguments.
    return parser.parse_args()


# Create a per-image output directory.
def make_output_dir(base_dir: Path, stem: str) -> Path:
    # Build the final output directory path.
    output_dir = base_dir / stem
    # Create the directory if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Return the ready-to-use path.
    return output_dir


# Expand the detected box slightly to avoid cutting off plate borders.
def expand_box(
    box: np.ndarray, width: int, height: int, padding_ratio: float
) -> tuple[int, int, int, int]:
    # Convert floating box coordinates to integers.
    x1, y1, x2, y2 = box.astype(int)
    # Compute horizontal padding from the current box width.
    pad_x = int((x2 - x1) * padding_ratio)
    # Compute vertical padding from the current box height.
    pad_y = int((y2 - y1) * padding_ratio)
    # Return a clipped box that always stays inside the image boundaries.
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


# Prepare several OCR-friendly views of the same plate image.
def preprocess_variants(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    # Convert the crop to grayscale.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Smooth noise while preserving edges around letters.
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    # Upscale the image to make small characters easier for OCR.
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Create a global thresholded version.
    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Create a local thresholded version for uneven lighting.
    adaptive = cv2.adaptiveThreshold(
        upscaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    # Invert black and white because some OCR cases respond better to that.
    inverted = cv2.bitwise_not(otsu)
    # Close small gaps inside characters.
    morph = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    # Return all prepared variants with names for debugging.
    return [
        ("gray", upscaled),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("inverted", inverted),
        ("morph", morph),
    ]


# Keep only uppercase alphanumeric OCR symbols.
def normalize_plate_text(text: str) -> str:
    # Convert OCR output to uppercase.
    text = text.upper()
    # Drop any punctuation, spaces, and unsupported characters.
    text = re.sub(r"[^A-Z0-9]", "", text)
    # Return an empty string when nothing useful remains.
    if not text:
        return ""
    # Return the cleaned OCR text.
    return text


# Convert OCR Latin letters to the corresponding Cyrillic plate letters.
def to_cyrillic_plate(text: str) -> str:
    # Translate every supported Latin plate letter to Cyrillic.
    return text.translate(LATIN_TO_CYRILLIC)


# Rank OCR candidates so valid plate-like strings win.
def score_plate_text(text: str) -> float:
    # Reject empty OCR output immediately.
    if not text:
        return 0.0
    # Start with a simple length-based score.
    score = len(text)
    # Strongly prefer the exact Russian plate pattern.
    if re.fullmatch(rf"[{LATIN_PLATE_LETTERS}]\d{{3}}[{LATIN_PLATE_LETTERS}]{{2}}\d{{2,3}}", text):
        score += 20
    # Give a smaller bonus to partially plausible plate strings.
    elif re.search(r"\d{3}", text) and re.search(rf"[{LATIN_PLATE_LETTERS}]{{1,3}}", text):
        score += 8
    # Return the final score.
    return score


# Order 4 points as top-left, top-right, bottom-right, bottom-left.
def order_quad_points(points: np.ndarray) -> np.ndarray:
    # Sum coordinates to find top-left and bottom-right corners.
    sums = points.sum(axis=1)
    # Subtract coordinates to find top-right and bottom-left corners.
    diffs = np.diff(points, axis=1).reshape(-1)
    # Return the points in the order expected by perspective transform.
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


# Try to straighten the plate by finding its rectangular border.
def rectify_plate(crop: np.ndarray) -> np.ndarray | None:
    # Convert the crop to grayscale for contour detection.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Blur small texture so the plate rectangle is easier to isolate.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold bright plate pixels from the darker surroundings.
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    # Extract external contours only.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check the largest contours first because the plate border is usually dominant.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Iterate through the biggest candidate contours.
    for contour in contours[:10]:
        # Measure the contour perimeter.
        peri = cv2.arcLength(contour, True)
        # Approximate the contour with fewer points.
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # Skip contours that are not quadrilaterals.
        if len(approx) != 4:
            continue
        # Compute the bounding box of the quadrilateral.
        _, _, w, h = cv2.boundingRect(approx)
        # Skip tiny rectangles that cannot be a license plate.
        if w < 40 or h < 15:
            continue
        # Put the four corners into a stable order.
        rect = order_quad_points(approx.reshape(4, 2).astype(np.float32))
        # Estimate the output width of the warped plate.
        width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
        # Estimate the output height of the warped plate.
        height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
        # Define a flat destination rectangle.
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        # Compute the perspective transform matrix.
        matrix = cv2.getPerspectiveTransform(rect, dst)
        # Return the rectified plate image.
        return cv2.warpPerspective(crop, matrix, (width, height))
    # Return nothing when no reliable plate rectangle is found.
    return None


# Run Tesseract on all preprocessed versions of the image.
def collect_tesseract_candidates(image: np.ndarray, prefix: str) -> list[dict[str, str | float]]:
    # Define several useful single-line OCR modes.
    configs = [
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]
    # Store every OCR attempt here.
    candidates: list[dict[str, str | float]] = []
    # Iterate through all prepared image variants.
    for variant_name, prepared in preprocess_variants(image):
        # Try every Tesseract configuration.
        for config in configs:
            # Read text from the current image variant.
            raw_text = pytesseract.image_to_string(prepared, lang="eng", config=config).strip()
            # Normalize the OCR text before scoring.
            normalized = normalize_plate_text(raw_text)
            # Save the candidate for later ranking.
            candidates.append(
                {
                    "variant": f"{prefix}_{variant_name}",
                    "config": config,
                    "raw_text": raw_text,
                    "normalized_text": normalized,
                    "score": score_plate_text(normalized),
                }
            )
    # Return all collected OCR candidates.
    return candidates


# Build extra candidates by reading the main plate area and the region code separately.
def build_composite_candidates(warped: np.ndarray) -> list[dict[str, str | float]]:
    # Read the height and width of the rectified plate.
    h, w = warped.shape[:2]
    # Crop the left part where the main plate number lives.
    main = warped[:, : int(w * 0.76)]
    # Crop the top-right part where the region code usually lives.
    region = warped[: int(h * 0.75), int(w * 0.72) :]
    # Collect composite OCR attempts here.
    candidates: list[dict[str, str | float]] = []
    # OCR configs for the main alphanumeric part.
    main_configs = [
        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]
    # OCR configs for the numeric region code.
    region_configs = [
        "--psm 8 -c tessedit_char_whitelist=0123456789",
        "--psm 7 -c tessedit_char_whitelist=0123456789",
    ]
    # Prepare multiple thresholded versions of the main crop.
    main_variants = preprocess_variants(main)
    # Prepare multiple thresholded versions of the region crop.
    region_variants = preprocess_variants(region)
    # Iterate through all main image variants.
    for main_variant_name, main_variant in main_variants:
        # Iterate through all region image variants.
        for region_variant_name, region_variant in region_variants:
            # Try each OCR mode for the main crop.
            for main_config in main_configs:
                # Try each OCR mode for the region crop.
                for region_config in region_configs:
                    # OCR the main alphanumeric part.
                    main_raw = pytesseract.image_to_string(
                        main_variant, lang="eng", config=main_config
                    ).strip()
                    # OCR the numeric region code.
                    region_raw = pytesseract.image_to_string(
                        region_variant, lang="eng", config=region_config
                    ).strip()
                    # Clean the main OCR string.
                    main_text = normalize_plate_text(main_raw)
                    # Keep digits only for the region code.
                    region_text = re.sub(r"[^0-9]", "", region_raw)
                    # Concatenate both OCR parts into a single candidate.
                    combined = f"{main_text}{region_text}"
                    # Save the combined candidate for scoring.
                    candidates.append(
                        {
                            "variant": f"warp_main:{main_variant_name}+region:{region_variant_name}",
                            "config": f"{main_config} | {region_config}",
                            "raw_text": f"{main_raw} | {region_raw}",
                            "normalized_text": combined,
                            "score": score_plate_text(combined),
                        }
                    )
    # Return all composite OCR candidates.
    return candidates


# Run the full OCR stage on the detected plate crop.
def run_ocr(
    crop: np.ndarray, output_dir: Path
) -> tuple[str, str, list[dict[str, str | float]], np.ndarray | None]:
    # Start with OCR candidates from the original crop.
    candidates = collect_tesseract_candidates(crop, "crop")
    # Try to rectify the plate into a front-facing rectangle.
    warped = rectify_plate(crop)
    # Continue only when plate rectification succeeded.
    if warped is not None:
        # Save the rectified plate for debugging.
        cv2.imwrite(str(output_dir / "plate_warped.png"), warped)
        # Add OCR candidates from the rectified plate.
        candidates.extend(collect_tesseract_candidates(warped, "warp"))
        # Add OCR candidates built from separate main and region reads.
        candidates.extend(build_composite_candidates(warped))
    # Pick the candidate with the best score.
    best = max(candidates, key=lambda item: item["score"], default=None)
    # Return empty results if OCR failed completely.
    if not best:
        return "", "", [], warped
    # Return the best normalized text, raw text, all candidates, and the warped image.
    return str(best["normalized_text"]), str(best["raw_text"]), candidates, warped


# Execute the full detection and OCR pipeline.
def main() -> None:
    # Read user-provided CLI arguments.
    args = parse_args()
    # Convert the input path into a Path object.
    source_path = Path(args.source)
    # Create an output directory named after the input file.
    output_dir = make_output_dir(Path(args.output_dir), source_path.stem)
    # Load the YOLO model from disk.
    model = YOLO(args.model)
    # Run detection on the input image.
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        verbose=False,
        save=False,
    )
    # Stop if the detector returned no results at all.
    if not results:
        raise SystemExit("No inference results returned.")
    # Use the first prediction result because we pass a single image.
    result = results[0]
    # Read the original image back from the result object.
    image = result.orig_img
    # Extract detected boxes from the result.
    boxes = result.boxes
    # Stop if no plate was detected.
    if boxes is None or len(boxes) == 0:
        raise SystemExit("License plate was not detected.")
    # Find the index of the highest-confidence detection.
    best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
    # Read the corresponding bounding box coordinates.
    box = boxes.xyxy.cpu().numpy()[best_idx]
    # Read the corresponding detection confidence.
    conf = float(boxes.conf.cpu().numpy()[best_idx])
    # Read the original image dimensions.
    h, w = image.shape[:2]
    # Expand the best box to keep some margin around the plate.
    x1, y1, x2, y2 = expand_box(box, w, h, args.padding)
    # Crop the detected plate area.
    crop = image[y1:y2, x1:x2]
    # Save the raw plate crop.
    cv2.imwrite(str(output_dir / "plate_crop.png"), crop)
    # Run OCR on the crop and collect diagnostics.
    plate_text_latin, raw_text, candidates, warped = run_ocr(crop, output_dir)
    # Convert the plate text to Cyrillic when OCR succeeded.
    plate_text = to_cyrillic_plate(plate_text_latin) if plate_text_latin else ""
    # Copy the original image for drawing.
    annotated = image.copy()
    # Draw the detection box on the image.
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Choose either the recognized plate text or a fallback label.
    label = plate_text or "OCR_FAILED"
    # Draw the recognized text above the box.
    cv2.putText(
        annotated,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    # Save the annotated image.
    cv2.imwrite(str(output_dir / "annotated.png"), annotated)
    # Build the detailed JSON report.
    report = {
        "source": str(source_path.resolve()),
        "model": str(Path(args.model).resolve()),
        "plate_text": plate_text,
        "plate_text_latin": plate_text_latin,
        "raw_text": raw_text,
        "detection_confidence": round(conf, 4),
        "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "outputs": {
            "crop": str((output_dir / "plate_crop.png").resolve()),
            "annotated": str((output_dir / "annotated.png").resolve()),
            "warped": str((output_dir / "plate_warped.png").resolve()) if warped is not None else None,
        },
        "ocr_candidates": candidates,
    }
    # Write the full diagnostic report to disk.
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Build a shorter summary for terminal output.
    summary = {
        "plate_text": plate_text,
        "plate_text_latin": plate_text_latin,
        "detection_confidence": round(conf, 4),
        "result_json": str((output_dir / "result.json").resolve()),
        "annotated": str((output_dir / "annotated.png").resolve()),
        "warped": str((output_dir / "plate_warped.png").resolve()) if warped is not None else None,
    }
    # Print the concise summary to stdout.
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# Run the script only when it is executed directly.
if __name__ == "__main__":
    # Enter the main program.
    main()
