from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import yaml
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training", "train2017"),
    "val": ("val", "valid", "validation", "dev", "val2017"),
    "test": ("test", "testing", "test2017"),
}


@dataclass
class PreparedDataset:
    root: Path
    train: str
    val: str
    test: str | None
    names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO26 on a Kaggle dataset.")
    parser.add_argument(
        "--dataset",
        default="fareselmenshawii/human-dataset",
        help="Kaggle dataset handle in the form owner/dataset-slug.",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO26 checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=None, help="Training device, for example cpu, 0, 0,1, mps.")
    parser.add_argument("--project", default="runs/train", help="Ultralytics project directory.")
    parser.add_argument("--name", default="yolo26-human", help="Ultralytics run name.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["person"],
        help="Fallback class names if the dataset does not contain them explicitly.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force KaggleHub to redownload the dataset even if it is cached.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def dataset_local_dir(handle: str) -> Path:
    return Path("datasets") / handle.replace("/", "__")


def ensure_dataset_downloaded(handle: str, force_download: bool) -> Path:
    path = kagglehub.dataset_download(handle, force_download=force_download)
    root = Path(path)
    if not root.exists():
        fail(f"KaggleHub returned '{root}', but this path does not exist.")
    return root


def find_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def find_dirs_named(root: Path, names: tuple[str, ...]) -> list[Path]:
    lowered = {name.lower() for name in names}
    return sorted(path for path in root.rglob("*") if path.is_dir() and path.name.lower() in lowered)


def detect_split(name: str) -> str | None:
    lowered = name.lower()
    for split, aliases in SPLIT_ALIASES.items():
        if lowered in aliases:
            return split
    return None


def write_data_yaml(target: Path, prepared: PreparedDataset) -> Path:
    data = {
        "path": str(prepared.root.resolve()),
        "train": prepared.train,
        "val": prepared.val,
        "names": {idx: name for idx, name in enumerate(prepared.names)},
    }
    if prepared.test:
        data["test"] = prepared.test

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return target


def maybe_read_existing_data_yaml(root: Path) -> PreparedDataset | None:
    for data_yaml in sorted(root.rglob("data.yaml")):
        with data_yaml.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not {"train", "val", "names"} <= set(data):
            continue
        base = Path(data.get("path") or data_yaml.parent)
        return PreparedDataset(
            root=base if base.is_absolute() else (data_yaml.parent / base).resolve(),
            train=str(data["train"]),
            val=str(data["val"]),
            test=str(data["test"]) if data.get("test") else None,
            names=_normalize_names(data["names"]),
        )
    return None


def _normalize_names(names: dict[int, str] | list[str]) -> list[str]:
    if isinstance(names, list):
        return [str(item) for item in names]
    return [str(name) for _, name in sorted((int(k), v) for k, v in names.items())]


def prepare_yolo_dataset(root: Path, fallback_names: list[str]) -> PreparedDataset | None:
    existing = maybe_read_existing_data_yaml(root)
    if existing:
        return existing

    label_dirs = find_dirs_named(root, ("labels",))
    if not label_dirs:
        return None

    split_map: dict[str, tuple[Path, Path]] = {}
    for labels_dir in label_dirs:
        split = detect_split(labels_dir.parent.name) or detect_split(labels_dir.name)
        images_dir = labels_dir.parent / "images"
        if not images_dir.exists():
            sibling = labels_dir.parent.parent / "images" / labels_dir.name
            if sibling.exists():
                images_dir = sibling
        if split and images_dir.exists():
            split_map[split] = (images_dir, labels_dir)

    if "train" not in split_map or "val" not in split_map:
        return None

    prepared_root = root
    names = fallback_names
    return PreparedDataset(
        root=prepared_root,
        train=str(split_map["train"][0].relative_to(prepared_root)),
        val=str(split_map["val"][0].relative_to(prepared_root)),
        test=str(split_map["test"][0].relative_to(prepared_root)) if "test" in split_map else None,
        names=names,
    )


def prepare_coco_dataset(root: Path, output_root: Path) -> PreparedDataset | None:
    json_files = find_files(root, (".json",))
    coco_jsons: list[Path] = []
    categories: list[str] | None = None
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            continue
        if {"images", "annotations", "categories"} <= set(data):
            coco_jsons.append(json_file)
            if categories is None:
                categories = [category["name"] for category in sorted(data["categories"], key=lambda item: item["id"])]

    if not coco_jsons:
        return None

    annotations_dir = coco_jsons[0].parent
    converted_root = output_root / "converted_coco"
    if not converted_root.exists():
        convert_coco(labels_dir=str(annotations_dir), save_dir=str(converted_root), cls91to80=False)

    split_to_images: dict[str, Path] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            candidates = [
                root / "images" / alias,
                root / alias,
            ]
            for candidate in candidates:
                if candidate.exists():
                    split_to_images[split] = candidate
                    break
            if split in split_to_images:
                break

    if "train" not in split_to_images or "val" not in split_to_images:
        fail(
            "COCO annotations found, but train/val image folders were not detected. "
            "Expected folders like images/train + images/val or train + val."
        )

    return PreparedDataset(
        root=root,
        train=str(split_to_images["train"].relative_to(root)),
        val=str(split_to_images["val"].relative_to(root)),
        test=str(split_to_images["test"].relative_to(root)) if "test" in split_to_images else None,
        names=categories or ["person"],
    )


def prepare_voc_dataset(root: Path, output_root: Path) -> PreparedDataset | None:
    xml_files = find_files(root, (".xml",))
    if not xml_files:
        return None

    valid_xmls: list[Path] = []
    classes: set[str] = set()
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue
        objects = tree.findall(".//object")
        if not objects:
            continue
        valid_xmls.append(xml_file)
        for obj in objects:
            name = obj.findtext("name")
            if name:
                classes.add(name)

    if not valid_xmls:
        return None

    image_sets = root / "ImageSets" / "Main"
    jpeg_images = root / "JPEGImages"
    if not image_sets.exists() or not jpeg_images.exists():
        fail(
            "VOC-style XML annotations found, but expected Pascal VOC folders 'ImageSets/Main' and 'JPEGImages' were not found."
        )

    prepared_root = output_root / "voc_prepared"
    labels_root = prepared_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    names = sorted(classes)
    class_to_id = {name: idx for idx, name in enumerate(names)}
    split_to_image_dir: dict[str, Path] = {}
    available_splits = {path.stem: path for path in image_sets.glob("*.txt")}

    for split, aliases in SPLIT_ALIASES.items():
        split_file = next((available_splits[alias] for alias in aliases if alias in available_splits), None)
        if not split_file:
            continue
        split_image_dir = prepared_root / "images" / split
        split_label_dir = labels_root / split
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for image_id in image_ids:
            image_path = next((jpeg_images / f"{image_id}{ext}" for ext in IMAGE_EXTENSIONS if (jpeg_images / f"{image_id}{ext}").exists()), None)
            xml_path = next((xml for xml in valid_xmls if xml.stem == image_id), None)
            if image_path is None or xml_path is None:
                continue
            shutil.copy2(image_path, split_image_dir / image_path.name)
            write_voc_label(xml_path, split_label_dir / f"{image_id}.txt", class_to_id)
        split_to_image_dir[split] = split_image_dir

    if "train" not in split_to_image_dir or "val" not in split_to_image_dir:
        fail("VOC dataset detected, but train/val split files were not found in ImageSets/Main.")

    return PreparedDataset(
        root=prepared_root,
        train="images/train",
        val="images/val",
        test="images/test" if "test" in split_to_image_dir else None,
        names=names,
    )


def write_voc_label(xml_path: Path, output_path: Path, class_to_id: dict[str, int]) -> None:
    tree = ET.parse(xml_path)
    size = tree.find("size")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return

    lines: list[str] = []
    for obj in tree.findall(".//object"):
        class_name = obj.findtext("name")
        if class_name not in class_to_id:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(f"{class_to_id[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(root: Path, handle: str, fallback_names: list[str]) -> Path:
    output_root = dataset_local_dir(handle)
    output_root.mkdir(parents=True, exist_ok=True)

    prepared = prepare_yolo_dataset(root, fallback_names)
    if prepared is None:
        prepared = prepare_coco_dataset(root, output_root)
    if prepared is None:
        prepared = prepare_voc_dataset(root, output_root)
    if prepared is None:
        fail(
            "Dataset format was not recognized. Supported layouts: YOLO detect, COCO detect, Pascal VOC detect."
        )

    data_yaml = write_data_yaml(output_root / "data.yaml", prepared)
    print(f"Prepared dataset yaml: {data_yaml.resolve()}")
    return data_yaml


def main() -> None:
    args = parse_args()
    dataset_root = ensure_dataset_downloaded(args.dataset, args.force_download)
    print(f"Dataset root: {dataset_root}")
    data_yaml = prepare_dataset(dataset_root, args.dataset, args.class_names)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print(results)


if __name__ == "__main__":
    main()
