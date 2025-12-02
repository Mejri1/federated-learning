"""
Utility to standardize member image datasets:
- copies from data/original/<member>
- converts every frame to RGB JPEG
- renames sequentially (e.g., abir01.jpg)
- saves metadata CSV per member
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pillow_heif = None


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}


def normalize_member(member: str, source_root: Path, output_root: Path) -> dict:
    source_dir = source_root / member
    target_dir = output_root / member
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder missing for member '{member}': {source_dir}")

    records = []
    for idx, src_path in enumerate(sorted(source_dir.iterdir())):
        if not src_path.is_file() or src_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        new_name = f"{member}{idx + 1:02d}.jpg"
        dst_path = target_dir / new_name

        with Image.open(src_path) as img:
            rgb = img.convert("RGB")
            rgb.save(dst_path, format="JPEG", quality=95)

        records.append({"original": src_path.name, "normalized": new_name})

    metadata_path = target_dir / f"metadata_{member}.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["original", "normalized"])
        writer.writeheader()
        writer.writerows(records)

    return {
        "member": member,
        "count": len(records),
        "metadata": metadata_path.relative_to(output_root.parent),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize member image datasets.")
    parser.add_argument(
        "--members",
        nargs="+",
        default=["abir", "jihene", "omarbr", "omarmej"],
        help="List of member folder names to process (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        default="data/original",
        type=Path,
        help="Directory containing original member folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data/normalized",
        type=Path,
        help="Directory to store normalized images (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []

    for member in args.members:
        info = normalize_member(member, args.source, args.output)
        summary.append(info)
        print(f"[+] {member}: {info['count']} images normalized -> {info['metadata']}")

    total = sum(item["count"] for item in summary)
    print(f"\nCompleted normalization for {len(summary)} members, {total} images total.")


if __name__ == "__main__":
    main(sys.argv[1:])

