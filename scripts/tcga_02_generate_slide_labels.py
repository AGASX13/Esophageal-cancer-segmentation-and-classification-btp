from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import get_paths


WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


@dataclass(frozen=True)
class LabeledSlide:
    slide_id: str
    label: int
    class_name: str
    path: Path


def iter_wsi_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in WSI_EXTS]


def build_labels(
    wsi_root: Path,
    cancer_dirname: str = "cancer",
    normal_dirname: str = "normal",
    cancer_label: int = 1,
    normal_label: int = 0,
) -> list[LabeledSlide]:
    cancer_dir = wsi_root / cancer_dirname
    normal_dir = wsi_root / normal_dirname

    slides: list[LabeledSlide] = []
    for class_dir, class_name, label in [
        (cancer_dir, cancer_dirname, cancer_label),
        (normal_dir, normal_dirname, normal_label),
    ]:
        for f in iter_wsi_files(class_dir):
            slide_id = f.stem
            slides.append(LabeledSlide(slide_id=slide_id, label=label, class_name=class_name, path=f))

    # Detect duplicate slide_ids across classes
    by_id: dict[str, list[LabeledSlide]] = {}
    for s in slides:
        by_id.setdefault(s.slide_id, []).append(s)

    duplicates = {k: v for k, v in by_id.items() if len(v) > 1}
    if duplicates:
        lines = ["Duplicate slide_id(s) found (same stem in multiple files/classes):"]
        for slide_id, items in sorted(duplicates.items()):
            lines.append(f"- {slide_id}:")
            for it in items:
                lines.append(f"  - label={it.label} class={it.class_name} path={it.path}")
        raise RuntimeError("\n".join(lines))

    slides.sort(key=lambda s: (s.label, s.slide_id))
    return slides


def write_csv(slides: list[LabeledSlide], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "label", "class", "path"])
        for s in slides:
            # Store relative paths to keep CSV portable
            try:
                rel_path = s.path.relative_to(get_paths().root)
            except ValueError:
                rel_path = s.path
            w.writerow([s.slide_id, s.label, s.class_name, str(rel_path)])


def main() -> int:
    paths = get_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Generate slide-level labels CSV based on folder names.\n\n"
            "Expected structure:\n"
            "  data/raw/tcga_esca_wsi/cancer/*.svs\n"
            "  data/raw/tcga_esca_wsi/normal/*.svs\n"
        )
    )
    parser.add_argument(
        "--wsi-root",
        type=Path,
        default=paths.tcga_wsi_raw,
        help="Root folder that contains the class subfolders (default: data/raw/tcga_esca_wsi).",
    )
    parser.add_argument("--cancer-dir", type=str, default="cancer", help="Cancer folder name.")
    parser.add_argument("--normal-dir", type=str, default="normal", help="Normal folder name.")
    parser.add_argument("--cancer-label", type=int, default=1, help="Label to use for cancer.")
    parser.add_argument("--normal-label", type=int, default=0, help="Label to use for normal.")
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.tcga_annotations_raw / "slide_labels.csv",
        help="Output CSV path (default: data/raw/tcga_esca_annotations/slide_labels.csv).",
    )
    args = parser.parse_args()

    slides = build_labels(
        wsi_root=args.wsi_root,
        cancer_dirname=args.cancer_dir,
        normal_dirname=args.normal_dir,
        cancer_label=args.cancer_label,
        normal_label=args.normal_label,
    )

    write_csv(slides, args.out)

    print(f"Wrote {len(slides)} labeled slides to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

