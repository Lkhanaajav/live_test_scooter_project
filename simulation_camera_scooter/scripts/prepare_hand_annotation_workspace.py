#!/usr/bin/env python3
"""
Copy annotation images plus starter masks into a workspace for manual correction.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent


DEFAULT_IMAGES_ROOT = SIM_DIR / "annotation_frames"
DEFAULT_MASKS_ROOT = PROJECT_ROOT / "outputs" / "pseudo_labels" / "oneformer_cityscapes_swin_large_binary" / "masks"
DEFAULT_PREVIEWS_ROOT = PROJECT_ROOT / "outputs" / "pseudo_labels" / "oneformer_cityscapes_swin_large_binary" / "previews"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks-root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--previews-root", default=str(DEFAULT_PREVIEWS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--video", default=None, help="Optional single video folder name, e.g. IMG_1922")
    return parser.parse_args()


def copy_tree(src_root: Path, dst_root: Path, files: list[Path]) -> None:
    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root).resolve()
    masks_root = Path(args.masks_root).resolve()
    previews_root = Path(args.previews_root).resolve()
    output_root = Path(args.output_root).resolve()

    def _select(files: list[Path]) -> list[Path]:
        if not args.video:
            return files
        needle = f"{args.video}\\"
        needle2 = f"{args.video}/"
        return [p for p in files if needle in str(p) or needle2 in str(p)]

    images = _select(sorted(images_root.rglob("*.jpg")))
    masks = _select(sorted(masks_root.rglob("*.png")))
    previews = _select(sorted(previews_root.rglob("*.jpg"))) if previews_root.exists() else []

    if not images:
        raise FileNotFoundError(f"No images found under {images_root}")
    if not masks:
        raise FileNotFoundError(f"No masks found under {masks_root}")

    workspace_images = output_root / "images"
    workspace_masks = output_root / "masks"
    workspace_previews = output_root / "previews"
    workspace_images.mkdir(parents=True, exist_ok=True)
    workspace_masks.mkdir(parents=True, exist_ok=True)
    if previews:
        workspace_previews.mkdir(parents=True, exist_ok=True)

    copy_tree(images_root, workspace_images, images)
    copy_tree(masks_root, workspace_masks, masks)
    if previews:
        copy_tree(previews_root, workspace_previews, previews)

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Hand Annotation Workspace",
                "",
                "Edit the PNG files under `masks/`.",
                "",
                "Rules:",
                "- white (`255`) = drivable",
                "- black (`0`) = non-drivable",
                "- keep masks single-channel PNG",
                "- do not rename files",
                "- keep the same folder structure as `images/`",
                "",
                "Helpful folders:",
                "- `images/` = original RGB frames",
                "- `masks/` = starter masks to correct",
                "- `previews/` = overlay previews from the teacher pass",
                "",
                "When you are done, retraining can use:",
                f"`python simulation_camera_scooter\\scripts\\train_binary_segformer.py --masks-root {workspace_masks}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[hand-annot] workspace={output_root}")
    print(f"[hand-annot] images={len(images)}")
    print(f"[hand-annot] masks={len(masks)}")
    if args.video:
        print(f"[hand-annot] video={args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
