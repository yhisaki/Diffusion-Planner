#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def collect_npz_under_named_dirs(root: Path, dir_name: str) -> list[str]:
    """Collect npz files under all directories named dir_name recursively."""
    paths: list[str] = []

    for target_dir in root.rglob(dir_name):
        if not target_dir.is_dir():
            continue

        paths.extend(
            str(path.resolve())
            for path in target_dir.rglob("*.npz")
            if path.is_file()
        )

    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory to search recursively",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Directory to save path_list_train.json and path_list_valid.json",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    save_dir = args.save_dir.resolve() if args.save_dir else dataset_root
    save_dir.mkdir(parents=True, exist_ok=True)

    train_paths = collect_npz_under_named_dirs(dataset_root, "train")
    valid_paths = collect_npz_under_named_dirs(dataset_root, "valid")

    train_json_path = save_dir / "path_list_train.json"
    valid_json_path = save_dir / "path_list_valid.json"

    with open(train_json_path, "w") as f:
        json.dump(train_paths, f, indent=2)

    with open(valid_json_path, "w") as f:
        json.dump(valid_paths, f, indent=2)

    print(f"Train npz files: {len(train_paths)}")
    print(f"Valid npz files: {len(valid_paths)}")
    print(f"Saved: {train_json_path}")
    print(f"Saved: {valid_json_path}")


if __name__ == "__main__":
    main()