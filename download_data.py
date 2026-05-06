"""Download the DetectDrill dataset from HuggingFace into ./data/.

Default repo: nhonhoccode/DetectDrill
Default target: ./data/DataAug

Examples:
    python download_data.py
    python download_data.py --repo nhonhoccode/DetectDrill --target ./data
    HF_TOKEN=hf_xxx python download_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _find_dataaug(root: Path) -> Path | None:
    """If the repo's snapshot already contains a 'DataAug' subdir, return it.
    Otherwise return None and the caller falls back to the snapshot root.
    """
    if (root / "DataAug").is_dir():
        return root / "DataAug"
    matches = list(root.glob("**/DataAug"))
    return matches[0] if matches else None


def main(repo: str, target: Path, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    target.mkdir(parents=True, exist_ok=True)
    cache_dir = target / "_hf_cache"

    print(f"Downloading {repo} -> {target} (cache: {cache_dir})")
    snapshot_path = Path(snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(target / "_snapshot"),
        cache_dir=str(cache_dir),
        token=token,
    ))
    print(f"Snapshot landed at: {snapshot_path}")

    src = _find_dataaug(snapshot_path) or snapshot_path
    dst = target / "DataAug"
    if dst.exists() and dst.resolve() == src.resolve():
        print(f"Dataset already in place: {dst}")
        return

    # Symlink if cross-device-safe; otherwise the user can rename manually.
    if dst.exists():
        print(f"Target exists: {dst} (delete it manually if you want to re-link).")
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"Linked {src} -> {dst}")
    except OSError as e:
        print(f"Symlink failed ({e}); copying instead...")
        import shutil
        shutil.copytree(src, dst)
        print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset.")
    parser.add_argument("--repo", default="nhonhoccode/DetectDrill",
                        help="HuggingFace dataset repo id")
    parser.add_argument("--target", default="./data", type=Path,
                        help="Target directory (default: ./data)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    main(args.repo, args.target.resolve(), args.token)
