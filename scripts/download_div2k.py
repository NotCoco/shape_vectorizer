from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
EXPECTED_BYTES = 3_530_603_713


def download_with_resume(url: str, destination: Path) -> None:
    existing = destination.stat().st_size if destination.exists() else 0
    if existing == EXPECTED_BYTES:
        print(f"Archive already downloaded: {destination}")
        return
    if existing > EXPECTED_BYTES:
        raise RuntimeError(f"Existing archive is unexpectedly large: {destination}")

    headers = {"User-Agent": "vector-shape-learner/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        append = existing > 0 and response.status == 206
        if existing and not append:
            existing = 0
        mode = "ab" if append else "wb"
        total = EXPECTED_BYTES
        downloaded = existing
        with destination.open(mode) as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                percent = downloaded * 100.0 / total
                print(
                    f"\rDownloading {downloaded / 1024**3:.2f}/{total / 1024**3:.2f} GiB "
                    f"({percent:.1f}%)",
                    end="",
                    flush=True,
                )
    print()
    if destination.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"Incomplete download: got {destination.stat().st_size} bytes, expected {EXPECTED_BYTES}"
        )


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_resolved not in member_path.parents and member_path != destination_resolved:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify DIV2K high-resolution training images")
    parser.add_argument("--data-dir", type=Path, default=Path("data/DIV2K"))
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "DIV2K_train_HR.zip"
    image_dir = args.data_dir / "DIV2K_train_HR"
    existing_images = list(image_dir.glob("*.png")) if image_dir.exists() else []
    if len(existing_images) == 800:
        print(f"DIV2K is ready: {image_dir} (800 images)")
        return

    download_with_resume(URL, archive)
    print("Testing archive integrity...")
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt file inside archive: {corrupt}")
    print("Extracting...")
    safe_extract(archive, args.data_dir)
    images = list(image_dir.glob("*.png"))
    if len(images) != 800:
        raise RuntimeError(f"Expected 800 PNG images after extraction, found {len(images)}")
    if not args.keep_archive:
        archive.unlink()
    print(f"DIV2K is ready: {image_dir} (800 images)")


if __name__ == "__main__":
    main()
