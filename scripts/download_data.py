"""Download QWS 2.0 and WS-DREAM dataset 1 from Zenodo into data/raw/.

The raw files are not committed: the QWS licence does not allow
redistribution. Run this once after cloning.

Usage: python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import pathlib
import urllib.request
import zipfile

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = [
    # (folder, file name, URL, md5 published on Zenodo)
    ("qws", "qws2.zip",
     "https://zenodo.org/records/3557008/files/qws2.zip?download=1",
     "c60230fea52eff40f0fafff8e08c4915"),
    ("wsdream", "wsdream_dataset1.zip",
     "https://zenodo.org/records/1133476/files/wsdream_dataset1.zip?download=1",
     "dde1f1f275d78d65b27015447f9db0e5"),
]


def main() -> None:
    for folder, name, url, md5 in FILES:
        target = RAW / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, target)
        digest = hashlib.md5(target.read_bytes()).hexdigest()
        if digest != md5:
            raise SystemExit(f"{name}: md5 {digest} does not match {md5}")
        with zipfile.ZipFile(target) as archive:
            archive.extractall(target.parent)
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
