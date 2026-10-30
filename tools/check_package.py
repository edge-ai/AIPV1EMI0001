#!/usr/bin/env python3
"""Verify an AIP distribution: exact file set, byte-identical copies, safe archive.

    python3 tools/check_package.py                       # check dist/ against the sources
    python3 tools/check_package.py --archive AIPV1EMI0001_dist.tgz

The expected set is what EAIS Tools 0.9.2 copies by itself (EAIS_FILES) plus every
file listed in .include. A missing, unexpected or modified file is an error, so
source models, credentials, node_modules or caches can never ship unnoticed.
Used by localdev.ipynb on the station and by the release workflow in CI.
"""
import argparse
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath

EAIS_FILES = {
    "manifest.json", "callback.py", "README.md", "CHANGELOG.md", "DESIGN.md", "PRODUCT.md",
    "resources/logo.png",
    "configs/config_infer_primary_detector_yolov7.txt", "configs/config_tracker_NvDCF_perf.yml",
    "models/AIPV1EMI0001_Yolo_Detector/config.pbtxt",
    "models/AIPV1EMI0001_Yolo_Detector/labels-rev3.txt",
    "models/AIPV1EMI0001_Yolo_Detector/libnvds_infercustomparser.so",
    "models/AIPV1EMI0001_Yolo_Detector/1/yolov7-fp16-1x8x8.engine",
}


class PackageError(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise PackageError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def include_entries(root):
    """Return .include entries after checking it is an explicit list of existing files."""
    root = Path(root)
    entries = [line.split("#", 1)[0].strip() for line in (root / ".include").read_text().splitlines()]
    entries = [entry for entry in entries if entry]
    check(entries, ".include must not be empty")
    check(len(entries) == len(set(entries)), "duplicate .include entry")
    for name in entries:
        path = PurePosixPath(name)
        check(not path.is_absolute() and ".." not in path.parts, f"unsafe .include entry: {name}")
        check(not any(char in name for char in "!*?[]"), f"use explicit files in .include: {name}")
        check((root / name).is_file(), f"missing included file: {name}")
        check(not (root / name).is_symlink(), f"symlink in .include: {name}")
        check("node_modules" not in path.parts and path.name != "flows_cred.json",
              f"dependency or credential in .include: {name}")
    return set(entries)


def expected_hashes(root):
    """Hashes of every file the package must contain, read from the source tree."""
    root = Path(root)
    names = EAIS_FILES | include_entries(root)
    if (root / "options.json").is_file():
        names.add("options.json")
    hashes = {}
    for name in sorted(names):
        path = root / name
        check(path.is_file() and path.stat().st_size, f"missing or empty package source: {name}")
        check(not path.is_symlink() and not any((root / parent).is_symlink()
                                                for parent in PurePosixPath(name).parents),
              f"symlinked package source: {name}")
        hashes[name] = sha256(path)
    return hashes


def check_dist(root, dist=None):
    root = Path(root)
    dist = Path(dist) if dist else root / "dist"
    expected = expected_hashes(root)
    paths = list(dist.rglob("*"))
    check(not any(p.is_symlink() for p in paths), "symlinks must not ship")
    actual = {p.relative_to(dist).as_posix() for p in paths if p.is_file()}
    check(actual == set(expected), f"dist/ mismatch: missing={sorted(set(expected) - actual)}, "
                                   f"unexpected={sorted(actual - set(expected))}")
    for name, digest in expected.items():
        check(sha256(dist / name) == digest, f"dist/ bytes differ from package source: {name}")
    return expected


def check_archive(archive, expected):
    seen = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            check(not path.is_absolute() and ".." not in path.parts, f"unsafe archive entry: {member.name}")
            check(member.isfile() or member.isdir(), f"unexpected archive entry: {member.name}")
            if member.isfile():
                check(path.parts[0] == "dist", f"archive entry outside dist/: {member.name}")
                name = path.relative_to("dist").as_posix()
                check(name not in seen, f"duplicate archive entry: {member.name}")
                with tar.extractfile(member) as source:
                    seen[name] = sha256_bytes(source.read())
    check(set(seen) == set(expected), f"archive mismatch: missing={sorted(set(expected) - set(seen))}, "
                                      f"unexpected={sorted(set(seen) - set(expected))}")
    for name, digest in expected.items():
        check(seen[name] == digest, f"archive bytes differ from package source: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=".", help="AIP source folder (default: current directory)")
    parser.add_argument("--dist", help="distribution folder (default: ROOT/dist)")
    parser.add_argument("--archive", help="also verify this exported .tgz")
    args = parser.parse_args(argv)
    try:
        expected = check_dist(args.root, args.dist)
        if args.archive:
            check_archive(args.archive, expected)
    except PackageError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(expected)} files, exact set and bytes"
          + (f"; archive {args.archive} matches" if args.archive else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
