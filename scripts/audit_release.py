"""Read-only heuristic audit of a curated release tree; never an upload tool."""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"processed", "features", "runs", "logs", "private", "snapshots", "previews"}
SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".joblib", ".cif", ".pdb", ".sif", ".pdf", ".png", ".svg", ".whl", ".gz"}


def audit(root=ROOT):
    files, issues = [], []
    # Do not follow symlinks; Git internals and regenerable Python caches are ignored.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink():
            issues.append(f"Symlink: {relative}")
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        files.append({"path": str(relative), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if set(relative.parts) & FORBIDDEN or path.suffix in SUFFIXES:
            issues.append(f"Excluded artifact: {relative}")
        if len(data) > 2 * 1024 * 1024:
            issues.append(f"Unexpected large file: {relative}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"Non-text file: {relative}")
            continue
        if re.search(r"/(?:lustre[12]|home|appsnew)/", text):
            issues.append(f"Cluster/home absolute path: {relative}")
        if re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text):
            issues.append(f"Private key marker: {relative}")
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,})", text):
            issues.append(f"Possible credential: {relative}")
        if re.search(r"\b[ACDEFGHIKLMNPQRSTVWY]{45,}\b", text):
            issues.append(f"Possible long protein sequence: {relative}")
    return {"status": "pass" if not issues else "review_required", "file_count": len(files),
            "bytes": sum(row["bytes"] for row in files), "issues": issues, "files": files}


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["issues"]))
