"""Convenient launcher for the MediaPipe demo scripts.

This helper provides two commands:

```
python run_demo.py list
python run_demo.py run face-detect -- --image path/to/file
```

The first lists the available demos with short descriptions, while the
second runs the chosen script and forwards extra arguments directly to it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent


def _build_script_catalog() -> Dict[str, Dict[str, str]]:
    """Return a dictionary describing available demo scripts."""

    return {
        "face-detect": {
            "path": REPO_ROOT / "face-detect.py",
            "description": "ตรวจจับใบหน้าแบบเรียลไทม์ด้วย MediaPipe Face Detection",
        },
        "face-mesh": {
            "path": REPO_ROOT / "face-mesh.py",
            "description": "แสดงจุด Face Mesh พร้อม FPS counter",
        },
        "hand-tracking": {
            "path": REPO_ROOT / "hand-tracking.py",
            "description": "ตรวจจับและติดตามมือด้วยโมดูล Hands",
        },
        "pose-detect": {
            "path": REPO_ROOT / "pose-detect.py",
            "description": "ตรวจจับท่าทางร่างกาย (Pose) พร้อมโครงกระดูก",
        },
        "face-registration": {
            "path": REPO_ROOT / "face_registration" / "face-recognition.py",
            "description": "รู้จำใบหน้าจากเวกเตอร์ที่บันทึกไว้และกด n เพื่อเพิ่มคนใหม่",
        },
        "face-registration-ui": {
            "path": REPO_ROOT / "face_registration" / "face_recognition_ui.py",
            "description": "อินเทอร์เฟซ GUI สำหรับลงทะเบียนและทดลองรู้จำใบหน้า",
        },
        "attendance": {
            "path": REPO_ROOT / "face_registration" / "attendance.py",
            "description": "ระบบบันทึกเวลาเข้าออกงานด้วยการรู้จำใบหน้า",
        },
    }


def list_scripts() -> None:
    """Print available scripts in a readable table."""

    catalog = _build_script_catalog()
    longest_name = max(len(name) for name in catalog)
    print("สคริปต์ที่พร้อมใช้งาน:")
    for name, meta in sorted(catalog.items()):
        path = meta["path"]
        description = meta["description"]
        status = "พร้อม" if path.exists() else "หาไฟล์ไม่พบ"
        print(f"  {name.ljust(longest_name)}  - {description} ({status})")


def run_script(name: str, extra_args: list[str]) -> int:
    """Execute the selected script and return its exit code."""

    catalog = _build_script_catalog()
    entry = catalog[name]
    path = entry["path"]

    if not path.exists():
        print(f"ไม่พบไฟล์สำหรับสคริปต์ '{name}' ที่ {path}", file=sys.stderr)
        return 1

    command = [sys.executable, str(path), *extra_args]
    print("กำลังรัน:", " ".join(command))
    try:
        result = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print("\nยกเลิกโดยผู้ใช้")
        return 130
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ตัวเลือกช่วยรันสคริปต์ MediaPipe ที่รวมอยู่ในโปรเจกต์นี้",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="แสดงสคริปต์ทั้งหมดที่มี")

    run_parser = subparsers.add_parser(
        "run", help="รันสคริปต์ตามชื่อ เช่น python run_demo.py run face-detect"
    )
    run_parser.add_argument(
        "name",
        choices=sorted(_build_script_catalog().keys()),
        help="ชื่อสคริปต์ที่ต้องการรัน",
    )
    run_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="อาร์กิวเมนต์เพิ่มเติมที่จะส่งต่อไปยังสคริปต์ปลายทาง",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        list_scripts()
        return 0

    if args.command == "run":
        extra_args = args.script_args or []
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        return run_script(args.name, extra_args)

    parser.error("ไม่รู้จักคำสั่ง")
    return 2


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
