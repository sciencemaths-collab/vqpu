from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "tests"


def main() -> int:
    test_files = sorted(TEST_DIR.glob("test_section_*_smoke.py"))
    if not test_files:
        print("No smoke tests found.")
        return 1

    failures = []
    for path in test_files:
        print(f"== {path.relative_to(ROOT)} ==")
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        if result.returncode != 0:
            failures.append(path.name)

    if failures:
        print("\nSmoke test failures:")
        for name in failures:
            print(f" - {name}")
        return 1

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
