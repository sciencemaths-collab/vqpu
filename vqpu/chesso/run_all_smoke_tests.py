from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
CHESSO_DIR = THIS.parent
TEST_DIR = CHESSO_DIR / "tests"
PROJECT_ROOT = CHESSO_DIR.parent.parent


def main() -> int:
    test_files = sorted(TEST_DIR.glob("test_section_*_smoke.py"))
    if not test_files:
        print("No smoke tests found.")
        return 1

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    )

    failures = []
    for path in test_files:
        print(f"== {path.relative_to(PROJECT_ROOT)} ==")
        result = subprocess.run(
            [sys.executable, str(path)], cwd=PROJECT_ROOT, env=env
        )
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
