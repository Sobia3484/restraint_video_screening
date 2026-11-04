#!/usr/bin/env python
"""Entry point:  python scripts/run_extraction.py [--platform both] [--queries Q001 ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
