"""server/__init__.py"""
from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / "version.txt"
__version__ = _VERSION_FILE.read_text().strip() if _VERSION_FILE.is_file() else "1.6.0"
