import sys
from pathlib import Path

# On est dans <repo_root>/conftest.py
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))