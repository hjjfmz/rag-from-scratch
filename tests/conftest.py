"""Keep all test scratch on the project's drive, including stdlib tempfile users."""
import os
from pathlib import Path
import tempfile

scratch = Path(__file__).resolve().parents[1] / ".runtime" / "tmp"
scratch.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(scratch)
tempfile.tempdir = str(scratch)
