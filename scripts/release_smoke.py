#!/usr/bin/env python3
"""Check packaged commands and runtime registration in a temporary directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    binary, python = args.binary.resolve(strict=True), args.python.absolute()
    if not python.is_file():
        parser.error("Python executable does not exist")
    with tempfile.TemporaryDirectory(prefix="vespasian-runtime-") as directory:
        env = {**os.environ, "VESPASIAN_DATA_HOME": directory}
        subprocess.run([str(binary), "setup", "--python", str(python)], env=env, check=True)
        settings = json.loads((Path(directory) / "runtime.json").read_text(encoding="utf-8"))
        assert Path(settings["python"]) == python
        subprocess.run([str(binary), "doctor"], env=env, check=True)
        subprocess.run([str(binary), "models"], env=env, check=True)
        subprocess.run([str(binary), "health-checks"], env=env, check=True, stdout=subprocess.DEVNULL)
