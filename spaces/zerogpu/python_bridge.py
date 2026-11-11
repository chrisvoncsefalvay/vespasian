#!/usr/bin/env python3
"""Python executable adapter used only by the Space's Rust controller."""

import json
import os
from pathlib import Path
import socket
import sys


def is_worker(args, expected):
    return bool(expected and len(args) == 2 and args[0] == "-u"
                and Path(args[1]).resolve() == Path(expected).resolve())


def main():
    args = sys.argv[1:]
    interpreter = os.environ.get("VESPASIAN_REAL_PYTHON", sys.executable)
    if not is_worker(args, os.environ.get("VESPASIAN_WORKER_SCRIPT")):
        os.execv(interpreter, [interpreter, *args])
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(os.environ["VESPASIAN_WORKER_SOCKET"])
        stream = client.makefile("rwb")
        stream.write(json.dumps({"secret": os.environ["VESPASIAN_WORKER_SECRET"]}).encode() + b"\n")
        stream.flush()
        for line in sys.stdin.buffer:
            stream.write(line)
            stream.flush()
            while True:
                response = stream.readline(32 * 1024 * 1024 + 1)
                if not response:
                    raise RuntimeError("The model connection closed")
                if len(response) > 32 * 1024 * 1024:
                    raise RuntimeError("Model response exceeds the message limit")
                value = json.loads(response)
                sys.stdout.buffer.write(response)
                sys.stdout.buffer.flush()
                if value.get("type") == "result":
                    break


if __name__ == "__main__":
    main()
