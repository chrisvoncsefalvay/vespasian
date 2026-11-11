"""Real local socket and executable-adapter checks without GPU calls."""

import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import unittest

from bridge import Bridge
from python_bridge import is_worker


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bridge = Bridge(self.root)
        self.addCleanup(self.bridge.close)

    def connect(self, secret=None):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2)
        client.connect(str(self.bridge.path))
        self.addCleanup(client.close)
        stream = client.makefile("rwb")
        self.addCleanup(stream.close)
        stream.write(json.dumps({"secret": self.bridge.secret if secret is None else secret}).encode() + b"\n")
        stream.flush()
        return stream

    def request(self, stream, identifier):
        stream.write(json.dumps({"id": identifier, "command": "inspect"}).encode() + b"\n")
        stream.flush()
        request = self.bridge.next(timeout=2)
        self.assertIsNotNone(request)
        return request

    def test_wrong_secret_cannot_submit_a_worker_request(self):
        stream = self.connect(secret="other-session-secret")
        self.assertEqual(stream.readline(), b"")
        self.assertIsNone(self.bridge.next(timeout=0.05))

    def test_connections_receive_separate_private_state_directories(self):
        first = self.request(self.connect(), 1)
        second = self.request(self.connect(), 2)
        self.assertNotEqual(first.state_dir, second.state_dir)
        for request in (first, second):
            self.assertEqual(stat.S_IMODE(request.state_dir.stat().st_mode), 0o700)
            self.assertEqual(request.state_dir.parent, self.root / "engines")
        self.assertEqual(stat.S_IMODE(self.bridge.path.stat().st_mode), 0o600)

    def test_one_worker_retains_state_and_streams_progress_before_results(self):
        stream = self.connect()
        first = self.request(stream, 1)
        first.send({"type": "progress", "id": 1, "message": "measuring"})
        first.send({"type": "result", "id": 1, "ok": True, "data": {"score": 0.25}})
        self.assertEqual(json.loads(stream.readline())["type"], "progress")
        self.assertEqual(json.loads(stream.readline())["data"]["score"], 0.25)
        second = self.request(stream, 2)
        self.assertEqual(first.state_dir, second.state_dir)
        second.send({"type": "result", "id": 2, "ok": True, "data": {}})
        self.assertEqual(json.loads(stream.readline())["id"], 2)

    def test_python_adapter_proxies_complete_jsonl_exchange(self):
        script = Path(__file__).with_name("python_bridge.py")
        worker = self.root / "worker.py"
        env = dict(os.environ, VESPASIAN_REAL_PYTHON=sys.executable,
                   VESPASIAN_WORKER_SOCKET=str(self.bridge.path),
                   VESPASIAN_WORKER_SECRET=self.bridge.secret,
                   VESPASIAN_WORKER_SCRIPT=str(worker))
        process = subprocess.Popen([sys.executable, str(script), "-u", str(worker)],
                                   env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)

        def stop():
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

        self.addCleanup(stop)
        process.stdin.write('{"id":1,"command":"inspect"}\n{"id":2,"command":"inspect"}\n')
        process.stdin.flush()
        states = []
        for identifier in (1, 2):
            request = self.bridge.next(timeout=2)
            self.assertIsNotNone(request)
            self.assertEqual(request.payload["id"], identifier)
            states.append(request.state_dir)
            request.send({"type": "progress", "id": identifier, "message": "ready"})
            request.send({"type": "result", "id": identifier, "ok": True, "data": {}})
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, stderr)
        events = [json.loads(line) for line in stdout.splitlines()]
        self.assertEqual([(event["type"], event["id"]) for event in events],
                         [("progress", 1), ("result", 1), ("progress", 2), ("result", 2)])
        self.assertEqual(states[0], states[1])

    def test_adapter_only_intercepts_the_exact_worker_invocation(self):
        expected = self.root / "worker.py"
        self.assertTrue(is_worker(["-u", str(expected)], str(expected)))
        for args in (["-c", "print('hi')"], [str(expected)], ["-u", str(self.root / "other.py")],
                     ["-u", str(expected), "extra"]):
            self.assertFalse(is_worker(args, str(expected)))


if __name__ == "__main__":
    unittest.main()
