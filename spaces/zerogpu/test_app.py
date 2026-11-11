"""Queued browser callbacks retain runs while refreshing GPU request context."""

import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class AppRequestTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {
            "VESPASIAN_SPACE_SKIP_MODEL_LOAD": "1",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "SYSTEM": "spaces",
            "SPACE_ID": "test/vespasian",
            "OAUTH_CLIENT_ID": "test-client",
            "OAUTH_CLIENT_SECRET": "test-placeholder",
            "OAUTH_SCOPES": "openid profile read-repos gated-repos write-repos",
            "OPENID_PROVIDER_URL": "https://huggingface.co",
        })
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        cls.app = importlib.import_module("app")
        cls.gr = importlib.import_module("gradio")
        cls.context = importlib.import_module("gradio.context").LocalContext
        cls.SessionState = importlib.import_module("gradio.state_holder").SessionState
        cls.demo = cls.app.create_app()
        cls.callbacks = {function.api_name: function for function in cls.demo.fns.values()}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.session = self.SessionState(self.demo)

    def request(self, owner="visitor-a", token="fixture-token"):
        return self.gr.Request(session_hash=owner, headers={"x-ip-token": token})

    def fixture_run(self, *, finished=False):
        run = Mock()
        run.id = "fixture-run"
        run.owner = "visitor-a"
        run.output = self.output
        run.archive = None
        state = {"run_id": run.id, "events": [], "status": "Complete" if finished else "Running",
                 "config": {"budget": 1.0}, "error": None, "finished": finished, "elapsed": 0.0}
        run.state.return_value = state
        run.advance.side_effect = lambda dispatch: iter([state])
        self.session[self.callbacks["advance"].inputs[0]._id] = state
        return run, state

    async def call(self, name, request, inputs=None):
        function = self.callbacks[name]
        if inputs is None:
            inputs = [None for _ in function.inputs]
        result = await self.demo.process_api(
            function, inputs=inputs, state=self.session, request=request,
            session_hash=request.session_hash,
        )
        results = [result]
        while result["is_generating"]:
            result = await self.demo.process_api(
                function, inputs=inputs, state=self.session, request=request,
                session_hash=request.session_hash, iterator=result["iterator"],
            )
            results.append(result)
        return results

    def test_timer_uses_new_queued_events_without_building_a_backlog(self):
        function = self.callbacks["advance"]
        config = function.get_config()
        timer = self.demo.blocks[config["targets"][0][0]]
        self.assertIsInstance(timer, self.gr.Timer)
        self.assertFalse(timer.active)
        self.assertEqual(timer.value, 0.5)
        self.assertTrue(config["queue"])
        self.assertEqual(config["trigger_mode"], "once")
        self.assertEqual(config["api_visibility"], "undocumented")
        self.assertGreaterEqual(function.concurrency_limit, self.app.runtime.MAX_ACTIVE)
        self.assertNotEqual(function.concurrency_id, self.callbacks["preview_dataset"].concurrency_id)
        self.assertFalse(self.callbacks["run"].get_config()["types"]["generator"])

    async def test_start_returns_without_scheduling_gpu_work(self):
        run, _ = self.fixture_run()
        function = self.callbacks["run"]
        values = {component: component.value for component in function.inputs}
        for component in function.inputs:
            if component.label == "Editor API key":
                values[component] = "visitor-editor-key"
            elif component.label == "Hugging Face token override":
                values[component] = "visitor-hub-token"
        with patch.object(self.app.runtime, "begin", return_value=run) as begin, \
                patch.object(self.app, "dispatch") as dispatch:
            results = await self.call("run", self.request(), [values[component] for component in function.inputs])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["is_generating"])
        self.assertTrue(results[0]["data"][-1]["active"])
        run.start.assert_called_once()
        run.advance.assert_not_called()
        dispatch.assert_not_called()
        self.assertEqual(begin.call_args.args[0], "visitor-a")
        self.assertEqual(begin.call_args.args[2], "visitor-hub-token")
        self.assertNotIn("visitor-editor-key", json.dumps(results[0]["data"]))
        self.assertNotIn("visitor-hub-token", json.dumps(results[0]["data"]))

    async def test_successive_allocations_receive_each_new_request_context(self):
        run, state = self.fixture_run()
        observed = []

        def scheduled(state_dir, payload):
            request = self.context.request.get()
            observed.append((request.session_hash, request.headers["x-ip-token"]))
            return []

        def advance(dispatch):
            list(dispatch("fixture-state", {"command": "evaluate"}))
            yield state

        run.advance.side_effect = advance
        with patch.dict(self.app.runtime.RUNS, {run.id: run}, clear=True), \
                patch.object(self.app, "dispatch", side_effect=scheduled):
            for token in ("first-request", "second-request", "third-request"):
                results = await self.call("advance", self.request(token=token))
                self.assertTrue(results[0]["data"][-1]["active"])
        self.assertEqual(observed, [("visitor-a", token) for token in
                                   ("first-request", "second-request", "third-request")])
        self.assertEqual(run.advance.call_count, 3)
        run.start.assert_not_called()

    async def test_completion_stops_timer_and_enables_saved_results(self):
        run, _ = self.fixture_run(finished=True)
        (self.output / "report.json").write_text("{}")
        (self.output / "trajectory.jsonl").write_text("{}\n")
        with patch.dict(self.app.runtime.RUNS, {run.id: run}, clear=True):
            results = await self.call("advance", self.request())
        data = results[0]["data"]
        self.assertFalse(data[-1]["active"])
        self.assertTrue(data[4]["interactive"])
        self.assertFalse(data[5]["interactive"])
        self.assertTrue(data[6]["interactive"])
        self.assertTrue(data[7]["interactive"])

    async def test_other_session_cannot_advance_a_known_run(self):
        run, _ = self.fixture_run()
        with patch.dict(self.app.runtime.RUNS, {run.id: run}, clear=True):
            with self.assertRaisesRegex(self.gr.Error, "this session"):
                await self.call("advance", self.request(owner="visitor-b"))
        run.advance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
