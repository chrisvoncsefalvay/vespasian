"""Queued browser callbacks retain runs while refreshing GPU request context."""

import importlib
import json
import os
import shutil
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
        cls.controls = cls.demo.vespasian_controls
        cls.callbacks = {function.api_name: function for function in cls.demo.fns.values()}

    def test_compact_login_label_comes_only_from_valid_session(self):
        from types import SimpleNamespace
        login = self.controls["login_button"]
        self.assertEqual(login.value, "Not logged in")
        self.assertEqual(login.logout_value, "{}")
        for session, expected in (
            ({}, "Not logged in"),
            ({"hf_token": "pasted-token"}, "Not logged in"),
            ({"oauth_info": {"expires_at": 1, "userinfo": {"preferred_username": "expired-user"}}}, "Not logged in"),
            ({"oauth_info": {"userinfo": {"preferred_username": "session-user"}}}, "session-user"),
            ({"oauth_info": {"userinfo": {"preferred_username": "long-" + "username" * 8}}}, "long-" + "username" * 8),
        ):
            request = SimpleNamespace(session=session, request=SimpleNamespace(session=session))
            self.assertEqual(login._check_login_status(request).value, expected)

    def test_setup_order_and_runtime_defaults(self):
        tabs = [component for component in self.demo.blocks.values() if isinstance(component, self.gr.Tab)]
        setup_tabs = [component.id for component in tabs if component.id in {"target", "behaviour", "data", "editor", "review"}]
        self.assertEqual(setup_tabs, ["target", "behaviour", "data", "editor", "review"])
        self.assertEqual(self.controls["setup_tabs"].selected, "target")
        self.assertEqual(self.controls["budget"].value, 10)
        self.assertEqual(self.controls["max_new_tokens"].value, 128)
        self.assertEqual(self.controls["max_turns"].value, "")
        self.assertEqual(self.controls["reasoning_effort"].value, "low")
        self.assertIn("reasoning_effort", self.app.FIELDS)
        for field, value in (("thinking_mode", "off"), ("max_seq_length", 1024), ("max_relative_norm", 0.1)):
            self.assertIn(field, self.app.FIELDS)
            self.assertEqual(self.controls[field].value, value)
        self.assertEqual(self.app.parse_selection({"turn": 75})["turn"], None)
        self.assertEqual(self.app.parse_selection('{"turn": 75}')["turn"], 75)


    def test_secondary_settings_are_collapsed_and_essentials_remain_available(self):
        by_id = {getattr(block, "elem_id", None): block for block in self.demo.blocks.values()}
        for key in ("vs-advanced-runtime", "vs-advanced-evaluation", "vs-deployment-limits", "vs-patch-backup"):
            self.assertFalse(by_id[key].open)
        for field in ("model", "hf_task", "goal", "editor", "editor_model", "reasoning_effort", "api_key", "budget"):
            self.assertTrue(self.controls[field].visible)
        self.assertFalse(self.controls["patch_backup"].value)
        self.assertFalse(self.controls["patch_create"].value)
        self.assertEqual(self.controls["save_format"].value, "model")

    def test_editor_effort_is_visible_beside_model_not_in_advanced(self):
        effort = self.controls["reasoning_effort"]
        self.assertIs(effort.parent, self.controls["editor_model"].parent)
        self.assertEqual(effort.value, "low")
        self.assertEqual([choice[1] for choice in effort.choices], ["low", "medium", "high"])
        ancestor = effort.parent
        while ancestor is not None:
            self.assertNotIsInstance(ancestor, self.gr.Accordion)
            ancestor = getattr(ancestor, "parent", None)

    def test_inspection_controls_use_operator_limits_and_live_in_advanced(self):
        for field in ("thinking_mode", "max_new_tokens", "max_seq_length", "max_relative_norm"):
            ancestor = self.controls[field].parent
            while ancestor is not None and getattr(ancestor, "elem_id", None) != "vs-advanced-runtime":
                ancestor = getattr(ancestor, "parent", None)
            self.assertIsNotNone(ancestor, field)
            self.assertFalse(ancestor.open)
        with patch.dict(os.environ, {"VESPASIAN_SPACE_MAX_NEW_TOKENS": "32768", "VESPASIAN_SPACE_MAX_SEQ_LENGTH": "131072"}):
            demo, controls = self.app.build_ui()
        self.assertEqual(controls["max_new_tokens"].maximum, 32768)
        self.assertEqual(controls["max_seq_length"].maximum, 131072)
        self.assertEqual(controls["max_new_tokens"].value, 128)
        self.assertEqual(controls["max_seq_length"].value, 1024)
        markup = " ".join(str(getattr(block, "value", "")) for block in demo.blocks.values())
        self.assertIn("32,768 generated tokens", markup)
        self.assertIn("do not guarantee long runs fit ZeroGPU", markup)

    async def test_private_patch_save_requires_explicit_consent(self):
        run, state = self.fixture_run(finished=True)
        function = self.callback("publish_patches_button", name="publish_patches")
        with patch.dict(self.app.runtime.RUNS, {run.id: run}, clear=True), patch.object(self.app.hub, "core_module") as module:
            result = (await self.call(function, self.request(), self.inputs(function, {"state": state, "patch_save_repo": "visitor/patches", "patch_save_consent": False, "hf_token": "visitor-token"})))[0]
        module.assert_not_called()
        self.assertIn("Explicitly authorize", self.returned(function, result, "publish_status"))

    async def test_private_patch_save_uses_session_credentials_and_private_helper(self):
        run, state = self.fixture_run(finished=True)
        function = self.callback("publish_patches_button", name="publish_patches")
        helper = Mock()
        with patch.dict(self.app.runtime.RUNS, {run.id: run}, clear=True), patch.object(self.app.hub, "core_module", return_value=helper):
            result = (await self.call(function, self.request(), self.inputs(function, {"state": state, "patch_save_repo": "visitor/patches", "patch_save_consent": True, "hf_token": "visitor-token"})))[0]
        helper.publish_private.assert_called_once_with(run.output / "patches", "visitor/patches", "visitor-token", consent=True, create=False)
        self.assertNotIn("visitor-token", str(result["data"]))


    async def test_model_tags_suggest_without_overwriting_task_choice(self):
        function = self.callback("model", "input", "model_task_hint")
        self.assertNotIn(self.controls["hf_task"], function.outputs)
        with patch.object(self.app.hub, "model_task_metadata", return_value={"model_pipeline_tag":"image-text-to-text","model_suggested_tasks":["text-generation"]}):
            result = (await self.call(function, self.request(), self.inputs(function)))[0]
        self.assertIn("text-generation", self.returned(function, result, "task_help"))
        self.assertIn("do not prove", self.returned(function, result, "task_help"))

    async def test_private_auto_backup_cannot_use_implicit_owner_credentials(self):
        with patch.object(self.app.hub, "token_for", return_value=False), patch.object(self.app.runtime, "begin") as begin:
            result = (await self.call("run", self.request(), self.inputs("run", {"patch_backup":True,"patch_repo":"visitor/private"}, valid=True)))[0]
        begin.assert_not_called()
        self.assertIn("session Hugging Face token", self.returned("run", result, "error_review"))

    async def test_patch_only_result_does_not_offer_model_publication(self):
        run, state = self.fixture_run(finished=True)
        (run.output / "patches").mkdir()
        (run.output / "report.json").write_text('{"checkpoint":null}')
        with patch.dict(self.app.runtime.RUNS, {run.id:run}, clear=True):
            result = (await self.call("advance", self.request()))[0]
        self.assertFalse(self.returned("advance", result, "publish_model_button")["interactive"])
        self.assertTrue(self.returned("advance", result, "publish_patches_button")["interactive"])

    async def test_target_navigation_validates_model_before_behaviour(self):
        function = self.callback("next_behaviour")
        invalid = (await self.call(function, self.request(), self.inputs(function, {"model": None})))[0]
        self.assertIn("Choose a base model", self.returned(function, invalid, "error_target"))
        valid = (await self.call(function, self.request(), self.inputs(function)))[0]
        self.assertEqual(self.returned(function, valid, "setup_tabs")["selected"], "behaviour")

    async def test_behaviour_preset_preserves_selected_target_model(self):
        function = self.callback("preset", "input", "preset_changed")
        model = self.controls["model"]
        before = model.value
        model.value = "google/gemma-3-1b-it"
        try:
            result = (await self.call(function, self.request(), self.inputs(function, {"preset": "greetings"})))[0]
            self.assertNotIn(self.controls["model"], function.outputs)
            self.assertEqual(model.value, "google/gemma-3-1b-it")
        finally:
            model.value = before

    async def test_quick_starter_loads_task_without_starting_or_touching_credentials(self):
        function = self.callback("starter_submit", name="load_quick_starter")
        with patch.object(self.app.runtime, "begin") as begin:
            for card in self.app.setup.quick_starter_cards():
                result = (await self.call(function, self.request(), self.inputs(function, {"starter_input": card["id"]})))[0]
                config = self.app.setup.preset_config(card["id"])
                self.assertEqual(self.returned(function, result, "goal"), config["goal"])
                self.assertEqual(self.returned(function, result, "hf_task"), config["hf_task"])
                for field in ("prompt_field", "target_field", "text_field"):
                    field_update = self.returned(function, result, field)
                    self.assertEqual(field_update["choices"], [])
                    self.assertEqual(field_update["value"], config[field] or None)
                self.assertEqual(self.returned(function, result, "example_limit"), config["example_limit"])
                self.assertEqual(self.returned(function, result, "setup_tabs")["selected"], "data")
                self.assertEqual(self.session[self.controls["preview_state"]._id], {})
                self.assertIn("Inspect the examples", self.returned(function, result, "starter_status"))
                self.assertFalse(self.returned(function, result, "patch_backup"))
                self.assertFalse(self.returned(function, result, "patch_create"))
                for name in ("model", "editor", "editor_model", "api_key", "hf_token"):
                    self.assertNotIn(self.controls[name], function.outputs)
            begin.assert_not_called()

    async def test_each_quick_starter_starts_the_normal_paid_hill_climb(self):
        for card in self.app.setup.quick_starter_cards():
            with self.subTest(starter=card["id"]):
                loader = self.callback("starter_submit", name="load_quick_starter")
                loaded = (await self.call(loader, self.request(), self.inputs(loader, {"starter_input": card["id"]})))[0]
                file_data = self.returned(loader, loaded, "dataset_file")
                config = self.app.setup.preset_config(card["id"])
                self.assertIsNone(file_data)
                config.update(editor="openai", editor_model="gpt-6.1-sol", api_key="fixture-editor-key")
                preview = self.preview_result()
                preview.update(_current=True, _fingerprint=self.app.setup.fingerprint(config))
                test_preview = self.preview_result("test")
                test_preview.update(_current=True, _fingerprint=self.app.setup.fingerprint(config, test=True))
                config.update(preview_state=preview, test_preview_state=test_preview)
                run, _ = self.fixture_run()
                with patch.object(self.app.runtime, "begin", return_value=run) as begin:
                    result = (await self.call("run", self.request(), self.inputs("run", {**config, "dataset_file": file_data})))[0]
                begin.assert_called_once()
                run.start.assert_called_once()
                started = begin.call_args.args[1]
                self.assertEqual(started["dataset_file"], config["dataset_file"])
                for key in ("goal", "hf_task", "metric", "max_new_tokens", "example_limit"):
                    self.assertEqual(started[key], config[key])
                self.assertEqual(started["budget"], 10)
                self.assertIsNone(started["max_turns"])
                self.assertEqual(started["reasoning_effort"], "low")
                self.assertEqual(started["thinking_mode"], "off")
                self.assertEqual(started["max_seq_length"], 2048)
                self.assertEqual(started["max_relative_norm"], 0.1)
                self.assertEqual(started["editor"], "openai")
                self.assertEqual(started["api_key"], "fixture-editor-key")
                self.assertTrue(started["save_trajectory"])
                self.assertFalse(started["patch_backup"])
                self.assertNotIn("fixture-editor-key", json.dumps(result["data"]))

    async def test_reviewed_inspection_settings_survive_starting_run(self):
        loader = self.callback("preset", "input", "preset_changed")
        loaded = (await self.call(loader, self.request(), self.inputs(loader, {"preset": "parity"})))[0]
        file_data = self.returned(loader, loaded, "dataset_file")
        config = self.app.setup.preset_config("parity")
        config.update(dataset_file=file_data["path"], editor="openai", editor_model="gpt-6.1-sol",
                      api_key="fixture-editor-key", thinking_mode="on", max_new_tokens=128,
                      max_seq_length=512, max_relative_norm=0.25, reasoning_effort="low")
        preview = self.app.hub.preview_upload(config["dataset_file"], config)
        preview.update(_current=True, _fingerprint=self.app.setup.fingerprint(config))
        config.update(preview_state=preview, test_preview_state={})
        run, _ = self.fixture_run()
        with patch.object(self.app.runtime, "begin", return_value=run) as begin:
            await self.call("run", self.request(), self.inputs("run", {**config, "dataset_file": file_data}))
        begin.assert_called_once()
        started = begin.call_args.args[1]
        for field in ("thinking_mode", "max_new_tokens", "max_seq_length", "max_relative_norm", "reasoning_effort"):
            self.assertEqual(started[field], config[field])
        for field in ("thinking_mode", "max_new_tokens", "max_seq_length", "max_relative_norm"):
            self.assertIn('data-vs-focus="#' + self.app.STEP_FIELDS[field] + '"', self.app.error_html({field: "Invalid value"}))

    async def test_preset_output_graph_cannot_replace_credentials_or_selected_model(self):
        for source, event, name in [("starter_submit", "click", "load_quick_starter"), ("preset", "input", "preset_changed")]:
            function = self.callback(source, event, name=name)
            for field in ("model", "editor", "editor_model", "api_key", "hf_token", "input_price", "output_price"):
                self.assertNotIn(self.controls[field], function.outputs)

    async def test_invalid_quick_starter_preserves_every_setting(self):
        function = self.callback("starter_submit", name="load_quick_starter")
        result = (await self.call(function, self.request(), self.inputs(function, {"starter_input": '<script>not-a-starter</script>'})))[0]
        self.assertIn("No settings were changed", self.returned(function, result, "starter_status"))
        self.assertNotIn("<script>", self.returned(function, result, "starter_status"))
        for name in self.app.FIELDS:
            if self.controls[name] in function.outputs:
                self.assertNotIn("value", self.returned(function, result, name))

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
        run.view_selection = dict(self.app.DEFAULT_SELECTION)
        state = {"run_id": run.id, "events": [], "status": "Complete" if finished else "Running",
                 "config": {"budget": 1.0}, "error": None, "finished": finished, "elapsed": 0.0}
        run.state.return_value = state
        run.advance.side_effect = lambda dispatch: iter([state])
        self.session[self.callbacks["advance"].inputs[0]._id] = state
        return run, state

    def callback(self, control, event="click", name=None):
        target = self.controls[control]._id
        found = [function for function in self.demo.fns.values()
                 if (target, event) in function.get_config()["targets"]
                 and (name is None or function.fn.__name__ == name)]
        self.assertEqual(len(found), 1, (control, event, name))
        return found[0]

    def inputs(self, function, overrides=None, valid=False):
        function = self.callbacks[function] if isinstance(function, str) else function
        values = {component: component.value for component in function.inputs}
        updates = {"hf_task": "text-generation", "goal": "Answer greetings with Salve!", "dataset_source": "openai/gsm8k",
                   "dataset_config": "main", "prompt_field": "question", "target_field": "answer",
                   "api_key": "visitor-editor-key"} if valid else {}
        updates.update(overrides or {})
        for name, value in updates.items():
            if self.controls[name] in values:
                values[self.controls[name]] = value
        config = {name: values[component] for name,component in self.controls.items() if component in values}
        if valid and "preview_state" not in updates:
            values[self.controls["preview_state"]] = {
                "repo": "openai/gsm8k", "config": "main", "split": "train", "_current": True,
                "_fingerprint": self.app.setup.fingerprint(config),
                "mapped_rows": [{"prompt": "Hello", "target": "Salve!"}],
            }
        for component, value in values.items():
            if isinstance(component, self.gr.State):
                self.session[component._id] = value
        return [None if isinstance(component,self.gr.State) else values[component] for component in function.inputs]

    def returned(self, function, result, name):
        function = self.callbacks[function] if isinstance(function, str) else function
        return result["data"][function.outputs.index(self.controls[name])]

    async def call(self, name, request, inputs=None):
        function = self.callbacks[name] if isinstance(name,str) else name
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

    def test_status_and_cancel_are_public_supported_apis(self):
        api = self.demo.get_api_info()["named_endpoints"]
        for name in ("/status", "/cancel"):
            self.assertIn(name, api)
            self.assertEqual(api[name]["parameters"], [])
        for name in ("status", "cancel"):
            self.assertFalse(self.callbacks[name].queue)
            self.assertEqual(self.callbacks[name].api_visibility, "public")

    async def test_status_and_cancel_require_same_owner_and_status_has_no_side_effects(self):
        runtime = self.app.runtime
        run = runtime.Run("visitor-a", {"api_key": "private-key", "goal": "private-key"})
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        self.session[self.controls["state"]._id] = {"run_id": run.id}
        run.last_seen = 42
        with patch.dict(runtime.RUNS, {run.id: run}, clear=True), \
             patch.object(run, "start", side_effect=AssertionError("frontier")), \
             patch.object(run, "advance", side_effect=AssertionError("GPU")), \
             patch.object(run, "drain", side_effect=AssertionError("mutation")):
            result = (await self.call("status", self.request()))[-1]["data"]
            result[0] = result[0].root
            self.assertFalse(result[0]["finished"])
            self.assertEqual(result[1:], [None, None])
            self.assertEqual(run.last_seen, 42)
            self.assertNotIn("private-key", json.dumps(result))
            for name in ("status", "cancel"):
                for owner in ("visitor-b", None, ""):
                    with self.assertRaisesRegex(ValueError, "^Start a run in this session first$"):
                        await self.call(name, self.request(owner=owner))
            self.assertFalse(run.cancelled.is_set())
            await self.call("cancel", self.request())
            await self.call("cancel", self.request())
            self.assertTrue(run.cancelled.is_set())
            self.assertEqual(run.last_seen, 42)

    async def test_terminal_status_delivers_artifacts_without_restart(self):
        runtime = self.app.runtime
        run = runtime.Run("visitor-a", {})
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        run.output.mkdir()
        (run.output / "report.json").write_text('{"reload_verified":true}')
        run.bundle()
        run.load_report()
        run.finished = True
        run.finished_at = run.started + 1
        self.session[self.controls["state"]._id] = {"run_id": run.id}
        with patch.dict(runtime.RUNS, {run.id: run}, clear=True), \
             patch.object(run, "advance", side_effect=AssertionError("GPU")):
            result = (await self.call("status", self.request()))[-1]["data"]
            result[0] = result[0].root
        self.assertTrue(result[0]["report"]["reload_verified"])
        self.assertTrue(result[0]["finished"])
        self.assertEqual(result[1]["orig_name"], run.archive.name)
        self.assertIsNone(result[2])

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
        inputs = self.inputs(function, {"hf_token": "visitor-hub-token"}, valid=True)
        with patch.object(self.app.runtime, "begin", return_value=run) as begin, \
                patch.object(self.app, "dispatch") as dispatch:
            results = await self.call("run", self.request(), inputs)
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
        self.assertIsNone(begin.call_args.args[1]["health_max_drop"])
        self.assertNotIn("preview_state", begin.call_args.args[1])

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
        (self.output / "checkpoint").mkdir(exist_ok=True)
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

    async def test_empty_start_identifies_required_field_without_controller_or_monitor_error(self):
        with patch.object(self.app.runtime,"begin") as begin, patch.object(self.app,"dispatch") as dispatch:
            results = await self.call("run",self.request(),self.inputs("run"))
        result = results[0]
        self.assertIn("Describe the behaviour",self.returned("run",result,"error_behaviour"))
        self.assertEqual(self.returned("run",result,"setup_tabs")["selected"],"behaviour")
        unchanged = self.returned("run",result,"dashboard_html")
        self.assertEqual(unchanged["__type__"],"update")
        self.assertNotIn("value",unchanged)
        self.assertNotIn("value",unchanged.get("props",{}))
        self.assertEqual(self.returned("run",result,"run_timer"),{"__type__":"update"})
        begin.assert_not_called()
        dispatch.assert_not_called()

    async def test_navigation_stays_on_invalid_step_and_advances_when_complete(self):
        function = self.callback("next_data")
        with patch.object(self.app.runtime,"begin") as begin:
            invalid = (await self.call(function,self.request(),self.inputs(function)))[0]
            self.assertIn("Describe the behaviour",self.returned(function,invalid,"error_behaviour"))
            self.assertEqual(self.returned(function,invalid,"setup_tabs"),{"__type__":"update"})
            valid = (await self.call(function,self.request(),self.inputs(function,{"goal":"Answer briefly"})))[0]
        self.assertEqual(self.returned(function,valid,"setup_tabs")["selected"],"data")
        begin.assert_not_called()

    async def test_editor_navigation_validates_required_key(self):
        function = self.callback("next_review")
        result = (await self.call(function,self.request(),self.inputs(function,{"api_key":""})))[0]
        self.assertIn("API key",self.returned(function,result,"error_editor"))
        self.assertEqual(self.returned(function,result,"setup_tabs"),{"__type__":"update"})

    async def test_start_rejects_preview_that_no_longer_matches_configuration(self):
        inputs = self.inputs("run",valid=True)
        self.session[self.controls["preview_state"]._id]["_fingerprint"] = "previous-repository"
        with patch.object(self.app.runtime,"begin") as begin:
            result = (await self.call("run",self.request(),inputs))[0]
        self.assertIn("Preview the current dataset",self.returned("run",result,"error_data"))
        self.assertEqual(self.returned("run",result,"setup_tabs")["selected"],"data")
        begin.assert_not_called()

    async def test_explicit_health_limit_converts_percentage_points_once(self):
        run,_ = self.fixture_run()
        inputs = self.inputs("run",{"health_policy":"enforce","health_limit_pp":10},valid=True)
        with patch.object(self.app.runtime,"begin",return_value=run) as begin:
            await self.call("run",self.request(),inputs)
        self.assertEqual(begin.call_args.args[1]["health_max_drop"],.1)

    async def test_mode_change_limits_metrics_without_overwriting_objective(self):
        function = self.callback("dataset_mode","change","mode_changed")
        result = (await self.call(function,self.request(),self.inputs(function,{"dataset_mode":"corpus","metric":"exact_match","target_type":"factual"})))[0]
        metric = self.returned(function,result,"metric")
        self.assertEqual(metric["value"],"semantic")
        self.assertEqual(metric["choices"],[["Goal score","semantic"]])
        self.assertEqual(self.returned(function,result,"target_type")["value"],"style")
        self.assertNotIn(self.controls["goal"],function.outputs)

    def preview_result(self,split="train"):
        return {"task_metadata":{"selected_task":"text-generation","suggested_tasks":["text-generation"]}, "repo":"openai/gsm8k","config":"main","split":split,"configs":["main"],
                "splits":["train","test"],"prompt_field":"question","target_field":"answer",
                "rows":[{"question":"Question?","answer":"Answer"}],
                "mapped_rows":[{"row_index":0,"prompt":"Question?","target":"Answer"}]}

    async def test_training_and_test_previews_have_independent_state_and_outputs(self):
        function = self.callbacks["preview_dataset"]
        with patch.object(self.app.hub,"preview",return_value=self.preview_result()):
            await self.call(function,self.request(),self.inputs(function,{"dataset_source":"openai/gsm8k"}))
        stored = dict(self.session[self.controls["preview_state"]._id])
        self.assertEqual(stored["split"],"train")
        self.assertTrue(stored["_current"])
        function = self.callbacks["preview_test_dataset"]
        with patch.object(self.app.hub,"preview",return_value=self.preview_result("test")):
            heldout = (await self.call(function,self.request(),self.inputs(function,{"test_dataset_source":"openai/gsm8k","test_dataset_config":"main"})))[0]
        self.assertEqual(self.session[self.controls["preview_state"]._id],stored)
        self.assertEqual(self.session[self.controls["test_preview_state"]._id]["split"],"test")
        self.assertNotIn(self.controls["preview_html"],function.outputs)
        self.assertIn("Held-out test",self.returned(function,heldout,"test_preview_html"))

    async def test_repository_edit_invalidates_only_its_own_preview(self):
        old = {"_current":True,"repo":"old/repository"}
        self.session[self.controls["preview_state"]._id] = dict(old)
        self.session[self.controls["test_preview_state"]._id] = dict(old)
        function = self.callback("dataset_source","input")
        result = (await self.call(function,self.request(),[]))[0]
        self.assertEqual(self.session[self.controls["preview_state"]._id],{})
        self.assertEqual(self.session[self.controls["test_preview_state"]._id],old)
        self.assertIn("Refresh",self.returned(function,result,"preview_html"))

    async def test_mapping_edit_invalidates_both_previews(self):
        for key in ("preview_state","test_preview_state"):
            self.session[self.controls[key]._id] = {"_current":True}
        function = self.callback("prompt_field","input")
        await self.call(function,self.request(),[])
        for key in ("preview_state","test_preview_state"):
            self.assertEqual(self.session[self.controls[key]._id],{})

    async def test_turn_selection_is_read_only_and_owned_by_session(self):
        run,_ = self.fixture_run()
        function = self.callback("selection_submit")
        inputs = [None,json.dumps({"turn":0,"follow":False,"split":"validation"}),None]
        with patch.dict(self.app.runtime.RUNS,{run.id:run},clear=True), patch.object(self.app,"dispatch") as dispatch:
            await self.call(function,self.request(),inputs)
            self.assertFalse(run.view_selection["follow"])
            self.assertEqual(run.view_selection["split"],"validation")
            owned = dict(run.view_selection)
            with self.assertRaisesRegex(ValueError,"this session"):
                await self.call(function,self.request(owner="visitor-b"),[None,json.dumps({"follow":True}),None])
            self.assertEqual(run.view_selection,owned)
        dispatch.assert_not_called()
        run.advance.assert_not_called()

    async def test_incoming_advance_preserves_the_current_pinned_turn(self):
        run,_ = self.fixture_run()
        run.view_selection = {**self.app.DEFAULT_SELECTION,"turn":0,"follow":False,"split":"validation"}
        self.session[self.controls["selection"]._id] = dict(self.app.DEFAULT_SELECTION)
        with patch.dict(self.app.runtime.RUNS,{run.id:run},clear=True):
            result = (await self.call("advance",self.request()))[0]
        rendered = self.returned("advance",result,"dashboard_html")
        self.assertIn('data-vs-follow="false"',rendered)
        self.assertIn('data-vs-selected-turn="0"',rendered)
        self.assertEqual(run.view_selection["split"],"validation")

    async def test_pricing_callback_matches_selected_provider_model(self):
        function = self.callback("editor_model","change","pricing")
        result = (await self.call(function,self.request(),self.inputs(function,{"editor":"openai","editor_model":"gpt-6.1-sol"})))[0]
        text = self.returned(function,result,"pricing_html")
        self.assertIn("$2",text)
        self.assertIn("$10",text)
        self.assertNotIn("None",text)

    async def test_unsupported_pricing_is_explained_without_editing_credentials(self):
        function = self.callback("editor_model", "change", "pricing")
        for updates in ({"editor_model": "custom-editor"}, {"input_price": 2}, {"output_price": 10},
                        {"editor": "anthropic", "editor_model": "gpt-6.1-sol"}):
            values = {"editor": "openai", "editor_model": "gpt-6.1-sol", "input_price": "", "output_price": "", **updates}
            result = (await self.call(function, self.request(), self.inputs(function, values)))[0]
            message = self.returned(function, result, "pricing_html")
            self.assertIn("--budget-policy auto", message)
            self.assertIn("strict", message)
            self.assertNotIn("Add input and output", message)
        for name in ("api_key", "editor", "editor_model"):
            self.assertNotIn(self.controls[name], function.outputs)
        for name in ("input_price", "output_price"):
            self.assertFalse(self.controls[name].visible)

    async def test_provider_switch_remembers_model_and_clears_credentials_and_overrides(self):
        function = self.callback("editor", "input", "editor_changed")
        memory = {"_provider": "openai", "anthropic": "claude-opus-5-5"}
        result = (await self.call(function, self.request(), self.inputs(function, {
            "editor": "anthropic", "editor_model": "custom-openai", "provider_memory": memory,
        })))[0]
        self.assertEqual(self.returned(function, result, "editor_model")["value"], "claude-opus-5-5")
        saved = self.session[self.controls["provider_memory"]._id]
        self.assertEqual(saved["openai"], "custom-openai")
        self.assertEqual(saved["_provider"], "anthropic")
        for name in ("api_key", "input_price", "output_price"):
            self.assertEqual(self.returned(function, result, name), "")
        result = (await self.call(function, self.request(), self.inputs(function, {
            "editor": "openai", "editor_model": "claude-opus-5-5", "provider_memory": saved,
        })))[0]
        self.assertEqual(self.returned(function, result, "editor_model")["value"], "custom-openai")

    async def test_model_selection_clears_only_price_overrides(self):
        function = self.callback("editor_model", "input")
        result = (await self.call(function, self.request(), []))[0]
        for name in ("input_price", "output_price"):
            self.assertEqual(self.returned(function, result, name), "")
        self.assertNotIn(self.controls["api_key"], function.outputs)

    async def test_token_edit_clears_account_and_both_previews(self):
        for name in ("preview_state", "test_preview_state"):
            self.session[self.controls[name]._id] = {"_current": True, "repo": "old/private"}
        self.session[self.controls["account_choices"]._id] = ["old/private"]
        account = self.callback("hf_token", "input", "<lambda>")
        result = (await self.call(account, self.request(), []))[0]
        self.assertEqual(self.session[self.controls["account_choices"]._id], [])
        self.assertIn("Press Enter", self.returned(account, result, "profile_html"))
        invalidate = self.callback("hf_token", "input", "callback")
        result = (await self.call(invalidate, self.request(), []))[0]
        for name in ("preview_state", "test_preview_state"):
            self.assertEqual(self.session[self.controls[name]._id], {})
        for name in ("preview_html", "test_preview_html"):
            self.assertIn("Refresh", self.returned(invalidate, result, name))

    async def test_account_strip_distinguishes_token_override_from_session(self):
        function = self.callback("hf_token", "submit", "account")
        for override, expected in (("  ", '<div class="vs-profile"></div>'), ("override-token", '<div class="vs-profile"><p>Token: visitor</p></div>')):
            with patch.object(self.app.hub, "account_datasets", return_value=("visitor", [])):
                result = (await self.call(function, self.request(), self.inputs(function, {"hf_token": override})))[0]
            self.assertEqual(self.returned(function, result, "profile_html"), expected)

    async def test_token_submit_refreshes_account_with_selected_token(self):
        function = self.callback("hf_token", "submit", "account")
        with patch.object(self.app.hub, "account_datasets", return_value=("visitor", ["visitor/private"])) as account:
            result = (await self.call(function, self.request(), self.inputs(function, {"hf_token": "visitor-token"})))[0]
        account.assert_called_once_with("visitor-token")
        self.assertEqual(self.session[self.controls["account_choices"]._id], ["visitor/private"])
        self.assertIn("visitor", self.returned(function, result, "profile_html"))
        self.assertNotIn("visitor-token", json.dumps(result["data"]))

    async def test_detected_mapping_invalidates_test_preview_only_when_mapping_changes(self):
        function = self.callbacks["preview_dataset"]
        old = {"_current": True, "repo": "test/data", "_fingerprint": "previous"}
        self.session[self.controls["test_preview_state"]._id] = dict(old)
        with patch.object(self.app.hub, "preview", return_value=self.preview_result()):
            changed = (await self.call(function, self.request(), self.inputs(function, {
                "hf_task": "text-generation", "dataset_source": "openai/gsm8k", "prompt_field": None, "target_field": None,
            })))[0]
        self.assertEqual(self.session[self.controls["test_preview_state"]._id], {})
        self.assertIn("Refresh", self.returned(function, changed, "test_preview_html"))
        self.session[self.controls["test_preview_state"]._id] = dict(old)
        with patch.object(self.app.hub, "preview", return_value=self.preview_result()):
            unchanged = (await self.call(function, self.request(), self.inputs(function, {
                "hf_task": "text-generation", "dataset_source": "openai/gsm8k", "prompt_field": "question", "target_field": "answer",
            })))[0]
        self.assertEqual(self.session[self.controls["test_preview_state"]._id], old)
        skipped = self.returned(function, unchanged, "test_preview_html")
        self.assertEqual(skipped["__type__"], "update")
        self.assertNotIn("value", skipped)

    async def test_preview_fingerprint_survives_blank_mapping_roundtrip_to_navigation(self):
        function = self.callbacks["preview_dataset"]
        with patch.object(self.app.hub, "preview", return_value=self.preview_result()):
            await self.call(function, self.request(), self.inputs(function, {
                "hf_task": "text-generation", "dataset_source": "openai/gsm8k", "text_field": "",
            }))
        preview = dict(self.session[self.controls["preview_state"]._id])
        function = self.callback("next_editor")
        result = (await self.call(function, self.request(), self.inputs(function, {
            "preview_state": preview, "text_field": None,
        }, valid=True)))[0]
        self.assertEqual(self.returned(function, result, "setup_tabs")["selected"], "editor")
        self.assertEqual(self.returned(function, result, "error_data"), "")

    async def test_loading_preset_replaces_data_and_invalidates_existing_previews(self):
        for name in ("preview_state", "test_preview_state"):
            self.session[self.controls[name]._id] = {"_current": True, "repo": "previous/data"}
        function = self.callback("preset", "input", "preset_changed")
        result = (await self.call(function, self.request(), self.inputs(function, {"preset": "gsm8k"})))[0]
        self.assertEqual(self.returned(function, result, "dataset_source"), "openai/gsm8k")
        self.assertEqual(self.returned(function, result, "metric"), "numeric_match")
        self.assertEqual(self.returned(function, result, "example_limit"), 32)
        self.assertTrue(self.returned(function, result, "goal"))
        for name in ("preview_state", "test_preview_state"):
            self.assertEqual(self.session[self.controls[name]._id], {})


if __name__ == "__main__":
    unittest.main()
