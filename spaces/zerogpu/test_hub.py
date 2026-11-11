"""Visitor-scoped dataset and publication contracts without Hub requests."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import hub


class HubTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.revision = "a" * 40
        self.rows = [{"prompt": f"Question {i}", "target": f"Answer {i}"} for i in range(8)]
        self.api = Mock()
        self.api.dataset_info.return_value = SimpleNamespace(sha=self.revision)
        self.client = Mock(return_value=self.api)
        self.datasets = SimpleNamespace(
            __version__="test",
            DownloadConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            load_dataset=Mock(side_effect=lambda *args, **kwargs: iter(self.rows)),
            get_dataset_config_names=Mock(return_value=["default"]),
            get_dataset_split_names=Mock(return_value=["train", "test"]),
        )
        self.add_operation = Mock()
        self.delete_operation = Mock()
        dependencies = patch.dict(sys.modules, {
            "datasets": self.datasets,
            "huggingface_hub": SimpleNamespace(HfApi=self.client,
                CommitOperationAdd=self.add_operation, CommitOperationDelete=self.delete_operation),
        })
        dependencies.start()
        self.addCleanup(dependencies.stop)
        self.real_subprocess = hub._dataset_subprocess
        isolated = patch.object(hub, "_dataset_subprocess", side_effect=lambda kind, config, token=False, directory=None, test=False:
                                hub._preview_local(config, token) if kind == "preview" else hub._fetch_local(config, directory, token, test=test))
        isolated.start()
        self.addCleanup(isolated.stop)

    def test_dataset_process_isolates_hub_metadata_and_credentials(self):
        def run(command, **kwargs):
            env = kwargs["env"]
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("editor-secret", kwargs["input"])
            self.assertNotIn("visitor", kwargs["input"])
            self.assertEqual(env["HF_TOKEN"], "visitor")
            self.assertEqual(env["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
            self.assertTrue(Path(env["HF_HOME"]).is_dir())
            self.assertTrue(Path(env["HF_HUB_CACHE"]).is_relative_to(env["HF_HOME"]))
            self.assertTrue(Path(env["HF_DATASETS_CACHE"]).is_relative_to(env["HF_HOME"]))
            self.cache = env["HF_HOME"]
            return SimpleNamespace(returncode=0, stdout='{"ok":true,"result":{"rows":[]}}')
        with patch.dict(os.environ, {"HF_TOKEN": "owner", "OPENAI_API_KEY": "owner-api"}), patch.object(hub.subprocess, "run", side_effect=run):
            self.assertEqual(self.real_subprocess("preview", {"api_key": "editor-secret", "hf_token": "visitor"}, "visitor"), {"rows": []})
        self.assertFalse(Path(self.cache).exists())

    def test_no_login_explicitly_disables_ambient_owner_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_owner_secret"}):
            token = hub.token_for()
            self.assertIs(token, False)
            result = hub.preview({"dataset_source": "alice/private-data"}, token)
        self.client.assert_called_once_with(token=False)
        self.assertIs(self.datasets.load_dataset.call_args.kwargs["token"], False)
        self.assertIs(self.datasets.get_dataset_config_names.call_args.kwargs["token"], False)
        self.assertIs(self.datasets.get_dataset_split_names.call_args.kwargs["token"], False)
        self.assertNotIn("hf_owner_secret", json.dumps(result))

    def test_explicit_token_override_and_oauth_precedence(self):
        oauth = SimpleNamespace(token="hf_visitor_oauth")
        self.assertEqual(hub.token_for(oauth), "hf_visitor_oauth")
        self.assertEqual(hub.token_for(oauth, "  hf_override  "), "hf_override")
        self.assertEqual(hub.token_for(oauth, "  "), "hf_visitor_oauth")
        self.assertIs(hub.token_for(SimpleNamespace(token=None)), False)

    def test_private_preview_passes_visitor_token_and_pinned_revision_everywhere(self):
        result = hub.preview({"dataset_source": "https://huggingface.co/datasets/alice/private-data"}, "hf_visitor")
        self.assertIsNone(result["error"])
        self.assertIsNone(result["mapping_error"])
        self.assertEqual(result["revision"], self.revision)
        self.assertEqual(len(result["rows"]), 3)
        self.client.assert_called_once_with(token="hf_visitor")
        self.api.dataset_info.assert_called_once_with(repo_id="alice/private-data", revision=None)
        for call in (self.datasets.load_dataset, self.datasets.get_dataset_config_names, self.datasets.get_dataset_split_names):
            self.assertEqual(call.call_args.kwargs["token"], "hf_visitor")
            self.assertEqual(call.call_args.kwargs["revision"], self.revision)
        cache = self.datasets.load_dataset.call_args.kwargs["cache_dir"]
        self.assertTrue(Path(cache).name.startswith("vs-preview-"))
        self.assertFalse(Path(cache).exists())

    def test_inaccessible_private_dataset_does_not_fall_back_or_leak_token(self):
        self.api.dataset_info.side_effect = RuntimeError("HTTP 401 hf_visitor")
        result = hub.preview({"dataset_source": "alice/private-data"}, "hf_visitor")
        self.assertIn("access", result["error"])
        self.assertEqual(result["rows"], [])
        self.assertNotIn("hf_visitor", json.dumps(result))
        self.datasets.load_dataset.assert_not_called()
        self.client.assert_called_once_with(token="hf_visitor")

    def test_fetch_writes_canonical_rows_and_receipt_without_token(self):
        config = {"dataset_source": "alice/private-data", "example_limit": 5}
        hub.fetch_source(config, self.root, "hf_visitor")
        rows = [json.loads(line) for line in (self.root / "training.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["prompt"], "Question 0")
        self.client.assert_called_once_with(token="hf_visitor")
        self.assertEqual(self.datasets.load_dataset.call_args.kwargs["token"], "hf_visitor")
        self.assertEqual(self.datasets.load_dataset.call_args.kwargs["revision"], self.revision)
        receipt = (self.root / "training-source.json").read_text()
        self.assertIn(self.revision, receipt)
        self.assertNotIn("hf_visitor", receipt)

    def test_account_picker_requires_explicit_visitor_auth(self):
        self.assertEqual(hub.account_datasets(False), (None, []))
        self.client.assert_not_called()
        self.api.whoami.return_value = {"name": "alice", "orgs": [{"name": "research"}]}
        self.api.list_datasets.side_effect = [
            [SimpleNamespace(id="alice/private"), SimpleNamespace(id="alice/public")],
            [SimpleNamespace(id="research/private"), SimpleNamespace(id="alice/private")],
        ]
        name, choices = hub.account_datasets("hf_visitor")
        self.assertEqual(name, "alice")
        self.assertEqual(choices, ["alice/private", "alice/public", "research/private"])
        self.client.assert_called_once_with(token="hf_visitor")
        self.assertEqual([call.kwargs["author"] for call in self.api.list_datasets.call_args_list], ["alice", "research"])
        for call in self.api.list_datasets.call_args_list:
            self.assertEqual(call.kwargs["sort"], "last_modified")
            self.assertNotIn("direction", call.kwargs)

    def test_publishing_requires_visitor_auth_before_importing_publisher(self):
        with patch.object(hub, "core_module") as load:
            with self.assertRaisesRegex(ValueError, "Sign in"):
                hub.publish_run(self.root, kind="model", repo="alice/model", visibility="private", create=False, token=False)
        load.assert_not_called()
        self.client.assert_not_called()

    def test_both_publication_paths_inject_explicit_api_and_commit_operations(self):
        for kind, module_name in (("model", "publish"), ("trajectory", "trajectory")):
            with self.subTest(kind=kind):
                self.client.reset_mock()
                module = SimpleNamespace(publish=Mock(return_value={"commit": "b" * 40}))
                with patch.object(hub, "core_module", return_value=module) as load:
                    result = hub.publish_run(self.root, kind=kind, repo="alice/output", visibility="private", create=False, token="hf_visitor")
                load.assert_called_once_with(module_name)
                self.client.assert_called_once_with(token="hf_visitor")
                kwargs = module.publish.call_args.kwargs
                self.assertIs(kwargs["api"], self.api)
                self.assertIs(kwargs["operation"], self.add_operation)
                self.assertEqual(kwargs["visibility"], "private")
                self.assertFalse(kwargs["create"])
                if kind == "model":
                    self.assertIs(kwargs["delete_operation"], self.delete_operation)
                self.assertEqual(result["commit"], "b" * 40)

    def test_dataset_html_escapes_metadata_and_values(self):
        rendered = hub.preview_html({
            "repo": '<script>alert("repo")</script>', "config": '<img src=x onerror="x">',
            "rows": [{"<field>": '<script>alert("row")</script>'}],
            "mapped_rows": [{"prompt": "<svg/onload=x>", "target": "<b>reference</b>"}],
            "mapping_error": "<iframe src=x>",
        })
        for tag in ("<script", "<img", "<svg", "<iframe", "<field>", "<b>"):
            self.assertNotIn(tag, rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_private_access_failure_is_visible_and_escaped_in_preview(self):
        rendered = hub.preview_html({"repo": "alice/private-data", "rows": [],
                                     "error": "Could not access <private> dataset"})
        self.assertIn("Could not access &lt;private&gt; dataset", rendered)
        self.assertNotIn("<private>", rendered)

    def test_train_test_overlap_is_rejected_and_assignments_are_deterministic(self):
        rows = self.rows[:6]
        before = json.dumps(rows)
        with self.assertRaisesRegex(ValueError, "distinct"):
            hub.assign_splits(rows, [{"prompt": " QUESTION   0 ", "target": "different"}])
        assigned = hub.assign_splits(rows)
        reversed_assignments = hub.assign_splits(list(reversed(rows)))
        self.assertEqual({row["prompt"]: row["split"] for row in assigned},
                         {row["prompt"]: row["split"] for row in reversed_assignments})
        self.assertEqual(json.dumps(rows), before)
        self.assertEqual(sum(row["split"] == "validation" for row in assigned), 1)

    def test_public_search_uses_visitor_token_and_omits_private_results(self):
        self.api.list_datasets.return_value = [
            SimpleNamespace(id="public/one", private=False, gated=False, downloads=120),
            SimpleNamespace(id="visitor/private", private=True),
            SimpleNamespace(id="public/gated", private=False, gated="auto", downloads=10),
        ]
        with patch.dict(os.environ, {"HF_TOKEN": "owner-secret"}):
            results = hub.search_dataset_results(" public ", token=False)
        self.client.assert_called_once_with(token=False)
        self.assertEqual([item["id"] for item in results], ["public/one", "public/gated"])
        self.assertTrue(results[1]["gated"])
        self.assertNotIn("owner-secret", json.dumps(results))
        self.api.list_datasets.assert_called_once_with(search="public", limit=12, sort="downloads")

    def test_search_is_bounded_and_errors_do_not_echo_tokens(self):
        self.assertEqual(hub.search_datasets("a"), [])
        self.client.assert_not_called()
        self.api.list_datasets.side_effect = RuntimeError("request failed hf_visitor")
        with self.assertRaisesRegex(ValueError, "Could not search") as caught:
            hub.search_datasets("data", token="hf_visitor")
        self.assertNotIn("hf_visitor", str(caught.exception))
        self.client.assert_called_once_with(token="hf_visitor")
        for limit in (0, 26, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                hub.search_datasets("data", limit=limit)

    def test_schema_choices_support_shared_nested_columns_and_arrays(self):
        result = {"rows": [{"examples": [{"input": "Question", "target": "Answer"}],
                            "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}],
                  "target_field": "examples[].target ?? examples[].target_scores"}
        choices = hub.field_choices(result)
        paths = {value for _, value in choices}
        self.assertTrue({"examples[].input", "examples[].target", "messages[0].content", "messages[1].content"}.issubset(paths))
        self.assertIn(result["target_field"], paths)
        self.assertIn("examples[].input · Question", [label for label, _ in choices])
        for path in paths - {result["target_field"]}:
            hub.dataset.field_value(result["rows"][0], path)

    def test_mapped_preview_precedes_collapsed_raw_record_and_has_role(self):
        rendered = hub.preview_html({"repo": "alice/data", "rows": [{"question": "q", "answer": "a"}],
                                     "mapped_rows": [{"prompt": "q", "target": "a"}]}, role="Held-out test")
        self.assertIn('data-preview-role="Held-out test"', rendered)
        self.assertLess(rendered.index("<h4>Prompt"), rendered.index("Raw record"))
        self.assertIn('<details class="vs-raw-record" ><summary>Raw record', rendered)
        corpus = hub.preview_html({"dataset_mode": "corpus", "rows": [{"text": "prose"}],
                                   "mapped_rows": [{"prompt": "", "target": "prose"}]})
        self.assertIn("<h4>Text</h4>", corpus)
        self.assertNotIn("<h4>Reference</h4>", corpus)

    def test_upload_preview_retains_raw_nested_schema_and_counts(self):
        path = self.root / "nested.jsonl"
        rows = [{"examples": [{"input": f"q{i}", "target": f"a{i}"}]} for i in range(6)]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        config = {"example_limit": 6, "dataset_mode": "paired"}
        result = hub.preview_upload(str(path), config)
        self.assertIsNone(result["mapping_error"])
        self.assertEqual(result["rows"], rows[:3])
        self.assertEqual(result["prompt_field"], "examples[].input")
        self.assertEqual(result["target_field"], "examples[].target")
        self.assertEqual(result["counts"], {"edit": 5, "validation": 1, "test": 0, "preserve": 0})
        self.assertEqual(result["total_rows"], 6)
        self.assertEqual(result["mapped_rows"][0]["prompt"], hub.read_upload(str(path), config)[0]["prompt"])

    def test_upload_mapping_error_keeps_raw_columns_available(self):
        path = self.root / "text.txt"
        path.write_text("\n".join(f"Example {i}" for i in range(6)))
        result = hub.preview_upload(str(path), {"example_limit": 6, "dataset_mode": "paired"})
        self.assertIsNotNone(result["mapping_error"])
        self.assertEqual(result["rows"][0], {"text": "Example 0"})
        self.assertIn("text", {value for _, value in hub.field_choices(result)})
        corpus = hub.preview_upload(str(path), {"example_limit": 6, "dataset_mode": "corpus"})
        self.assertIsNone(corpus["mapping_error"])
        self.assertEqual(corpus["text_field"], "text")


if __name__ == "__main__":
    unittest.main()
