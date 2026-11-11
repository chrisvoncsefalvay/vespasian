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


if __name__ == "__main__":
    unittest.main()
