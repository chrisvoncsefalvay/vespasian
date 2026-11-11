"""Offline contract checks for metadata guidance and text-task boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import task_metadata as tasks
import dataset


class TaskMetadataTests(unittest.TestCase):
    def test_scalar_list_and_tag_metadata_normalize(self):
        for info in ({"cardData":{"task_categories":"text-generation"}},
                     SimpleNamespace(card_data={"task_categories":["text-generation"]}),
                     {"tags":["task_categories:text-generation"]}):
            result=tasks.assess_task(info)
            self.assertEqual(result["selected_task"],"text-generation")
            self.assertFalse(result["requires_choice"])
            self.assertEqual(result["selection_source"],"dataset_metadata")

    def test_task_ids_are_canonical_and_unknown_tags_are_preserved(self):
        result=tasks.assess_task({"cardData":{"task_ids":["sentiment-classification","new-unknown-task"]}})
        self.assertEqual(result["suggested_tasks"],["text-classification"])
        self.assertEqual(result["unknown_tags"],["new-unknown-task"])
        self.assertTrue(result["requires_choice"])

    def test_missing_ambiguous_and_unknown_need_user_choice(self):
        for info in ({}, {"cardData":{"task_categories":["translation","summarization"]}},
                     {"tags":["task_categories:unknown"]}):
            with self.subTest(info=info):
                with self.assertRaisesRegex(ValueError,"Choose a task explicitly"):
                    tasks.require_task(tasks.assess_task(info))
                result=tasks.assess_task(info,"text-generation")
                self.assertEqual(tasks.require_task(result),"text-generation")
                self.assertEqual(result["selection_source"],"user")

    def test_multimodal_model_tag_does_not_reject_text_adapter(self):
        result=tasks.assess_task({"cardData":{"task_categories":"text-generation"}},
                                model_info={"pipeline_tag":"image-text-to-text"})
        self.assertEqual(tasks.require_task(result),"text-generation")
        self.assertEqual(result["model_pipeline_tag"],"image-text-to-text")

    def test_explicit_task_mismatch_is_disclosed(self):
        result=tasks.assess_task({"cardData":{"task_categories":"text-classification"}},"text-generation")
        self.assertIn("differs",result["warning"])

    def test_unsupported_task_and_nontext_metadata_fail_clearly(self):
        with self.assertRaisesRegex(ValueError,"Unsupported editing task"):
            tasks.assess_task(selected_task="image-classification")
        with self.assertRaisesRegex(ValueError,"non-text"):
            tasks.require_task(tasks.assess_task({"tags":["task_categories:image-classification"]}))

    def test_classlabel_names_replace_numeric_ids_without_mutating_source(self):
        row={"text":"review","label":1}
        mapped=tasks.prepare_row(row,("text","label"),"paired","text-classification",{"label":SimpleNamespace(names=["negative","positive"])})
        self.assertEqual(mapped["label"],"positive")
        self.assertEqual(row["label"],1)
        for value in (2,-1,1.5):
            with self.assertRaises(ValueError):
                tasks.prepare_row({"label":value},("text","label"),"paired","text-classification",{"label":{"names":["a","b"]}})
        with self.assertRaisesRegex(ValueError,"ClassLabel"):
            tasks.prepare_row(row,("text","label"),"paired","text-classification")

    def test_nested_numeric_classification_labels_are_not_guessed(self):
        with self.assertRaisesRegex(dataset.DatasetError,"Nested numeric"):
            dataset.prepare_task_row({"data":{"text":"review","label":1}},("data.text","data.label"),"paired","text-classification")
        row={"data":{"text":"review","label":"positive"}}
        self.assertEqual(dataset.prepare_task_row(row,("data.text","data.label"),"paired","text-classification"),row)

    def test_separate_qa_context_must_not_be_silently_discarded(self):
        row={"question":"Who?","context":"Relevant facts","answer":"Ada","prompt":"Relevant facts. Who?"}
        with self.assertRaisesRegex(ValueError,"separate context"):
            tasks.prepare_row(row,("question","answer"),"paired","question-answering")
        self.assertEqual(tasks.prepare_row(row,("prompt","answer"),"paired","question-answering"),row)

    def test_paired_classification_inputs_are_not_silently_dropped(self):
        row={"premise":"A cat sits.","hypothesis":"An animal sits.","label":"entailment"}
        with self.assertRaisesRegex(ValueError,"paired input"):
            tasks.prepare_row(row,("premise","label"),"paired","text-classification")

    def test_local_cli_requires_choice_without_network(self):
        script=str(Path(tasks.__file__))
        failed=subprocess.run([sys.executable,script],capture_output=True,text=True)
        self.assertEqual(failed.returncode,1)
        self.assertIn("--task",failed.stderr)
        passed=subprocess.run([sys.executable,script,"--task","summarization","--dataset-mode","paired"],capture_output=True,text=True)
        self.assertEqual(passed.returncode,0)
        self.assertEqual(json.loads(passed.stdout)["selected_task"],"summarization")

    def test_fetch_missing_tags_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); api=Mock();api.dataset_info.return_value=SimpleNamespace(sha="a"*40)
            loader=Mock(return_value=[{"prompt":"q","target":"a"}])
            kwargs=dict(repo="org/data",split="train",output=root/"data.jsonl",receipt=root/"receipt.json",api=api,loader=loader)
            with self.assertRaisesRegex(dataset.DatasetError,"Choose a task explicitly"):
                dataset.fetch(**kwargs)
            self.assertFalse((root/"data.jsonl").exists())
            loader.assert_not_called()
            result=dataset.fetch(**kwargs,hf_task="text-generation")
            self.assertEqual(result["hf_task"],"text-generation")

    def test_preview_and_fetch_share_classlabel_mapping(self):
        class Stream(list):
            features={"label":SimpleNamespace(names=["negative","positive"])}
        api=Mock();api.dataset_info.return_value=SimpleNamespace(sha="a"*40,card_data={"task_categories":"text-classification"})
        api.model_info.return_value=SimpleNamespace(pipeline_tag="image-text-to-text")
        loader=lambda *args,**kwargs: Stream([{"text":"review","label":1}])
        kwargs=dict(repo="org/data",api=api,loader=loader,model_id="Qwen/example",prompt_field="text",target_field="label",config_names=lambda *a,**k:["default"])
        preview=dataset.preview(**kwargs,split_names=lambda *a,**k:["train"])
        self.assertIsNone(preview["mapping_error"])
        api.model_info.assert_called_once_with(repo_id="Qwen/example")
        self.assertEqual(preview["task_metadata"]["model_suggested_tasks"],["text-generation"])
        self.assertEqual(preview["task_metadata"]["model_pipeline_tag"],"image-text-to-text")
        self.assertEqual(preview["mapped_rows"][0]["target"],"positive")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            receipt=dataset.fetch(**kwargs,split="train",output=root/"data.jsonl",receipt=root/"receipt.json")
            self.assertEqual(receipt["hf_task"],preview["task_metadata"]["selected_task"])
            self.assertEqual(json.loads((root/"data.jsonl").read_text())["target"],"positive")


if __name__=="__main__": unittest.main()
