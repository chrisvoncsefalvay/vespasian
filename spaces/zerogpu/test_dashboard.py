"""Measured-turn inspection and result presentation contracts."""

from __future__ import annotations

from html.parser import HTMLParser
import json
import unittest

import dashboard


def metrics(score, loss=4.0, count=3):
    return {"validation": {"exact_match": score, "loss": loss, "count": count}}


def answer(ident, turn, text, score=False, split="validation", label=None):
    return {"type": "answer", "turn": turn, "label": label or ("baseline" if turn == 0 else f"turn-{turn:03d}"),
            "example": {"id": ident, "prompt": f"Question {ident}", "target": "Reference answer",
                        "split": split, "output": text, "exact_match": score, "loss": 4.0}}


def state():
    baseline_health = {"copying": {"label": "Copying", "count": 4, "score": 1.0}}
    events = [
        {"type": "start", "model": "Qwen/Qwen3.5-0.8B", "goal": "Answer greetings", "budget_usd": 1.0, "max_turns": 2},
        {"type": "assessment", "metric": "exact_match", "suitable": True, "reason": "References are available", "plan": "Measure validation agreement"},
        {"type": "model_structure", "num_layers": 4},
        {"type": "baseline", "metrics": metrics(0, 4.9650), "health": baseline_health},
        answer("v1", 0, "Hello"),
        {"type": "proposal", "turn": 1, "parent_turn": 0,
         "proposal": {"operation": "scale", "strength": .1, "targets": ["model.layers.1.mlp.weight"], "rationale": "Increase reference likelihood"}},
        answer("v1", 1, "Hello there"),
        {"type": "turn", "turn": 1, "parent_turn": 0, "accepted": True,
         "metrics": metrics(0, 4.8623), "health": baseline_health, "health_passed": True,
         "mutation": {"operation": "scale", "strength": .1, "elapsed_seconds": .057,
                      "changed_tensors": [{"name": "model.layers.1.mlp.weight", "shape": [4,4], "relative_norm": .1}]},
         "decision_reason": "Validation score tied with lower target loss"},
        {"type": "proposal", "turn": 2, "parent_turn": 1,
         "proposal": {"operation": "noise", "strength": .02, "targets": ["model.layers.2.mlp.weight"], "rationale": "Test a second layer"}},
        answer("v1", 2, "Changed answer"),
        {"type": "turn", "turn": 2, "parent_turn": 1, "accepted": False,
         "metrics": metrics(0, 5.0), "health": {"copying": {"count": 4, "score": .75}},
         "health_passed": False, "decision_reason": "Reference loss increased"},
    ]
    return {"events": events, "config": {"metric": "exact_match", "health_max_drop": None},
            "status": "running", "elapsed": 30}


class StrategySummaryTests(unittest.TestCase):
    def test_explicit_strategy_summaries_are_visible_and_escaped(self):
        current = state()
        next(event for event in current["events"] if event["type"] == "proposal")["proposal"].update(strategy="local-sweep", plan="Apply <small> change", reflection="Previous loss decreased")
        rendered = dashboard.render_dashboard(current, {"turn": 1, "follow": False})
        for expected in ("Strategy:", "local-sweep", "Plan:", "Apply &lt;small&gt; change", "Reflection:", "Previous loss decreased"):
            self.assertIn(expected, rendered)
        self.assertNotIn("Apply <small>", rendered)


class Elements(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.nodes = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))


class DashboardTests(unittest.TestCase):
    def test_empty_does_not_invent_measurements(self):
        rendered = dashboard.render_dashboard()
        self.assertIn("Waiting for the baseline", rendered)
        self.assertNotIn("vs-point-retained", rendered)
        self.assertNotIn("0.0%", rendered)
        self.assertNotIn("vs-result-summary", rendered)

    def test_retained_tie_explains_loss_next_to_decision(self):
        rendered = dashboard.render_dashboard(state(), {"turn": 1, "follow": False})
        inspector = rendered.split('class="vs-turn-inspector"')[1].split('</section>')[0]
        self.assertIn("Validation score tied with lower target loss", inspector)
        self.assertIn("4.9650", inspector)
        self.assertIn("4.8623", inspector)
        self.assertIn("-0.1027", inspector)
        self.assertIn("Tensor update 0.057s", inspector)

    def test_selection_highlights_only_selected_layers(self):
        rendered = dashboard.render_dashboard(state(), {"turn": 2, "follow": False})
        nodes = Elements(rendered).nodes
        selected_layers = [attrs["aria-label"] for _,attrs in nodes if attrs.get("class") == "vs-layer vs-layer-selected"]
        self.assertEqual(selected_layers, ["Layer 2: selected"])
        self.assertIn("Parent: Turn 1", rendered)
        self.assertIn("Reference loss increased", rendered)

    def test_graph_and_lineage_are_keyboard_selectable(self):
        nodes = Elements(dashboard.render_dashboard(state())).nodes
        points = [attrs for tag,attrs in nodes if tag == "g" and "data-vs-turn" in attrs]
        self.assertEqual(len(points), 9)
        self.assertTrue(all(attrs.get("tabindex") == "0" and attrs.get("role") == "button" for attrs in points))
        buttons = [attrs for tag,attrs in nodes if tag == "button" and attrs.get("data-vs-turn") == "2"]
        self.assertTrue(buttons)
        self.assertTrue(all(attrs.get("aria-pressed") == "true" for attrs in buttons))

    def test_follow_and_pin_choose_different_turns(self):
        view = dashboard._replay(state())
        self.assertEqual(dashboard.normalise_selection(view, {"turn":1,"follow":True})["turn"],2)
        self.assertEqual(dashboard.normalise_selection(view, {"turn":1,"follow":False})["turn"],1)
        self.assertEqual(dashboard.normalise_selection(view, {"turn":999,"follow":False})["turn"],2)

    def test_all_recorded_samples_can_be_paginated(self):
        current = state()
        current["events"] = [event for event in current["events"] if event["type"] != "answer"]
        for i in range(23):
            current["events"].extend([answer(str(i),0,f"Before {i}"),answer(str(i),1,f"After {i}")])
        selection = {"turn":1,"follow":False,"page":0,"split":"all","outcome":"all"}
        rows = dashboard.comparison_rows(dashboard._replay(current),selection)
        self.assertEqual(len(rows),23)
        seen = set()
        for page in range(4):
            rendered = dashboard.render_dashboard(current,{**selection,"page":page})
            samples = [attrs["data-vs-sample"] for _,attrs in Elements(rendered).nodes if "data-vs-sample" in attrs]
            self.assertLessEqual(len(samples),6)
            self.assertFalse(seen.intersection(samples))
            seen.update(samples)
        self.assertEqual(len(seen),23)
        self.assertIn("Page 4 of 4 · 23 examples",rendered)
        self.assertIn("Before 22",rendered)
        self.assertIn("After 22",rendered)

    def test_reloaded_output_pairs_with_original_baseline(self):
        current = state()
        current["events"].append(answer("v1",1,"Reloaded text",label="reloaded"))
        rendered = dashboard.render_dashboard(current,{"turn":1,"follow":False})
        self.assertIn("Reloaded text",rendered)
        self.assertIn("Reloaded checkpoint",rendered)
        self.assertIn('class="vs-generated-text">Hello</p>',rendered)
        self.assertNotIn('class="vs-generated-text">Hello there</p>',rendered)

    def test_health_examples_are_separate_by_default(self):
        current = state()
        for turn in (0,1):
            event = answer("copying:0000",turn,"COPY",label="Baseline health" if turn == 0 else "Health · turn 1")
            event["example"]["split"] = None
            event["example"]["passed"] = True
            current["events"].append(event)
        view = dashboard._replay(current)
        selected = dashboard.normalise_selection(view,{"turn":1,"follow":False})
        self.assertEqual(len(dashboard.comparison_rows(view,selected)),1)
        health = dashboard.comparison_rows(view,{**selected,"split":"health"})
        self.assertEqual([item["id"] for item in health],["copying:0000"])

    def test_outcome_filters_use_selected_metric_and_require_two_scores(self):
        current = state()
        current["events"].extend([answer("improved",0,"old",False),answer("improved",1,"new",True),
                                  answer("regressed",0,"correct",True),answer("regressed",1,"wrong",False),
                                  answer("test-only",1,"held out",True,split="test")])
        view = dashboard._replay(current)
        selected = dashboard.normalise_selection(view,{"turn":1,"follow":False})
        self.assertEqual([row["id"] for row in dashboard.comparison_rows(view,{**selected,"outcome":"improved"})],["improved"])
        self.assertEqual([row["id"] for row in dashboard.comparison_rows(view,{**selected,"outcome":"regressed"})],["regressed"])
        self.assertEqual([row["id"] for row in dashboard.comparison_rows(view,{**selected,"split":"test"})],["test-only"])

    def test_zero_limit_is_enforced_and_tracking_has_no_threshold(self):
        current = state()
        tracking = dashboard.render_dashboard(current)
        self.assertIn("Tracking scores",tracking)
        self.assertNotIn('class="vs-threshold-line"',tracking)
        current["config"]["health_max_drop"] = 0
        enforced = dashboard.render_dashboard(current)
        self.assertIn("Rejecting drops above 0 percentage points",enforced)
        self.assertIn('class="vs-threshold-line"',enforced)
        self.assertIn("3/4 correct",enforced)
        self.assertIn("A preservation limit was exceeded",enforced)

    def test_completed_summary_uses_final_test_and_stop_reason(self):
        current = state()
        current["status"] = "complete"
        current["report"] = {"metric":"exact_match", "best_turn":1,"stop_reason":"turn_limit","reload_verified":True,
                             "baseline_metrics":metrics(0,4.9650), "best_metrics":metrics(0,4.8623),
                             "final_metrics":{**metrics(0,4.8623),"test":{"count":2,"exact_match":.5,"loss":5.1144}},
                             "final_health":{"copying":{"label":"Copying","count":4,"score":1.0}},
                             "checkpoint":"/private/session/run/checkpoint","budget":{"spent_usd":.12}}
        rendered = dashboard.render_dashboard(current)
        self.assertIn("Held-out test",rendered)
        self.assertIn("50.0%",rendered)
        self.assertIn("5.1144",rendered)
        self.assertIn("Maximum edit turns reached",rendered)
        self.assertIn("Checkpoint reload verified",rendered)
        self.assertIn("4/4 correct",rendered)
        self.assertIn('href="#vs-results"',rendered)
        self.assertNotIn("/private/session",rendered)

    def test_untrusted_text_is_escaped_in_markup_and_attributes(self):
        current = state()
        current["events"][0]["goal"] = '<script>alert("bad")</script>'
        current["events"].append(answer('" onmouseover="alert(1)',2,'<img src=x onerror=alert(1)>'))
        current["events"][-2]["decision_reason"] = '<svg onload=alert(2)>'
        rendered = dashboard.render_dashboard(current,{"split":'<img>',"outcome":'<script>',"follow":False,"turn":2})
        self.assertNotIn("<script>",rendered)
        self.assertNotIn("<img src",rendered)
        self.assertNotIn("<svg onload",rendered)
        nodes = Elements(rendered).nodes
        self.assertFalse(any("onmouseover" in attrs or "onerror" in attrs for _,attrs in nodes))
        data = next(attrs["data-vs-selection"] for _,attrs in nodes if "data-vs-selection" in attrs)
        self.assertEqual(json.loads(data)["split"],"all")

    def test_selected_baseline_never_highlights_retained_layers(self):
        rendered = dashboard.render_dashboard(state(),{"turn":0,"follow":False})
        self.assertNotIn('class="vs-layer vs-layer-selected"',rendered)
        self.assertIn('data-vs-selected-turn="0"',rendered)

    def test_health_chart_uses_reserved_axis_space(self):
        rendered = dashboard.render_dashboard(state())
        self.assertIn('viewBox="0 0 440 180"',rendered)
        self.assertIn('x="58"',rendered)



class PatchCheckpointTests(unittest.TestCase):
    def test_only_observed_safe_patch_status_is_rendered(self):
        self.assertNotIn("Patch recovery status", dashboard.render_dashboard({"events": []}))
        saved = dashboard.render_dashboard({"events": [{"type":"patch_checkpoint","turn":2,"status":"saved","bundle_bytes":4096,"raw":"PRIVATE"}]})
        self.assertIn("4,096 bytes", saved)
        self.assertNotIn("PRIVATE", saved)
        failed = dashboard.render_dashboard({"events": [{"type":"patch_checkpoint","turn":2,"status":"backup_failed","local_saved":True,"error":"SECRET"}]})
        self.assertIn("local patch checkpoint is retained", failed)
        self.assertNotIn("SECRET", failed)
        backed = dashboard.render_dashboard({"events": [{"type":"patch_checkpoint","turn":2,"status":"backed_up","repo":"<script>unsafe</script>"}]})
        self.assertIn("backed up privately", backed)
        self.assertNotIn("<script>", backed)

if __name__ == "__main__":
    unittest.main()

class MeasuredLoopTests(unittest.TestCase):
    @staticmethod
    def stage(stage, status='completed', turn=1, **details):
        return {'type':'loop_stage','stage':stage,'status':status,'turn':turn,'summary':stage+' receipt','details':details}

    def render(self, events, **extra):
        return dashboard.render_dashboard({'events':events, **extra})

    def test_later_stage_does_not_fabricate_completion(self):
        html=self.render([self.stage('test','started')])
        self.assertIn('data-vs-stage="patch" data-vs-stage-status="pending"',html)
        self.assertIn('data-vs-stage="test" data-vs-stage-status="started"',html)

    def test_rejected_plan_is_not_reverted_patch(self):
        events=[self.stage('plan',proposal={'operation':'token_row_scale','token_ids':[7]}),self.stage('decide',accepted=False,decision_reason='Unsupported target')]
        view=dashboard._replay({'events':events})
        self.assertEqual(dashboard._decision(view,1,view['changes'][1]),'Rejected')
        events.insert(1,self.stage('patch',mutation={'changed_tensors':[]}))
        view=dashboard._replay({'events':events})
        self.assertEqual(dashboard._decision(view,1,view['changes'][1]),'Reverted')

    def test_cancelled_stage_not_claimed_completed(self):
        html=self.render([self.stage('patch','started')],status='cancelled')
        self.assertIn('Stopped before completion',html)
        self.assertIn('data-vs-stage="patch" data-vs-stage-status="started"',html)

    def test_measured_rows_and_shared_weights_are_escaped(self):
        mutation={'changed_tensors':[{'name':'model.embed_tokens.weight','tied':True,'roles':['input','output'],'rows':[{'token_id':7,'piece':'<unsafe>','text':'hello','source_token_id':8,'effective_fraction':.01,'relative_norm':.01}]}]}
        events=[self.stage('plan',proposal={'operation':'token_row_blend','token_ids':[7],'source_token_ids':[8]}), self.stage('patch',mutation=mutation)]
        html=self.render(events)
        self.assertIn('Measured token-row patch',html)
        self.assertIn('One physical change affects both roles',html)
        self.assertIn('&lt;unsafe&gt;',html)
        self.assertNotIn('<unsafe>',html)
        self.assertNotIn('No stored patch',html)

    def test_private_unselected_evidence_is_not_rendered(self):
        html=self.render([self.stage('plan',reasoning='PRIVATE_CHAIN',proposal={'operation':'scale','rationale':'Public rationale'})])
        self.assertNotIn('PRIVATE_CHAIN',html)
        self.assertIn('Public rationale',html)

    def test_pinned_turn_does_not_show_latest_stage(self):
        events=[self.stage('plan',turn=1,proposal={'operation':'scale'}),self.stage('plan',turn=2,proposal={'operation':'noise'}),self.stage('patch','started',turn=2)]
        view=dashboard._replay({'events':events})
        html=dashboard._edit_loop(view,{'turn':1,'follow':False})
        self.assertIn('data-vs-stage="patch" data-vs-stage-status="pending"',html)

    def test_observed_token_catalogue_and_budget_limits(self):
        inspection={'token_tables':[{'name':'embed.weight','roles':['input','output'],'tied':True}], 'token_candidates':[{'id':7,'piece':'<hello>','text':'hello'}]}
        html=self.render([{'type':'start','budget_mode':'reserved_api'},self.stage('inspect',turn=0,inspection=inspection)])
        self.assertIn('Observed token tables and tokenisation',html)
        self.assertIn('&lt;hello&gt;',html)
        self.assertIn('configured token rates',html)
        html=self.render([{'type':'start','budget_mode':'estimated_between_calls'}])
        self.assertIn('not a hard dollar cap',html)


class AgentToolActivityTests(unittest.TestCase):
    @staticmethod
    def event(tool="read_skill", status="completed", turn=1, call_id="1", summary="Loaded a bounded row-edit skill", **extra):
        return {"type": "agent_tool", "tool": tool, "status": status, "turn": turn,
                "call_id": call_id, "summary": summary, **extra}

    def test_public_receipts_only_and_no_inferred_patch(self):
        event = self.event(summary="Loaded <bounded> skill", details={"request": "PRIVATE_REQUEST", "result": "PRIVATE_RESULT"}, reasoning="PRIVATE_REASONING")
        rendered = dashboard.render_dashboard({"events": [event]})
        self.assertIn("Skills and inspection tools", rendered)
        self.assertIn("Loaded &lt;bounded&gt; skill", rendered)
        self.assertIn('data-vs-tool="read_skill" data-vs-tool-status="completed"', rendered)
        self.assertIn('data-vs-stage="patch" data-vs-stage-status="pending"', rendered)
        for private in ("PRIVATE_REQUEST", "PRIVATE_RESULT", "PRIVATE_REASONING"):
            self.assertNotIn(private, rendered)

    def test_failed_tool_is_not_a_successful_plan(self):
        rendered = dashboard.render_dashboard({"events": [self.event(status="failed", summary="Skill unavailable on this model")]})
        self.assertIn('data-vs-tool-status="failed"', rendered)
        self.assertIn("Skill unavailable on this model", rendered)
        self.assertIn('data-vs-stage="plan" data-vs-stage-status="pending"', rendered)

    def test_pin_preserves_per_turn_history(self):
        view = dashboard._replay({"events": [self.event(summary="First turn skill"), self.event(turn=2, summary="Second turn skill")]})
        rendered = dashboard._edit_loop(view, {"turn":1, "follow":False})
        self.assertIn("First turn skill", rendered)
        self.assertNotIn("Second turn skill", rendered)
        followed = dashboard._edit_loop(view, {"turn":1, "follow":True})
        self.assertIn("Second turn skill", followed)
        self.assertNotIn("First turn skill", followed)

    def test_same_call_updates_without_duplicates_and_interruption_is_honest(self):
        events = [self.event(status="started"), self.event(status="completed")]
        view = dashboard._replay({"events": events})
        self.assertEqual(len(view["agent_tools"]), 1)
        self.assertIn('data-vs-tool-status="completed"', dashboard.render_dashboard({"events":events}))
        rendered = dashboard.render_dashboard({"events": [self.event(status="started")], "status":"cancelled"})
        self.assertIn("Stopped before completion", rendered)
        self.assertNotIn('data-vs-tool-status="completed"', rendered)

    def test_activity_is_bounded_and_unknown_status_ignored(self):
        events = [self.event(call_id=str(i), summary=f"Receipt {i}") for i in range(20)]
        events += [self.event(status="imagined", summary="NOT_A_RECEIPT")]
        rendered = dashboard.render_dashboard({"events": events})
        self.assertEqual(rendered.count('data-vs-tool="read_skill"'), 12)
        self.assertIn("latest 12 of 20", rendered)
        self.assertNotIn("NOT_A_RECEIPT", rendered)
