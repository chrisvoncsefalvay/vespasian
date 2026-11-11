//! Capability-filtered search guidance, not claims of transferred benchmark gains.
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn row_tables(inspection: &Value, operation: &str, minimum_candidates: usize) -> Vec<Value> {
    let candidates: BTreeSet<u64> = inspection["token_candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|candidate| candidate["id"].as_u64())
        .collect();
    inspection["token_tables"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|table| {
            let name = table["name"].as_str().filter(|name| !name.is_empty())?;
            let vocabulary = table["shape"][0].as_u64()?;
            if !table["operations"]
                .as_array()?
                .iter()
                .any(|value| value.as_str() == Some(operation))
            {
                return None;
            }
            let ids: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|id| *id < vocabulary)
                .collect();
            (ids.len() >= minimum_candidates).then(|| {
                json!({"name":name,"aliases":table["aliases"],"roles":table["roles"],
                    "tied":table["tied"],"candidate_ids":ids})
            })
        })
        .collect()
}

/// Both frontends reach this catalogue through the same Rust controller.
/// The worker remains authoritative about available operations, tensors and
/// grounded token IDs. Benchmark token IDs are never candidate selectors.
pub fn catalogue(dataset_mode: &str, allowed_operations: &Value, inspection: &Value) -> Value {
    let scale_tables = row_tables(inspection, "token_row_scale", 1);
    let interpolate_tables = row_tables(inspection, "token_row_interpolate", 2);
    let allowed: Vec<&str> = allowed_operations
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|op| {
            matches!(
                *op,
                "scale"
                    | "low_rank"
                    | "noise"
                    | "activation_rank_one"
                    | "activation_preserve"
                    | "token_row_residual"
                    | "token_row_scale"
                    | "token_row_interpolate"
            )
        })
        .filter(|op| {
            dataset_mode != "prompts"
                || !matches!(
                    *op,
                    "activation_rank_one" | "activation_preserve" | "token_row_residual"
                )
        })
        .filter(|op| *op != "token_row_scale" || !scale_tables.is_empty())
        .filter(|op| *op != "token_row_interpolate" || !interpolate_tables.is_empty())
        .filter(|op| {
            *op != "token_row_residual"
                || (!row_tables(inspection, "token_row_residual", 1).is_empty()
                    && inspection["token_candidates"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|r| r["edit_reference"] == true))
        })
        .collect();
    let mut cards = Vec::new();
    if allowed.contains(&"activation_preserve") {
        cards.push(json!({"id":"preservation_aware","operations":["activation_preserve"],
            "when_to_use":"Reference errors coexist with preservation examples, especially after collateral regression.",
            "plan":"Inspect a small MLP down-projection target set. Use positive strength up to 0.1 and rank 1–16. The worker measures edit/reference keys and target-minus-predicted embedding directions, then projects edit keys away from a bounded measured preservation-key span before constructing the patch. Start small, test validation and preservation, and revert regressions. Return empty token selectors.",
            "evidence":"A forward-measurement contrastive heuristic with local preservation-key projection. This implementation is not ROME or MEMIT.",
            "limitation":"Requires supported reference-bearing edit and preservation examples. Projection protects only sampled local directions, not whole-model behaviour. A degenerate projected direction rejects safely. No autograd, backpropagation or optimised latent target is used."}));
    }
    let residual_ids: Vec<_> = inspection["token_candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| r["edit_reference"] == true)
        .filter_map(|r| r["id"].as_u64())
        .collect();
    let residual_tables = row_tables(inspection, "token_row_residual", 1);
    if allowed.contains(&"token_row_residual")
        && !residual_ids.is_empty()
        && !residual_tables.is_empty()
    {
        cards.push(json!({"id":"residual_output_row","operations":["token_row_residual"],
            "when_to_use":"A non-special edit-reference token is underpredicted and an ordinary linear output head supports measured residual writes.",
            "plan":"Inspect the output table and token candidates marked edit_reference. Select one canonical output table and one to 32 such token IDs, no source IDs, positive strength up to 0.1 and rank 1–16 edit examples. The worker measures actual output-head input h at reference positions, accumulates (1[target=id]-p_id)*h/||h||², then normalises and bounds the stored row delta. Inspect actual changes and compare validation plus preservation before keeping.",
            "evidence":"Executable forward-pass measurement and residual algebra, not a target-embedding copy or exact ROME/MEMIT method.",
            "limitation":"Only supported linear output heads and reference-bearing data qualify. IDs must occur in the sampled edit references (first 64 target positions per selected example) or the edit is rejected. Tied heads also alter input embeddings. Local residual direction and small norm do not guarantee improved generation. No autograd/backpropagation or training loop is run."}));
    }

    if allowed.contains(&"token_row_scale") {
        cards.push(json!({
            "id":"token_row_scaling",
            "operations":["token_row_scale"],
            "eligible_tables":scale_tables,
            "when_to_use":"Observed answers show a token-specific error and inspection identifies the relevant non-special token IDs and an eligible embedding or output table. Token pieces are context-dependent; one word may span several rows.",
            "plan":"Select exactly one canonical table and one to 32 distinct grounded token_ids. Explain which observed failure each selected piece addresses. Scale only those rows by 1+strength, with nonzero absolute strength at most 0.1 and source_token_ids empty. Start small and test both target behaviour and preservation. A tied table changes input and output behaviour together; its aliases are one physical edit, not separate targets.",
            "evidence":"IWIBench lexical-inhibition trace 49fb438b57a34eb2bff43441ad0e4f22 scaled 17 selected lm_head rows by factors 0.4 or 0.6. Final target success was 1/2 to 2/2, retaining 66/66 controls.",
            "limitation":"This is real selected-row scaling, but Vespasian bounds each proposal to factors 0.9 through 1.1 rather than reproducing the benchmark magnitudes. Scaling is not guaranteed to suppress a token because logits depend on signed dot products and tied weights can change representations. Use current tokenizer inspection, never benchmark row IDs. No special-token or arbitrary-vocabulary edits are allowed."
        }));
    }
    if allowed.contains(&"token_row_interpolate") {
        cards.push(json!({
            "id":"token_row_transfer",
            "operations":["token_row_interpolate"],
            "eligible_tables":interpolate_tables,
            "when_to_use":"A concrete alias or token-association hypothesis is supported by the actual current tokenisation and the inspected table has both source and destination rows. Similar spelling alone is not evidence of a useful source row.",
            "plan":"Select exactly one canonical table and pair each grounded destination token_id with a grounded source_token_id in that table. Use a small positive strength up to 0.1 for the blend toward source. The stored delta is also norm-bounded per destination row, so effective interpolation may be smaller. Sources are read before the candidate edit. Explain tied input/output consequences, test the intended association, and test copying, original source meaning and unrelated answers before retaining.",
            "evidence":"IWIBench alias-acquisition trace 62d170a5993548e583b5a2e687f7a259 first replaced one embedding row with 0.75 times a source row plus 0.25 times its prior destination, then used further arithmetic and scaling between generation probes. Target success rose from 1/3 to 3/3, but explicit copying remained unreliable and only 69/70 controls were retained.",
            "limitation":"Vespasian performs bounded within-table interpolation, not unrestricted copying or the trace's stronger replacement and arithmetic sequence. Subword rows affect many contexts and a tied table changes both embedding and output roles. A successful target test is insufficient if preservation fails. Do not assume matching token IDs across models or tokenisers."
        }));
    }
    if allowed.contains(&"activation_rank_one") {
        cards.push(json!({
            "id":"reference_alignment",
            "operations":["activation_rank_one"],
            "tensor_scope":"Only worker-listed MLP down-projection tensors, not vocabulary tables.",
            "when_to_use":"Contrastive reference editing: reference-bearing examples identify a concrete generated error. For corpus data, references describe style rather than factual answers to reproduce.",
            "plan":"Measure forward activation keys and desired-minus-predicted output embedding directions on reference positions. Use one or a few eligible MLP output tensors and a small positive strength. Compare actual validation answers, reference loss where measured, and preservation outcomes. Change layer or strength only in response to those measurements.",
            "evidence":"IWIBench trace 7ea6c3e0cd5d4087b347bc8989484a3e used observed activations and token-specific writes; final target success was 0/3 to 1/3, with only 61/68 controls retained.",
            "limitation":"Vespasian activation_rank_one is a bounded adaptation, not the trace operation. That trace does not establish efficacy here; it also shows collateral regression. Arbitrary activation handles, autograd and backpropagation are unavailable. This is contrastive forward-only editing, not exact ROME/MEMIT. Token-row operations, when separately listed, have different selectors and bounds."
        }));
    }
    if allowed.contains(&"scale") {
        cards.push(json!({
            "id":"localised_scaling",
            "operations":["scale"],
            "tensor_scope":"Only worker-listed MLP down-projection tensors, not vocabulary tables.",
            "when_to_use":"Measured failures suggest a modest change in the contribution of a listed MLP tensor, and a specific preservation comparison can detect collateral effects.",
            "plan":"Change one listed MLP tensor initially. Use a small relative strength, inspect the generated answers, and reject harmful candidates. Vary strength without simultaneously changing many targets.",
            "evidence":"IWIBench trace 49fb438b57a34eb2bff43441ad0e4f22 selectively scaled output-token rows; final lexical target success was 1/2 to 2/2 with 66/66 controls retained.",
            "limitation":"Whole-MLP-tensor scale is less selective than token_row_scale and does not reproduce that trace. Select the row strategy when a grounded token-specific experiment is justified. The trace's factors must not be copied into relative-strength values; a local small experiment has no guaranteed gain."
        }));
    }
    if !allowed.is_empty() {
        cards.push(json!({
            "id":"measured_refinement",
            "operations":allowed,
            "when_to_use":"A previous measured candidate improved the objective or exposed a preservation failure.",
            "plan":"Name the relevant turn and its measured outcome. Reuse a useful target/direction, changing one of strength, rank, tensor selection or grounded token pair at a time. A seed only fixes random directions; reference-directed activations can change after retained edits. Candidates start from current retained weights, so repeating a strength is another incremental edit, not an absolute reset. If preservation regresses, reduce scope or strength rather than claiming success from target gain alone.",
            "evidence":"IWIBench trace 62d170a5993548e583b5a2e687f7a259 interleaved five writes with four generation probes. Target success rose from 1/3 to 3/3, but explicit copying still failed and only 69/70 controls were retained.",
            "limitation":"The probe-and-adjust procedure transfers; neither its outcome nor its stronger row arithmetic is guaranteed here. Respect each operation's current selectors and bounds. Do not claim an unmeasured candidate will work; do not repeat an identical rejected proposal."
        }));
    }
    let exploratory: Vec<_> = allowed
        .iter()
        .copied()
        .filter(|op| matches!(*op, "low_rank" | "noise"))
        .collect();
    if !exploratory.is_empty() {
        cards.push(json!({
            "id":"bounded_exploration",
            "operations":exploratory,
            "tensor_scope":"Only worker-listed MLP down-projection tensors, not vocabulary tables.",
            "when_to_use":"There is no supported reference-directed candidate or no measured useful direction yet.",
            "plan":"Try one small seeded direction on a small tensor set, state the uncertain hypothesis, and compare validation and preservation measurements before expanding. Prefer low_rank before full-rank noise when both are available. Treat flat or failed results as evidence to change the hypothesis rather than proof that more magnitude will help.",
            "evidence":"This is a Vespasian search fallback, not a successful IWIBench technique established by the reviewed traces. The reviewed Qwen Olympics improvement run finished with zero changed parameters and target success 0/3 to 0/3.",
            "limitation":"Random perturbations have no demonstrated effectiveness from these traces. Stopping without improvement is a valid outcome. Never infer a gain from an attempted write or a favourable explanation."
        }));
    }
    json!({
        "version":3,
        "available_operations":allowed,
        "selection":"Judge the measured current state, inspect model/token evidence, then choose a relevant card. Return its id as strategy, a brief application plan, rationale and reflection citing actual prior outcomes. On the first turn say no previous candidate has been measured. These are public decision summaries, not hidden reasoning. Use only allowed operations, their supported tensors and current grounded row IDs. A plan is not an applied patch, and applying a candidate is not evidence that it improved anything; wait for its test and keep/rollback decision.",
        "evidence_scope":"Small development episodes on Gemma 3 4B and one Qwen 2.5 1.5B run; no claim of transfer to this model, dataset or implementation. Held-out test rows must never enter inspection or strategy selection.",
        "cards":cards
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inspect(ids: Value) -> Value {
        json!({"token_candidates":ids,"token_tables":[{"name":"embedding.weight",
            "shape":[100,32],"aliases":["embedding.weight","lm_head.weight"],
            "roles":["input","output"],"tied":true,
            "operations":["token_row_scale","token_row_interpolate"]}]})
    }

    #[test]
    fn references_enable_only_supported_reference_strategy() {
        let value = catalogue("paired", &json!(["activation_rank_one"]), &Value::Null);
        let cards = value["cards"].as_array().unwrap();
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0]["id"], "reference_alignment");
        assert!(cards
            .iter()
            .all(|c| c["operations"] == json!(["activation_rank_one"])));
    }

    #[test]
    fn prompt_only_never_offers_reference_alignment() {
        let value = catalogue(
            "prompts",
            &json!(["scale", "low_rank", "activation_rank_one"]),
            &Value::Null,
        );
        for card in value["cards"].as_array().unwrap() {
            assert_ne!(card["id"], "reference_alignment");
            assert!(!card["operations"]
                .as_array()
                .unwrap()
                .contains(&json!("activation_rank_one")));
        }
    }

    #[test]
    fn unsupported_or_missing_operations_offer_no_cards() {
        for operations in [Value::Null, json!([]), json!(["write_weights", "execute"])] {
            assert_eq!(
                catalogue("paired", &operations, &Value::Null)["cards"],
                json!([])
            );
        }
    }

    #[test]
    fn row_operations_require_worker_declaration_and_current_grounding() {
        let operations = json!(["token_row_scale", "token_row_interpolate"]);
        for inspection in [
            Value::Null,
            inspect(json!([])),
            inspect(json!([{"id":1000},{"id":-1}])),
        ] {
            assert_eq!(
                catalogue("paired", &operations, &inspection)["cards"],
                json!([])
            );
        }
        let inspection = inspect(json!([{"id":3},{"id":7}]));
        assert_eq!(
            catalogue("paired", &json!([]), &inspection)["cards"],
            json!([])
        );
        let value = catalogue("prompts", &operations, &inspection);
        assert_eq!(value["cards"][0]["id"], "token_row_scaling");
        assert_eq!(value["cards"][1]["id"], "token_row_transfer");
        assert_eq!(
            value["cards"][0]["eligible_tables"][0]["candidate_ids"],
            json!([3, 7])
        );
        assert_eq!(value["cards"][0]["eligible_tables"][0]["tied"], true);
    }

    #[test]
    fn interpolation_needs_two_distinct_candidates_and_a_matching_table() {
        let operations = json!(["token_row_scale", "token_row_interpolate"]);
        let value = catalogue("paired", &operations, &inspect(json!([{"id":3},{"id":3}])));
        assert_eq!(value["cards"].as_array().unwrap().len(), 2);
        assert_eq!(value["cards"][0]["id"], "token_row_scaling");
        assert_eq!(value["cards"][1]["operations"], json!(["token_row_scale"]));
        let mut inspection = inspect(json!([{"id":3},{"id":7}]));
        inspection["token_tables"][0]["operations"] = json!(["token_row_scale"]);
        let value = catalogue("paired", &operations, &inspection);
        assert!(value["cards"]
            .as_array()
            .unwrap()
            .iter()
            .all(|card| card["id"] != "token_row_transfer"));
    }

    #[test]
    fn catalogue_is_deterministic_and_has_explicit_limitations() {
        let operations = json!([
            "scale",
            "low_rank",
            "noise",
            "activation_rank_one",
            "token_row_scale",
            "token_row_interpolate"
        ]);
        let inspection = inspect(json!([{"id":7},{"id":3}]));
        let value = catalogue("paired", &operations, &inspection);
        assert_eq!(value, catalogue("paired", &operations, &inspection));
        for card in value["cards"].as_array().unwrap() {
            assert!(!card["evidence"].as_str().unwrap().is_empty());
            assert!(!card["limitation"].as_str().unwrap().is_empty());
        }
    }
    #[test]
    fn measured_skills_require_declared_capability_and_reference_tokens() {
        let ops = json!(["activation_preserve", "token_row_residual"]);
        let mut inspection = inspect(json!([{"id":3,"edit_reference":true}]));
        inspection["token_tables"][0]["operations"] = json!(["token_row_residual"]);
        let cards = catalogue("paired", &ops, &inspection);
        assert!(cards["cards"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == "preservation_aware"));
        assert!(cards["cards"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == "residual_output_row"));
        assert_eq!(catalogue("prompts", &ops, &inspection)["cards"], json!([]));
        inspection["token_candidates"][0]["edit_reference"] = json!(false);
        assert!(!catalogue("paired", &ops, &inspection)["cards"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == "residual_output_row"));
        assert_eq!(
            catalogue("paired", &json!([]), &inspection)["cards"],
            json!([])
        );
    }
}
