use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::editor::OpenAiUsage;

// Public, standard-tier text rates checked 2026-11-12. Account-specific charges,
// taxes and future provider price changes are not an invoice guarantee.
// https://developers.openai.com/api/docs/pricing
// https://developers.openai.com/api/docs/guides/prompt-caching
const LONG_CONTEXT_THRESHOLD: u64 = 272_000;

pub(crate) struct OpenAiRates {
    pub input: f64,
    pub output: f64,
    cached: f64,
}

pub(crate) fn openai_rates(model: &str) -> Option<OpenAiRates> {
    let (input, output, cached) = match model {
        "gpt-6.1-sol" => (2., 10., 0.10),
        "gpt-6-astra" => (10., 50., 1.),
        "gpt-6.1-terra" => (1., 5., 0.05),
        "gpt-6-luna" => (0.1, 0.5, 0.01),
        _ => return None,
    };
    Some(OpenAiRates {
        input,
        output,
        cached,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub limit_usd: f64,
    pub spent_usd: f64,
    pub reserved_usd: f64,
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub mode: String,
    #[serde(default)]
    pub pricing_basis: String,
    #[serde(default)]
    pub last_cost_basis: String,
    #[serde(default)]
    pub openai_model: Option<String>,
}
impl Budget {
    pub fn new(
        limit: f64,
        model: &str,
        input: Option<f64>,
        output: Option<f64>,
        mode: &str,
    ) -> Result<Self> {
        let openai = openai_rates(model);
        let rates = if let Some(rates) = &openai {
            (rates.input, rates.output)
        } else {
            match model {
                "claude-sonnet-5-5" => (2., 10.),
                "claude-opus-5-5" => (4., 20.),
                "claude-sonnet-4-6" => (3., 15.),
                "claude-opus-4-6" => (5., 25.),
                _ => {
                    if input.is_none() || output.is_none() {
                        bail!("Pricing is unknown for {model}. Provide --input-price and --output-price in USD per million tokens")
                    }
                    (0., 0.)
                }
            }
        };
        let i = input.unwrap_or(rates.0);
        let o = output.unwrap_or(rates.1);
        if !limit.is_finite()
            || limit <= 0.
            || !i.is_finite()
            || i <= 0.
            || !o.is_finite()
            || o <= 0.
        {
            bail!("Budget and token prices must be positive finite numbers");
        }
        // Overrides retain their documented flat per-token meaning. Never apply
        // provider cache/long-context multipliers silently to a custom price.
        let custom = input.is_some() || output.is_some();
        let use_openai = openai.is_some() && !custom && mode != "estimated_between_calls";
        Ok(Self {
            limit_usd: limit,
            spent_usd: 0.,
            reserved_usd: 0.,
            input_per_million: i,
            output_per_million: o,
            input_tokens: 0,
            output_tokens: 0,
            mode: mode.into(),
            pricing_basis: if custom {
                "custom_flat_estimate"
            } else if use_openai {
                "openai_standard_list_prices_2026_11_12"
            } else {
                "configured_token_estimate"
            }
            .into(),
            last_cost_basis: "not_settled".into(),
            openai_model: use_openai.then(|| model.to_owned()),
        })
    }
    pub fn configure_provider(&mut self, editor: &str, strict: bool) -> Result<()> {
        if editor != "openai" {
            self.openai_model = None;
            if self.pricing_basis.starts_with("openai_") {
                self.pricing_basis = "configured_token_estimate".into();
            }
        }
        if self.pricing_basis == "custom_flat_estimate" {
            if strict {
                bail!("Strict budget policy cannot verify custom token prices or unknown model pricing. Remove price overrides and use a supported model, or choose auto for explicitly estimated spending")
            }
            if self.mode == "reserved_api" {
                self.mode = "estimated_custom_api".into();
            }
        }
        if editor == "openai" && self.openai_model.is_none() && self.mode == "reserved_api" {
            if strict {
                bail!("Strict OpenAI budgeting requires a supported OpenAI pricing model")
            }
            self.mode = "estimated_custom_api".into();
        }
        Ok(())
    }
    pub fn remaining(&self) -> f64 {
        (self.limit_usd - self.spent_usd - self.reserved_usd).max(0.)
    }
    pub fn reservation(&self, context: &Value, schema: &Value, max_output: u32) -> f64 {
        // UTF-8 byte count bounds text tokens; include schema and framing.
        let input = crate::editor::conservative_input_tokens(context, schema);
        let (input_multiplier, output_multiplier) = if self.openai_model.is_some() {
            // All input might be a cache miss written for the pinned 30m TTL.
            if input > LONG_CONTEXT_THRESHOLD {
                (2.5, 1.5)
            } else {
                (1.25, 1.)
            }
        } else {
            (1., 1.)
        };
        (input as f64 * self.input_per_million * input_multiplier
            + max_output as f64 * self.output_per_million * output_multiplier)
            / 1e6
    }
    pub fn reserve(&mut self, amount: f64) -> Result<()> {
        if !amount.is_finite() || amount <= 0. {
            bail!("Invalid inference cost reservation")
        }
        if self.reserved_usd > 0.
            || self.last_cost_basis == "unresolved_usage"
            || self.last_cost_basis.ends_with("_reservation_breach")
        {
            bail!("A previous inference request has unresolved usage; reconcile its provider receipt before continuing")
        }
        if amount > self.remaining() {
            bail!(
                "Remaining inference budget ${:.4} cannot cover the next request (${amount:.4})",
                self.remaining()
            )
        }
        self.reserved_usd = amount;
        Ok(())
    }
    pub fn mark_unresolved(&mut self) {
        // Keep the full reservation. Missing usage is not a free request and
        // must never free capacity for another potentially billable call.
        self.last_cost_basis = "unresolved_usage".into();
    }
    pub fn settle(
        &mut self,
        input: u64,
        output: u64,
        reported: Option<f64>,
        usage: Option<&OpenAiUsage>,
    ) -> Result<f64> {
        let calculation = if let Some(model) = &self.openai_model {
            openai_cost(model, input, output, usage)
        } else {
            Ok((
                reported.unwrap_or(
                    (input as f64 * self.input_per_million
                        + output as f64 * self.output_per_million)
                        / 1e6,
                ),
                if reported.is_some() {
                    "provider_reported"
                } else {
                    self.pricing_basis.as_str()
                },
                false,
            ))
        };
        let (cost, basis, changed_policy) = match calculation {
            Ok(value) => value,
            Err(error) => {
                self.mark_unresolved();
                return Err(error);
            }
        };
        if !cost.is_finite() || cost < 0. || !(self.spent_usd + cost).is_finite() {
            self.mark_unresolved();
            bail!("Editor returned invalid usage cost")
        }
        let basis = basis.to_owned();
        let (Some(total_input), Some(total_output)) = (
            self.input_tokens.checked_add(input),
            self.output_tokens.checked_add(output),
        ) else {
            self.mark_unresolved();
            bail!("Editor token totals overflowed; cost remains unresolved")
        };
        let exceeded = self.mode == "reserved_api" && cost > self.reserved_usd + 1e-10;
        self.reserved_usd = 0.;
        self.spent_usd += cost;
        self.input_tokens = total_input;
        self.output_tokens = total_output;
        self.last_cost_basis = basis;
        // Record the measured charge even on a policy mismatch, then stop. A
        // provider returning an unexpected tier/TTL must not enable more calls.
        if changed_policy || exceeded {
            self.last_cost_basis.push_str("_reservation_breach");
            bail!("The provider's usage or billing policy did not match the reserved request policy. Recorded the recovered cost and stopped; check the provider receipt")
        }
        Ok(cost)
    }
}

fn openai_cost(
    model: &str,
    input: u64,
    output: u64,
    usage: Option<&OpenAiUsage>,
) -> Result<(f64, &'static str, bool)> {
    let rates = openai_rates(model).context("OpenAI model pricing is unresolved")?;
    let usage = usage.context("OpenAI billing usage is missing; cost remains unresolved")?;
    if !usage.valid || usage.model.as_deref() != Some(model) {
        bail!("OpenAI returned invalid usage or an unverified model; cost remains unresolved")
    }
    let tier = match usage.service_tier.as_deref() {
        Some("default") => 1.,
        Some("priority" | "fast") => 2.,
        Some("ultrafast") => 6.,
        Some("flex") => 0.5,
        _ => bail!("OpenAI returned no verified billing tier; cost remains unresolved"),
    };
    let cached = usage.cached_tokens.unwrap_or(0);
    let available = input
        .checked_sub(cached)
        .context("Cached tokens exceed input usage")?;
    let written = usage.cache_write_tokens.unwrap_or(available);
    let ordinary = available
        .checked_sub(written)
        .context("Cache reads and writes exceed input usage")?;
    let ttl_12h = usage.ttl_12h_tokens.unwrap_or(0);
    let has_lifetimes = usage.ttl_12h_tokens.is_some() || usage.ttl_30m_tokens.is_some();
    let ttl_30m = usage
        .ttl_30m_tokens
        .unwrap_or(if has_lifetimes { 0 } else { written });
    let unknown_lifetime = written
        .checked_sub(ttl_12h)
        .and_then(|n| n.checked_sub(ttl_30m))
        .context("Cache write lifetimes exceed cache-write usage")?;
    let estimated =
        usage.cached_tokens.is_none() || usage.cache_write_tokens.is_none() || unknown_lifetime > 0;
    // Missing cache counts use the maximum 30m cost under the explicit request
    // policy. Partial TTL detail is charged at the longer, more expensive TTL.
    let input_cost = ordinary as f64 * rates.input
        + cached as f64 * rates.cached
        + ttl_30m as f64 * rates.input * 1.25
        + (ttl_12h as f64 + unknown_lifetime as f64) * rates.input * 2.5;
    let (input_multiplier, output_multiplier) = if input > LONG_CONTEXT_THRESHOLD {
        (2., 1.5)
    } else {
        (1., 1.)
    };
    let cost = (input_cost * input_multiplier + output as f64 * rates.output * output_multiplier)
        * tier
        / 1e6;
    Ok((
        cost,
        if estimated {
            "openai_conservative_cache_estimate"
        } else {
            "openai_usage_list_prices"
        },
        tier != 1. || ttl_12h > 0 || unknown_lifetime > 0,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn complete_usage(cached: u64, written: u64) -> OpenAiUsage {
        OpenAiUsage {
            cached_tokens: Some(cached),
            cache_write_tokens: Some(written),
            service_tier: Some("default".into()),
            model: Some("gpt-6.1-sol".into()),
            valid: true,
            ..OpenAiUsage::default()
        }
    }
    #[test]
    fn refuses_unaffordable_call() {
        let mut b = Budget::new(0.01, "gpt-6.1-sol", None, None, "reserved").unwrap();
        let v = b.reservation(&json!({}), &json!({}), 1024);
        assert!(b.reserve(v).is_err());
        assert_eq!(b.spent_usd, 0.);
    }
    #[test]
    fn usage_and_unknown_reservation() {
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved").unwrap();
        b.reserve(0.1).unwrap();
        assert!((b.remaining() - 0.9).abs() < 1e-10);
        b.settle(1000, 100, None, Some(&complete_usage(0, 0)))
            .unwrap();
        assert!((b.spent_usd - 0.003).abs() < 1e-10);
        assert_eq!(b.reserved_usd, 0.);
    }
    #[test]
    fn rejects_bad_rates() {
        assert!(Budget::new(1., "unknown", None, None, "reserved").is_err());
        assert!(Budget::new(f64::NAN, "gpt-6.1-sol", None, None, "reserved").is_err());
    }
    #[test]
    fn standard_cache_reads_and_writes_are_distinct_charges() {
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
        b.reserve(0.1).unwrap();
        let cost = b
            .settle(1000, 100, None, Some(&complete_usage(500, 300)))
            .unwrap();
        assert!((cost - 0.0022).abs() < 1e-12);
        assert_eq!(b.last_cost_basis, "openai_usage_list_prices");
        assert_eq!((b.input_tokens, b.output_tokens), (1000, 100));
        assert_eq!(b.reserved_usd, 0.);
    }

    #[test]
    fn cache_reads_use_each_models_verified_price() {
        for (model, cached_price) in [
            ("gpt-6.1-sol", 0.1),
            ("gpt-6.1-terra", 0.05),
            ("gpt-6-astra", 1.),
            ("gpt-6-luna", 0.01),
        ] {
            let mut usage = complete_usage(1000, 0);
            usage.model = Some(model.into());
            let (cost, _, _) = openai_cost(model, 1000, 0, Some(&usage)).unwrap();
            assert!((cost - cached_price / 1000.).abs() < 1e-12);
        }
    }

    #[test]
    fn reservation_covers_cold_cache_and_long_context_output() {
        for length in [100, 280_000] {
            let context = json!({"text":"x".repeat(length)});
            let schema = json!({});
            let input = crate::editor::conservative_input_tokens(&context, &schema);
            let mut b = Budget::new(10., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
            let reserved = b.reservation(&context, &schema, 1024);
            let (cost, _, _) =
                openai_cost("gpt-6.1-sol", input, 1024, Some(&complete_usage(0, input))).unwrap();
            assert!((reserved - cost).abs() < 1e-12);
            b.reserve(reserved).unwrap();
            b.settle(input, 1024, None, Some(&complete_usage(0, input)))
                .unwrap();
            assert!(b.spent_usd <= 10.);
        }
    }

    #[test]
    fn long_context_threshold_uses_total_input_including_cached_tokens() {
        for (input, im, om) in [(272_000, 1., 1.), (272_001, 2., 1.5)] {
            let (cost, _, _) =
                openai_cost("gpt-6.1-sol", input, 100, Some(&complete_usage(input, 0))).unwrap();
            let expected = (input as f64 * 0.1 * im + 100. * 10. * om) / 1e6;
            assert!((cost - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn mixed_write_lifetimes_are_recorded_before_stopping() {
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
        b.reserve(0.1).unwrap();
        let usage = OpenAiUsage {
            ttl_30m_tokens: Some(400),
            ttl_12h_tokens: Some(200),
            ..complete_usage(200, 600)
        };
        assert!(b.settle(1000, 100, None, Some(&usage)).is_err());
        assert!((b.spent_usd - 0.00342).abs() < 1e-12);
        assert!(b.last_cost_basis.ends_with("_reservation_breach"));
        assert!(b.reserve(0.01).is_err());
    }

    #[test]
    fn missing_cache_counts_are_explicit_conservative_estimates() {
        let usage = OpenAiUsage {
            cached_tokens: None,
            cache_write_tokens: None,
            ..complete_usage(0, 0)
        };
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
        b.reserve(0.1).unwrap();
        let cost = b.settle(1000, 100, None, Some(&usage)).unwrap();
        assert!((cost - 0.0035).abs() < 1e-12);
        assert_eq!(b.last_cost_basis, "openai_conservative_cache_estimate");
    }

    #[test]
    fn incomplete_lifetime_breakdown_uses_expensive_bound_and_stops() {
        let usage = OpenAiUsage {
            ttl_30m_tokens: Some(500),
            ..complete_usage(0, 1000)
        };
        let (cost, basis, stop) = openai_cost("gpt-6.1-sol", 1000, 100, Some(&usage)).unwrap();
        assert!((cost - 0.00475).abs() < 1e-12);
        assert_eq!(basis, "openai_conservative_cache_estimate");
        assert!(stop);
    }

    #[test]
    fn missing_usage_model_or_tier_never_releases_reservation() {
        for usage in [
            None,
            Some(OpenAiUsage {
                model: None,
                ..complete_usage(0, 0)
            }),
            Some(OpenAiUsage {
                service_tier: None,
                ..complete_usage(0, 0)
            }),
            Some(OpenAiUsage {
                service_tier: Some("auto".into()),
                ..complete_usage(0, 0)
            }),
            Some(OpenAiUsage {
                model: Some("gpt-6-astra".into()),
                ..complete_usage(0, 0)
            }),
            Some(OpenAiUsage {
                valid: false,
                ..complete_usage(0, 0)
            }),
        ] {
            let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
            b.reserve(0.1).unwrap();
            assert!(b.settle(1000, 100, None, usage.as_ref()).is_err());
            assert_eq!(b.spent_usd, 0.);
            assert_eq!(b.reserved_usd, 0.1);
            assert_eq!(b.last_cost_basis, "unresolved_usage");
            assert!(b.reserve(0.01).is_err());
        }
    }

    #[test]
    fn impossible_cache_counts_do_not_underflow_or_free_budget() {
        for usage in [
            complete_usage(1001, 0),
            complete_usage(500, 501),
            OpenAiUsage {
                ttl_30m_tokens: Some(400),
                ttl_12h_tokens: Some(101),
                ..complete_usage(0, 500)
            },
            complete_usage(u64::MAX, u64::MAX),
        ] {
            assert!(openai_cost("gpt-6.1-sol", 1000, 100, Some(&usage)).is_err());
        }
    }

    #[test]
    fn unexpected_tier_is_charged_and_stops_further_spending() {
        for (tier, multiplier) in [
            ("priority", 2.),
            ("fast", 2.),
            ("ultrafast", 6.),
            ("flex", 0.5),
        ] {
            let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
            b.reserve(0.1).unwrap();
            let usage = OpenAiUsage {
                service_tier: Some(tier.into()),
                ..complete_usage(0, 0)
            };
            assert!(b.settle(1000, 100, None, Some(&usage)).is_err());
            assert!((b.spent_usd - 0.003 * multiplier).abs() < 1e-12);
            assert_eq!(b.reserved_usd, 0.);
            assert!(b.reserve(0.01).is_err());
        }
    }

    #[test]
    fn custom_prices_remain_flat_and_cannot_claim_strict_metering() {
        for model in ["gpt-6.1-sol", "unknown"] {
            let mut b = Budget::new(20., model, Some(3.), Some(7.), "reserved_api").unwrap();
            assert!(b.configure_provider("openai", true).is_err());
            b.configure_provider("openai", false).unwrap();
            assert_eq!(b.mode, "estimated_custom_api");
            assert!(b.openai_model.is_none());
            b.reserve(10.).unwrap();
            b.settle(300_000, 100, None, Some(&complete_usage(1000, 2000)))
                .unwrap();
            assert!((b.spent_usd - 0.9007).abs() < 1e-12);
            assert_eq!(b.last_cost_basis, "custom_flat_estimate");
        }
        let mut partial = Budget::new(1., "gpt-6.1-sol", Some(3.), None, "reserved_api").unwrap();
        assert!(partial.configure_provider("openai", true).is_err());
        assert_eq!(partial.output_per_million, 10.);
        assert!(partial.openai_model.is_none());
    }

    #[test]
    fn measured_overrun_is_not_erased_by_error() {
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
        b.reserve(0.001).unwrap();
        assert!(b
            .settle(1000, 100, None, Some(&complete_usage(0, 0)))
            .is_err());
        assert!((b.spent_usd - 0.003).abs() < 1e-12);
        assert_eq!(b.input_tokens, 1000);
        assert_eq!(b.reserved_usd, 0.);
        assert!(b.reserve(0.01).is_err());
    }

    #[test]
    fn invalid_or_duplicate_reservations_never_change_balance() {
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "reserved_api").unwrap();
        for amount in [f64::NAN, f64::INFINITY, -1., 0.] {
            assert!(b.reserve(amount).is_err());
            assert_eq!(b.reserved_usd, 0.);
        }
        b.reserve(0.1).unwrap();
        assert!(b.reserve(0.1).is_err());
        b.mark_unresolved();
        assert_eq!(b.reserved_usd, 0.1);
    }

    #[test]
    fn cli_estimates_do_not_inherit_api_cache_policy() {
        let mut b = Budget::new(1., "gpt-6.1-sol", None, None, "estimated_between_calls").unwrap();
        b.configure_provider("codex", false).unwrap();
        b.reserve(0.1).unwrap();
        b.settle(1000, 100, None, None).unwrap();
        assert!((b.spent_usd - 0.003).abs() < 1e-12);
        assert_eq!(b.last_cost_basis, "configured_token_estimate");
    }
}
