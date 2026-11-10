use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
}
impl Budget {
    pub fn new(
        limit: f64,
        model: &str,
        input: Option<f64>,
        output: Option<f64>,
        mode: &str,
    ) -> Result<Self> {
        let rates = match model {
            "gpt-6.1-sol" => (2., 10.),
            "gpt-6-astra" => (10., 50.),
            "gpt-6.1-terra" => (1., 5.),
            "gpt-6-luna" => (0.1, 0.5),
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
        Ok(Self {
            limit_usd: limit,
            spent_usd: 0.,
            reserved_usd: 0.,
            input_per_million: i,
            output_per_million: o,
            input_tokens: 0,
            output_tokens: 0,
            mode: mode.into(),
        })
    }
    pub fn remaining(&self) -> f64 {
        (self.limit_usd - self.spent_usd - self.reserved_usd).max(0.)
    }
    pub fn reservation(&self, context: &Value, schema: &Value, max_output: u32) -> f64 {
        // UTF-8 byte count upper-bounds byte-level text tokens; include schema and envelope allowance.
        let bytes = crate::editor::conservative_input_tokens(context, schema);
        (bytes as f64 * self.input_per_million + max_output as f64 * self.output_per_million) / 1e6
    }
    pub fn reserve(&mut self, amount: f64) -> Result<()> {
        if amount > self.remaining() {
            bail!(
                "Remaining inference budget ${:.4} cannot cover the next request (${amount:.4})",
                self.remaining()
            )
        }
        self.reserved_usd += amount;
        Ok(())
    }
    pub fn settle(&mut self, input: u64, output: u64, reported: Option<f64>) -> Result<f64> {
        let cost = reported.unwrap_or(
            (input as f64 * self.input_per_million + output as f64 * self.output_per_million) / 1e6,
        );
        if !cost.is_finite() || cost < 0. {
            bail!("Editor returned invalid usage cost")
        }
        self.reserved_usd = 0.;
        self.spent_usd += cost;
        self.input_tokens += input;
        self.output_tokens += output;
        Ok(cost)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
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
        b.settle(1000, 100, None).unwrap();
        assert!((b.spent_usd - 0.003).abs() < 1e-10);
        assert_eq!(b.reserved_usd, 0.);
    }
    #[test]
    fn rejects_bad_rates() {
        assert!(Budget::new(1., "unknown", None, None, "reserved").is_err());
        assert!(Budget::new(f64::NAN, "gpt-6.1-sol", None, None, "reserved").is_err());
    }
}
