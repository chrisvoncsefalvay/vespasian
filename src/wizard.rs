use std::io;
use std::path::PathBuf;
use std::time::Duration;

use crate::preview::{PreviewClient, PreviewRequest};
use anyhow::Result;
use crossterm::{
    cursor::Show,
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use serde_json::Value;

const INK: Color = Color::Rgb(20, 23, 27);
const IVORY: Color = Color::Rgb(236, 226, 210);
const COPPER: Color = Color::Rgb(214, 160, 113);
const MUTED: Color = Color::Rgb(148, 156, 165);
const RULE: Color = Color::Rgb(62, 66, 72);
const GREEN: Color = Color::Rgb(151, 185, 147);
const RED: Color = Color::Rgb(224, 147, 137);

const STEPS: [&str; 11] = [
    "Behaviour",
    "Model",
    "Dataset",
    "Objective",
    "Measure",
    "Health checks",
    "Editor",
    "Budget",
    "Checkpoint",
    "Trajectory",
    "Review",
];
const DATA_MODES: [(&str, &str); 3] = [
    ("paired", "Paired examples"),
    ("corpus", "Text corpus"),
    ("prompts", "Prompts"),
];

const TARGETS: [(&str, &str, &str); 5] = [
    (
        "factual",
        "Factual answers",
        "Teach or correct information represented by the examples.",
    ),
    (
        "style",
        "Style and tone",
        "Shape how the model writes and responds.",
    ),
    (
        "format",
        "Response format",
        "Develop a consistent structure or output convention.",
    ),
    (
        "classification",
        "Classification",
        "Map inputs to the labels in your dataset.",
    ),
    (
        "custom",
        "Custom behaviour",
        "Describe another behaviour in your own words.",
    ),
];
const MODELS: [(&str, &str, &str); 3] = [
    ("Qwen3-0.8B", "Qwen 3.5 · 0.8B", "Qwen/Qwen3.5-0.8B"),
    ("Gemma3-1B", "Gemma 3 · 1B", "google/gemma-3-1b-it"),
    (
        "custom",
        "Another checkpoint",
        "Enter a Hugging Face model ID or local directory.",
    ),
];
const METRICS: [(&str, &str, &str); 4] = [
    (
        "auto",
        "Let the editor choose",
        "Assess the goal and examples, then select a measure.",
    ),
    (
        "exact_match",
        "Exact match",
        "Compare answers with the supplied reference text.",
    ),
    (
        "token_f1",
        "Token F1",
        "Measure overlap between generated and reference answers.",
    ),
    (
        "semantic",
        "Goal score",
        "Use the frontier model to assess the requested behaviour.",
    ),
];
const EDITORS: [(&str, &str, &str); 4] = [
    ("codex", "Codex", "Use the Codex login on this machine."),
    ("claude", "Claude", "Use the Claude login on this machine."),
    (
        "openai",
        "OpenAI API",
        "Use OPENAI_API_KEY from your environment.",
    ),
    (
        "anthropic",
        "Anthropic API",
        "Use ANTHROPIC_API_KEY from your environment.",
    ),
];
const PUBLISH: [(&str, &str, &str); 2] = [
    (
        "local",
        "Keep the checkpoint locally",
        "Save the model in the run directory.",
    ),
    (
        "hub",
        "Upload to Hugging Face",
        "Upload the verified checkpoint to a model repository.",
    ),
];

const TRAJECTORIES: [(&str, &str, &str); 3] = [
    (
        "off",
        "Do not save a trajectory",
        "Keep the normal run reports.",
    ),
    (
        "local",
        "Save a local trajectory",
        "Write editor exchanges and measured outcomes as JSONL.",
    ),
    (
        "hub",
        "Save and upload to Hugging Face",
        "Store the trajectory in a dataset repository.",
    ),
];

#[derive(Clone)]
pub struct WizardOptions {
    pub python: PathBuf,
    pub model: String,
    pub editor: String,
    pub metric: String,
    pub target_type: Option<String>,
    pub limit: u32,
    pub dataset_config: Option<String>,
    pub dataset_split: String,
    pub dataset_revision: Option<String>,
    pub dataset_mode: String,
    pub text_field: Option<String>,
    pub prompt_field: Option<String>,
    pub target_field: Option<String>,
    pub goal: Option<String>,
    pub editor_model: Option<String>,
    pub budget: f64,
    pub controls: String,
    pub hf_token: Option<String>,
    pub publish_repo: Option<String>,
    pub publish_visibility: String,
    pub publish_create: bool,
    pub health_checks: Vec<crate::health::Selection>,
    pub health_max_drop: Option<f64>,
    pub save_trajectory: bool,
    pub trajectory_repo: Option<String>,
    pub trajectory_visibility: String,
    pub trajectory_create: bool,
}

impl Default for WizardOptions {
    fn default() -> Self {
        Self {
            python: PathBuf::from(crate::platform::default_python()),
            model: "Qwen3-0.8B".into(),
            editor: "codex".into(),
            metric: "auto".into(),
            target_type: None,
            limit: 100,
            dataset_config: None,
            dataset_split: "train".into(),
            dataset_revision: None,
            dataset_mode: "paired".into(),
            text_field: None,
            prompt_field: None,
            target_field: None,
            goal: None,
            editor_model: None,
            budget: 5.0,
            controls: String::new(),
            hf_token: None,
            publish_repo: None,
            publish_visibility: "private".into(),
            publish_create: false,
            health_checks: Vec::new(),
            health_max_drop: None,
            save_trajectory: false,
            trajectory_repo: None,
            trajectory_visibility: "private".into(),
            trajectory_create: false,
        }
    }
}

#[derive(Clone)]
pub struct WizardResult {
    pub target_type: String,
    pub model: String,
    pub dataset: String,
    pub dataset_mode: String,
    pub text_field: String,
    pub goal: String,
    pub metric: String,
    pub budget: f64,
    pub editor: String,
    pub editor_model: Option<String>,
    pub prompt_field: String,
    pub target_field: String,
    pub dataset_split: String,
    pub dataset_config: Option<String>,
    pub dataset_revision: Option<String>,
    pub limit: u32,
    pub controls: String,
    pub hf_token: Option<String>,
    pub publish_repo: Option<String>,
    pub publish_visibility: String,
    pub publish_create: bool,
    pub health_checks: Vec<crate::health::Selection>,
    pub health_max_drop: Option<f64>,
    pub save_trajectory: bool,
    pub trajectory_repo: Option<String>,
    pub trajectory_visibility: String,
    pub trajectory_create: bool,
}

#[derive(Default, Debug, Clone)]
struct Input {
    text: String,
    /// A character index, never a byte offset.
    cursor: usize,
}

impl Input {
    fn new(text: &str) -> Self {
        let mut input = Self::default();
        input.insert(text);
        input
    }
    fn byte(&self, index: usize) -> usize {
        self.text
            .char_indices()
            .nth(index)
            .map_or(self.text.len(), |(offset, _)| offset)
    }
    fn insert(&mut self, text: &str) {
        let text: String = text.chars().filter_map(|ch| {
            if ch == '\n' || ch == '\r' || ch == '\t' { Some(' ') }
            else if ch.is_control() || matches!(ch, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}') { None }
            else { Some(ch) }
        }).collect();
        self.text.insert_str(self.byte(self.cursor), &text);
        self.cursor += text.chars().count();
    }
    fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => self.cursor = 0,
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = self.text.chars().count()
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.text.clear();
                self.cursor = 0;
            }
            KeyCode::Char(ch)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert(&ch.to_string())
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.text.chars().count()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.text.chars().count(),
            KeyCode::Backspace if self.cursor > 0 => {
                let end = self.byte(self.cursor);
                let begin = self.byte(self.cursor - 1);
                self.text.replace_range(begin..end, "");
                self.cursor -= 1;
            }
            KeyCode::Delete if self.cursor < self.text.chars().count() => {
                self.text
                    .replace_range(self.byte(self.cursor)..self.byte(self.cursor + 1), "");
            }
            _ => {}
        }
    }
    fn line(&self, columns: usize, active: bool) -> Line<'static> {
        if !active {
            return Line::from(self.text.clone());
        }
        let prefix: String = self.text.chars().take(self.cursor).collect();
        let mut start = 0;
        let mut width = Span::raw(prefix.as_str()).width();
        for ch in prefix.chars() {
            if width < columns.saturating_sub(1) {
                break;
            }
            start += 1;
            width = width.saturating_sub(Span::raw(ch.to_string()).width());
        }
        let before: String = self
            .text
            .chars()
            .skip(start)
            .take(self.cursor.saturating_sub(start))
            .collect();
        let cursor = self.text.chars().nth(self.cursor).unwrap_or(' ');
        let after: String = self.text.chars().skip(self.cursor + 1).collect();
        Line::from(vec![
            Span::raw(before),
            Span::styled(cursor.to_string(), Style::default().bg(COPPER).fg(INK)),
            Span::raw(after),
        ])
    }
}

struct HealthItem {
    id: String,
    label: String,
    group: String,
    enabled: bool,
    count: Input,
}

fn health_items() -> Vec<HealthItem> {
    crate::health::catalogue()
        .into_iter()
        .map(|definition| HealthItem {
            id: definition.id.into(),
            label: definition.label.into(),
            group: definition.group.into(),
            enabled: false,
            count: Input::new("20"),
        })
        .collect()
}

struct Form {
    step: usize,
    target: usize,
    custom_target_type: Option<String>,
    model: usize,
    metric: usize,
    editor: usize,
    field: usize,
    typing: bool,
    custom_model: Input,
    dataset: Input,
    dataset_mode: usize,
    text_field: Input,
    goal: Input,
    budget: Input,
    editor_model: Input,
    prompt_field: Input,
    target_field: Input,
    dataset_split: Input,
    dataset_config: Input,
    dataset_revision: Option<String>,
    limit: Input,
    controls: Input,
    hf_token: Input,
    publish: usize,
    publish_repo: Input,
    publish_visibility: Input,
    publish_mode: Input,
    health_items: Vec<HealthItem>,
    health_index: usize,
    health_drop: Input,
    trajectory: usize,
    trajectory_repo: Input,
    trajectory_visibility: Input,
    trajectory_mode: Input,
    generated_goal: Option<String>,
    preview: Option<Value>,
    preview_for: Option<PreviewRequest>,
    confirmed_mapping: Option<PreviewRequest>,
    loading_for: Option<PreviewRequest>,
    preview_row: usize,
    preview_scroll: u16,
    review_scroll: u16,
    error: String,
}

impl Default for Form {
    fn default() -> Self {
        Self {
            step: 0,
            target: 0,
            custom_target_type: None,
            model: 0,
            metric: 0,
            editor: 0,
            field: 0,
            typing: false,
            custom_model: Input::default(),
            dataset: Input::default(),
            dataset_mode: 0,
            text_field: Input::default(),
            goal: Input::default(),
            budget: Input::new("5"),
            editor_model: Input::default(),
            prompt_field: Input::default(),
            target_field: Input::default(),
            dataset_split: Input::new("train"),
            dataset_config: Input::default(),
            dataset_revision: None,
            limit: Input::new("100"),
            controls: Input::default(),
            hf_token: Input::default(),
            publish: 0,
            publish_repo: Input::default(),
            publish_visibility: Input::new("private"),
            publish_mode: Input::new("existing"),
            health_items: health_items(),
            health_index: 0,
            health_drop: Input::default(),
            trajectory: 0,
            trajectory_repo: Input::default(),
            trajectory_visibility: Input::new("private"),
            trajectory_mode: Input::new("existing"),
            generated_goal: None,
            preview: None,
            preview_for: None,
            confirmed_mapping: None,
            loading_for: None,
            preview_row: 0,
            preview_scroll: 0,
            review_scroll: 0,
            error: String::new(),
        }
    }
}

enum Action {
    Continue,
    Cancel,
    Start(Box<WizardResult>),
    Preview(PreviewRequest),
}

impl Form {
    fn from_options(options: &WizardOptions) -> Self {
        let target = TARGETS
            .iter()
            .position(|(id, _, _)| Some(*id) == options.target_type.as_deref());
        let custom_target_type = options.target_type.clone().filter(|_| target.is_none());
        let model = MODELS
            .iter()
            .position(|(id, _, _)| *id == options.model)
            .unwrap_or(2);
        let mut items = health_items();
        for item in &mut items {
            if let Some(selected) = options
                .health_checks
                .iter()
                .find(|selection| selection.check == item.id)
            {
                item.enabled = true;
                item.count = Input::new(&selected.examples.to_string());
            }
        }
        Self {
            target: target.unwrap_or(if custom_target_type.is_some() {
                TARGETS.len() - 1
            } else {
                0
            }),
            custom_target_type,
            model,
            custom_model: if model == 2 {
                Input::new(&options.model)
            } else {
                Input::default()
            },
            editor: EDITORS
                .iter()
                .position(|(id, _, _)| *id == options.editor)
                .unwrap_or(0),
            dataset_mode: DATA_MODES
                .iter()
                .position(|(id, _)| *id == options.dataset_mode)
                .unwrap_or(0),
            text_field: Input::new(options.text_field.as_deref().unwrap_or("")),
            metric: if options.dataset_mode == "paired" {
                METRICS
                    .iter()
                    .position(|(id, _, _)| *id == options.metric)
                    .unwrap_or(0)
            } else {
                3
            },
            health_items: items,
            health_drop: Input::new(
                &options
                    .health_max_drop
                    .map(|value| (value * 100.0).to_string())
                    .unwrap_or_default(),
            ),
            trajectory: if options.trajectory_repo.is_some() {
                2
            } else {
                usize::from(options.save_trajectory)
            },
            trajectory_repo: Input::new(options.trajectory_repo.as_deref().unwrap_or("")),
            trajectory_visibility: Input::new(&options.trajectory_visibility),
            trajectory_mode: Input::new(if options.trajectory_create {
                "new"
            } else {
                "existing"
            }),
            prompt_field: Input::new(options.prompt_field.as_deref().unwrap_or("")),
            target_field: Input::new(options.target_field.as_deref().unwrap_or("")),
            dataset_config: Input::new(options.dataset_config.as_deref().unwrap_or("")),
            dataset_split: Input::new(&options.dataset_split),
            dataset_revision: options.dataset_revision.clone(),
            limit: Input::new(&options.limit.to_string()),
            goal: Input::new(options.goal.as_deref().unwrap_or("")),
            editor_model: Input::new(options.editor_model.as_deref().unwrap_or("")),
            budget: Input::new(&options.budget.to_string()),
            controls: Input::new(&options.controls),
            hf_token: Input::new(options.hf_token.as_deref().unwrap_or("")),
            publish: usize::from(options.publish_repo.is_some()),
            publish_repo: Input::new(options.publish_repo.as_deref().unwrap_or("")),
            publish_visibility: Input::new(&options.publish_visibility),
            publish_mode: Input::new(if options.publish_create {
                "new"
            } else {
                "existing"
            }),
            ..Self::default()
        }
    }

    fn dataset_mode(&self) -> &str {
        DATA_MODES[self.dataset_mode].0
    }

    fn dataset_field_indices(&self) -> &'static [usize] {
        match (self.remote(), self.dataset_mode) {
            (false, 0) => &[0, 6],
            (false, _) => &[0, 3, 6],
            (true, 0) => &[0, 1, 2, 3, 4, 5, 6],
            (true, _) => &[0, 1, 2, 3, 5, 6],
        }
    }

    fn preview_request(&self) -> Option<PreviewRequest> {
        let repo = self.dataset.text.trim().strip_prefix("hf://")?;
        let parts: Vec<_> = repo.split('/').collect();
        if parts.len() != 2
            || parts
                .iter()
                .any(|p| p.is_empty() || p.chars().any(char::is_whitespace))
            || self.dataset_split.text.trim().is_empty()
        {
            return None;
        }
        Some(PreviewRequest {
            repo: repo.into(),
            config: optional(&self.dataset_config.text),
            split: self.dataset_split.text.trim().into(),
            revision: self.dataset_revision.clone(),
            dataset_mode: self.dataset_mode().into(),
            text_field: if self.dataset_mode == 1 {
                optional(&self.text_field.text)
            } else {
                None
            },
            prompt_field: if self.dataset_mode != 1 {
                optional(&self.prompt_field.text)
            } else {
                None
            },
            target_field: if self.dataset_mode == 0 {
                optional(&self.target_field.text)
            } else {
                None
            },
        })
    }

    fn mapping_confirmed(&self) -> bool {
        self.preview_ready()
            && self.confirmed_mapping.is_some()
            && self.confirmed_mapping == self.preview_request()
    }

    fn preview_count(&self) -> usize {
        self.preview.as_ref().map_or(0, |preview| {
            let rows = if self.mapping_confirmed() {
                preview.get("mapped_rows").unwrap_or(&preview["rows"])
            } else {
                &preview["rows"]
            };
            rows.as_array().map_or(0, Vec::len)
        })
    }

    fn invalidate_preview(&mut self) {
        let current = self.preview_request();
        if self.confirmed_mapping.is_some() && self.confirmed_mapping != current {
            self.confirmed_mapping = None;
            self.preview_row = 0;
            self.preview_scroll = 0;
        }
        if self.preview_for.is_some() && self.preview_for != current {
            let same_source =
                self.preview_for
                    .as_ref()
                    .zip(current.as_ref())
                    .is_some_and(|(old, new)| {
                        old.repo == new.repo
                            && old.config == new.config
                            && old.split == new.split
                            && old.revision == new.revision
                    });
            // Mapping edits do not change the fetched records. Keep those
            // visible, but retain the old request so readiness stays false.
            if !same_source {
                self.preview = None;
                self.preview_for = None;
                self.preview_row = 0;
                self.preview_scroll = 0;
            }
        }
        if self.loading_for.is_some() && self.loading_for != current {
            self.loading_for = None;
        }
    }

    fn request_preview(&mut self, force: bool) -> Option<Action> {
        self.invalidate_preview();
        let request = self.preview_request()?;
        if !force
            && (self.preview_for.as_ref() == Some(&request)
                || self.loading_for.as_ref() == Some(&request))
        {
            return None;
        }
        self.error.clear();
        if force {
            self.preview = None;
            self.preview_for = None;
            self.confirmed_mapping = None;
            self.preview_row = 0;
            self.preview_scroll = 0;
        }
        self.loading_for = Some(request.clone());
        Some(Action::Preview(request))
    }

    fn accept_preview(
        &mut self,
        request: PreviewRequest,
        result: std::result::Result<Value, String>,
    ) {
        if self.preview_request().as_ref() != Some(&request) {
            return;
        }
        self.loading_for = None;
        self.confirmed_mapping = None;
        match result {
            Err(error) => {
                self.error = clean(&error);
                self.preview = None;
                self.preview_for = None;
                self.confirmed_mapping = None;
            }
            Ok(value) => {
                for (input, key) in [
                    (&mut self.prompt_field, "prompt_field"),
                    (&mut self.target_field, "target_field"),
                    (&mut self.text_field, "text_field"),
                    (&mut self.dataset_config, "config"),
                ] {
                    if input.text.trim().is_empty() {
                        if let Some(text) = value[key].as_str() {
                            *input = Input::new(text);
                        }
                    }
                }
                self.error = value["error"]
                    .as_str()
                    .or(value["mapping_error"].as_str())
                    .map(clean)
                    .unwrap_or_default();
                self.preview = Some(value);
                self.preview_for = self.preview_request();
                self.preview_row = 0;
                self.preview_scroll = 0;
            }
        }
    }

    fn preview_ready(&self) -> bool {
        self.preview_for.is_some()
            && self.preview_for == self.preview_request()
            && self.preview.as_ref().is_some_and(|v| {
                v["error"].is_null()
                    && v["mapping_error"].is_null()
                    && v["revision"].as_str().is_some_and(|s| !s.is_empty())
                    && v["rows"].as_array().is_some_and(|r| !r.is_empty())
                    && v["dataset_mode"].as_str().unwrap_or("paired") == self.dataset_mode()
                    && match self.dataset_mode {
                        1 => v["text_field"].as_str() == Some(self.text_field.text.trim()),
                        2 => v["prompt_field"].as_str() == Some(self.prompt_field.text.trim()),
                        _ => {
                            v["prompt_field"].as_str() == Some(self.prompt_field.text.trim())
                                && v["target_field"].as_str() == Some(self.target_field.text.trim())
                        }
                    }
                    && v["split"].as_str() == Some(self.dataset_split.text.trim())
            })
    }

    fn remote(&self) -> bool {
        self.dataset.text.trim().starts_with("hf://")
    }

    fn token_changed(&mut self, previous: &str) {
        if previous != self.hf_token.text {
            self.preview = None;
            self.preview_for = None;
            self.confirmed_mapping = None;
            self.loading_for = None;
            self.preview_row = 0;
            self.preview_scroll = 0;
        }
    }

    fn default_editor_model(&self) -> &'static str {
        if matches!(self.editor, 1 | 3) {
            "claude-sonnet-5-5"
        } else {
            "gpt-6.1-sol"
        }
    }

    fn suggested_models(&self) -> &'static [&'static str] {
        if matches!(self.editor, 1 | 3) {
            &[
                "claude-sonnet-5-5",
                "claude-opus-5-5",
                "claude-sonnet-4-6",
                "claude-opus-4-6",
            ]
        } else {
            &["gpt-6.1-sol", "gpt-6-astra", "gpt-6.1-terra", "gpt-6-luna"]
        }
    }

    fn prefill_goal(&mut self) {
        if !self.goal.text.trim().is_empty()
            && self.generated_goal.as_deref() != Some(self.goal.text.trim())
        {
            return;
        }
        if self.dataset_mode != 0 {
            let text = if self.dataset_mode == 1 {
                "Match the style and tone of the supplied text examples."
            } else {
                match self.target {
                    0 => "Answer the supplied prompts accurately.",
                    1 => "Use a consistent style and tone across the supplied prompts.",
                    2 => "Follow the requested output format for each prompt.",
                    3 => "Assign the requested category to each prompt.",
                    _ => "Improve responses to the supplied prompts against the editing goal.",
                }
            }
            .to_owned();
            self.goal = Input::new(&text);
            self.generated_goal = Some(text);
            return;
        }
        let field = if self.target_field.text.is_empty() {
            "target"
        } else {
            self.target_field.text.trim()
        };
        let numeric = self
            .preview
            .as_ref()
            .and_then(|v| v["columns"].as_array())
            .and_then(|cols| cols.iter().find(|col| col["name"].as_str() == Some(field)))
            .and_then(|col| col["dtype"].as_str())
            .is_some_and(|dtype| {
                dtype.contains("int") || dtype.contains("float") || dtype.contains("decimal")
            });
        let reference = if field.contains(" ?? ") {
            "the selected reference answers".to_owned()
        } else if field.contains('.') || field.contains('[') {
            format!("the {field} field")
        } else {
            format!("the {field} column")
        };
        let text = format!(
            "Minimise {} errors against {reference}.",
            if numeric { "prediction" } else { "answer" }
        );
        self.goal = Input::new(&text);
        self.generated_goal = Some(text);
    }

    fn selection(&mut self) -> Option<(&mut usize, usize)> {
        match self.step {
            0 => Some((&mut self.target, TARGETS.len())),
            1 if !self.typing => Some((&mut self.model, MODELS.len())),
            4 if self.dataset_mode == 0 => Some((&mut self.metric, METRICS.len())),
            6 if !self.typing => Some((&mut self.editor, EDITORS.len())),
            8 if !self.typing => Some((&mut self.publish, 2)),
            9 if !self.typing => Some((&mut self.trajectory, 3)),
            _ => None,
        }
    }

    fn input(&mut self) -> Option<&mut Input> {
        match self.step {
            1 if self.model == 2 && self.typing => Some(&mut self.custom_model),
            2 => match self.field {
                1 => Some(&mut self.dataset_config),
                2 => Some(&mut self.dataset_split),
                3 if self.dataset_mode == 1 => Some(&mut self.text_field),
                3 => Some(&mut self.prompt_field),
                4 => Some(&mut self.target_field),
                5 => Some(&mut self.limit),
                6 => Some(&mut self.hf_token),
                _ => Some(&mut self.dataset),
            },
            3 => {
                if self.field == 1 {
                    Some(&mut self.controls)
                } else {
                    Some(&mut self.goal)
                }
            }
            5 => match self.field {
                1 => self
                    .health_items
                    .get_mut(self.health_index)
                    .map(|item| &mut item.count),
                2 => Some(&mut self.health_drop),
                _ => None,
            },
            7 => {
                if self.field == 1 {
                    Some(&mut self.editor_model)
                } else {
                    Some(&mut self.budget)
                }
            }
            8 if self.publish == 1 && self.typing => match self.field {
                1 => Some(&mut self.publish_visibility),
                2 => Some(&mut self.publish_mode),
                3 => Some(&mut self.hf_token),
                _ => Some(&mut self.publish_repo),
            },
            9 if self.trajectory == 2 && self.typing => match self.field {
                1 => Some(&mut self.trajectory_visibility),
                2 => Some(&mut self.trajectory_mode),
                3 => Some(&mut self.hf_token),
                _ => Some(&mut self.trajectory_repo),
            },
            _ => None,
        }
    }

    fn validate(&self) -> std::result::Result<(), String> {
        match self.step {
            1 if self.model == 2 && self.custom_model.text.trim().is_empty() => {
                Err("Enter a model ID or checkpoint directory.".into())
            }
            2 => {
                let source = self.dataset.text.trim();
                if source.is_empty() {
                    return Err("Enter a JSONL file or hf://organisation/dataset.".into());
                }
                if let Some(id) = source.strip_prefix("hf://") {
                    let parts: Vec<_> = id.split('/').collect();
                    if parts.len() != 2
                        || parts
                            .iter()
                            .any(|part| part.is_empty() || part.chars().any(char::is_whitespace))
                    {
                        return Err(
                            "Use hf://organisation/dataset for a Hugging Face dataset.".into()
                        );
                    }
                    if self.dataset_split.text.trim().is_empty() {
                        return Err("Provide the source split.".into());
                    }
                    if self.dataset_mode == 0
                        && !self.prompt_field.text.trim().is_empty()
                        && self.prompt_field.text.trim() == self.target_field.text.trim()
                    {
                        return Err(
                            "Choose distinct prompt and target paths within the record.".into()
                        );
                    }
                    match self.limit.text.trim().parse::<u32>() {
                        Ok(value) if (5..=100_000).contains(&value) => (),
                        _ => return Err("Examples must be a whole number from 5 to 100000.".into()),
                    }
                } else if !std::path::Path::new(&expand_home(source)).is_file() {
                    return Err(format!("File not found: {source}"));
                }
                Ok(())
            }
            3 if self.goal.text.trim().is_empty() => {
                Err("Describe the behaviour you want the model to develop.".into())
            }
            5 => {
                if !self.health_drop.text.trim().is_empty()
                    && !self.health_items.iter().any(|item| item.enabled)
                {
                    return Err("Select a health check before setting a drop limit.".into());
                }
                for item in self.health_items.iter().filter(|item| item.enabled) {
                    if !item
                        .count
                        .text
                        .trim()
                        .parse::<u32>()
                        .is_ok_and(|count| (1..=1000).contains(&count))
                    {
                        return Err(format!("{} needs 1 to 1000 examples.", item.label));
                    }
                }
                if !self.health_drop.text.trim().is_empty()
                    && !self
                        .health_drop
                        .text
                        .trim()
                        .parse::<f64>()
                        .is_ok_and(|value| value.is_finite() && (0.0..=100.0).contains(&value))
                {
                    return Err(
                        "Enter a drop from 0 to 100 percentage points, or leave it blank.".into(),
                    );
                }
                Ok(())
            }
            7 => match self.budget.text.trim().parse::<f64>() {
                Ok(value)
                    if value.is_finite()
                        && value > 0.0
                        && !self.editor_model.text.trim().is_empty() =>
                {
                    Ok(())
                }
                Ok(value) if value.is_finite() && value > 0.0 => {
                    Err("Choose a frontier model or enter its model ID.".into())
                }
                _ => Err("Enter a positive USD amount, such as 5 or 2.50.".into()),
            },
            8 if self.publish == 1 => {
                let parts: Vec<_> = self.publish_repo.text.trim().split('/').collect();
                if parts.len() != 2
                    || parts
                        .iter()
                        .any(|p| p.is_empty() || p.chars().any(char::is_whitespace))
                {
                    return Err("Enter a model repository as owner/name.".into());
                }
                if !["private", "public"].contains(&self.publish_visibility.text.trim()) {
                    return Err("Visibility must be private or public.".into());
                }
                if !["existing", "new"].contains(&self.publish_mode.text.trim()) {
                    return Err("Choose existing or new for the repository.".into());
                }
                Ok(())
            }
            9 if self.trajectory == 2 => validate_destination(
                &self.trajectory_repo,
                &self.trajectory_visibility,
                &self.trajectory_mode,
                "dataset",
            ),
            _ => Ok(()),
        }
    }

    fn result(&self) -> WizardResult {
        WizardResult {
            target_type: self
                .custom_target_type
                .as_deref()
                .unwrap_or(TARGETS[self.target].0)
                .into(),
            model: if self.model == 2 {
                self.custom_model.text.trim().into()
            } else {
                MODELS[self.model].0.into()
            },
            dataset: expand_home(self.dataset.text.trim()),
            dataset_mode: self.dataset_mode().into(),
            text_field: if self.dataset_mode == 1 {
                self.text_field.text.trim().into()
            } else {
                String::new()
            },
            goal: self.goal.text.trim().into(),
            metric: if self.dataset_mode == 0 {
                METRICS[self.metric].0.into()
            } else {
                "semantic".into()
            },
            budget: self.budget.text.trim().parse().unwrap_or(0.0),
            editor: EDITORS[self.editor].0.into(),
            editor_model: if self.editor_model.text.trim().is_empty() {
                None
            } else {
                Some(self.editor_model.text.trim().into())
            },
            prompt_field: if self.dataset_mode != 1 {
                self.prompt_field.text.trim().into()
            } else {
                String::new()
            },
            target_field: if self.dataset_mode == 0 {
                self.target_field.text.trim().into()
            } else {
                String::new()
            },
            dataset_split: self.dataset_split.text.trim().into(),
            dataset_config: optional(&self.dataset_config.text),
            dataset_revision: if self.remote() {
                self.preview
                    .as_ref()
                    .and_then(|v| v["revision"].as_str())
                    .map(str::to_owned)
                    .or(self.dataset_revision.clone())
            } else {
                None
            },
            limit: self.limit.text.trim().parse().unwrap_or(100),
            controls: self.controls.text.trim().into(),
            hf_token: optional(&self.hf_token.text),
            publish_repo: if self.publish == 1 {
                optional(&self.publish_repo.text)
            } else {
                None
            },
            publish_visibility: self.publish_visibility.text.trim().into(),
            publish_create: self.publish_mode.text.trim() == "new",
            health_checks: self
                .health_items
                .iter()
                .filter(|item| item.enabled)
                .map(|item| crate::health::Selection {
                    check: item.id.clone(),
                    examples: item.count.text.trim().parse().unwrap_or(0),
                })
                .collect(),
            health_max_drop: self
                .health_drop
                .text
                .trim()
                .parse::<f64>()
                .ok()
                .map(|value| value / 100.0),
            save_trajectory: self.trajectory != 0,
            trajectory_repo: if self.trajectory == 2 {
                optional(&self.trajectory_repo.text)
            } else {
                None
            },
            trajectory_visibility: self.trajectory_visibility.text.trim().into(),
            trajectory_create: self.trajectory_mode.text.trim() == "new",
        }
    }

    fn key(&mut self, key: KeyEvent) -> Action {
        if key.kind == KeyEventKind::Release {
            return Action::Continue;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Action::Cancel;
        }
        self.error.clear();
        let previous_token = self.hf_token.text.clone();
        let previous_editor = self.editor;
        let previous_editor_default = self.default_editor_model();
        match key.code {
            KeyCode::Char(' ') if self.step == 5 && self.field == 0 => {
                if let Some(item) = self.health_items.get_mut(self.health_index) {
                    item.enabled = !item.enabled;
                }
            }
            KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown
                if self.step == 5 && self.field == 0 =>
            {
                let count = self.health_items.len();
                let distance = if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) {
                    8
                } else {
                    1
                };
                if count > 0 {
                    self.health_index = if matches!(key.code, KeyCode::Up | KeyCode::PageUp) {
                        self.health_index.saturating_sub(distance)
                    } else {
                        (self.health_index + distance).min(count - 1)
                    };
                }
            }
            KeyCode::F(2) if self.step == 2 => {
                self.dataset_mode = (self.dataset_mode + 1) % DATA_MODES.len();
                self.metric = if self.dataset_mode == 0 { 0 } else { 3 };
                if self.field == 4 && self.dataset_mode != 0 {
                    self.field = 3;
                }
                self.confirmed_mapping = None;
                self.preview_row = 0;
                self.preview_scroll = 0;
                if let Some(action) = self.request_preview(false) {
                    return action;
                }
            }
            KeyCode::Char('p')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(self.step, 2 | 8 | 9)
                    && self.remote() =>
            {
                if let Some(action) = self.request_preview(true) {
                    return action;
                }
                self.error = "Enter a valid Hugging Face dataset and source split.".into();
            }
            KeyCode::PageUp | KeyCode::PageDown if self.step == 2 && self.remote() => {
                let n = self.preview_count();
                if n > 0 {
                    self.preview_row = if key.code == KeyCode::PageUp {
                        (self.preview_row + n - 1) % n
                    } else {
                        (self.preview_row + 1) % n
                    };
                    self.preview_scroll = 0;
                }
            }
            KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown
                if self.step == 10 =>
            {
                self.review_scroll = if matches!(key.code, KeyCode::Up | KeyCode::PageUp) {
                    self.review_scroll.saturating_sub(1)
                } else {
                    self.review_scroll.saturating_add(1).min(2000)
                };
            }
            KeyCode::Up | KeyCode::Down
                if self.step == 2 && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.preview_scroll = if key.code == KeyCode::Up {
                    self.preview_scroll.saturating_sub(1)
                } else {
                    self.preview_scroll.saturating_add(1).min(2000)
                };
            }
            KeyCode::Esc => {
                if self.typing && matches!(self.step, 1 | 6 | 8 | 9) {
                    self.typing = false;
                } else if self.step == 0 {
                    return Action::Cancel;
                } else {
                    self.step -= 1;
                    self.field = 0;
                    self.typing = false;
                }
            }
            KeyCode::Up | KeyCode::Down if self.selection().is_some() => {
                let (selected, count) = self.selection().unwrap();
                *selected = if key.code == KeyCode::Up {
                    (*selected + count - 1) % count
                } else {
                    (*selected + 1) % count
                };
                if self.step == 0 {
                    self.custom_target_type = None;
                }
            }
            KeyCode::Up | KeyCode::Down if self.step == 7 && self.field == 1 => {
                let choices = self.suggested_models();
                let current = choices
                    .iter()
                    .position(|m| *m == self.editor_model.text)
                    .unwrap_or(0);
                let next = if key.code == KeyCode::Up {
                    (current + choices.len() - 1) % choices.len()
                } else {
                    (current + 1) % choices.len()
                };
                self.editor_model = Input::new(choices[next]);
            }
            KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right
                if matches!(self.step, 8 | 9) && self.typing && matches!(self.field, 1 | 2) =>
            {
                let value = if self.step == 9 {
                    if self.field == 1 {
                        &mut self.trajectory_visibility
                    } else {
                        &mut self.trajectory_mode
                    }
                } else if self.field == 1 {
                    &mut self.publish_visibility
                } else {
                    &mut self.publish_mode
                };
                *value = Input::new(match value.text.as_str() {
                    "private" => "public",
                    "public" => "private",
                    "existing" => "new",
                    _ => "existing",
                });
            }
            KeyCode::Tab | KeyCode::BackTab => {
                if self.step == 2 {
                    let fields = self.dataset_field_indices();
                    let index = fields
                        .iter()
                        .position(|field| *field == self.field)
                        .unwrap_or(0);
                    self.field = fields[if key.code == KeyCode::BackTab {
                        (index + fields.len() - 1) % fields.len()
                    } else {
                        (index + 1) % fields.len()
                    }];
                    if let Some(action) = self.request_preview(false) {
                        return action;
                    }
                } else if self.step == 1 && self.model == 2 {
                    self.typing = !self.typing;
                } else if self.step == 5 {
                    self.field = if key.code == KeyCode::BackTab {
                        (self.field + 2) % 3
                    } else {
                        (self.field + 1) % 3
                    };
                } else if matches!(self.step, 3 | 7) {
                    self.field = (self.field + 1) % 2;
                } else if (self.step == 8 && self.publish == 1)
                    || (self.step == 9 && self.trajectory == 2)
                {
                    self.typing = true;
                    self.field = if key.code == KeyCode::BackTab {
                        (self.field + 3) % 4
                    } else {
                        (self.field + 1) % 4
                    };
                    if self.remote() {
                        if let Some(action) = self.request_preview(false) {
                            return action;
                        }
                    }
                }
            }
            KeyCode::Enter => {
                if (self.step == 1 && self.model == 2
                    || self.step == 8 && self.publish == 1
                    || self.step == 9 && self.trajectory == 2)
                    && !self.typing
                {
                    self.typing = true;
                } else if let Err(error) = self.validate() {
                    self.error = error;
                } else if matches!(self.step, 2 | 8 | 9) && self.remote() && !self.preview_ready() {
                    if let Some(action) = self.request_preview(false) {
                        return action;
                    }
                    if self.loading_for.is_some() {
                        self.error = "Fetching preview. You can keep editing.".into();
                    } else {
                        self.error =
                            "Correct the dataset fields, then press Ctrl-P to preview.".into();
                    }
                } else if self.step == 2 && self.remote() && !self.mapping_confirmed() {
                    let record = self.preview_row;
                    self.confirmed_mapping = self.preview_request();
                    self.preview_row = self
                        .preview
                        .as_ref()
                        .and_then(|preview| preview["mapped_rows"].as_array())
                        .and_then(|pairs| {
                            pairs
                                .iter()
                                .position(|pair| pair["row_index"].as_u64() == Some(record as u64))
                        })
                        .unwrap_or_else(|| record.min(self.preview_count().saturating_sub(1)));
                    self.preview_scroll = 0;
                } else if self.step == 10 {
                    return Action::Start(Box::new(self.result()));
                } else {
                    if self.step == 2 {
                        self.prefill_goal();
                    }
                    if self.step == 6 && self.editor_model.text.trim().is_empty() {
                        self.editor_model = Input::new(self.default_editor_model());
                    }
                    self.step += 1;
                    self.field = 0;
                    self.typing = false;
                }
            }
            _ => {
                if let Some(input) = self.input() {
                    input.key(key);
                }
            }
        }
        if self.step == 2 && !self.dataset_field_indices().contains(&self.field) {
            self.field = 0;
        }
        if self.editor != previous_editor && self.editor_model.text == previous_editor_default {
            self.editor_model = Input::new(self.default_editor_model());
        }
        self.invalidate_preview();
        self.token_changed(&previous_token);
        if !matches!(self.step, 2 | 8 | 9) {
            self.loading_for = None;
        }
        Action::Continue
    }
}

fn validate_destination(
    repo: &Input,
    visibility: &Input,
    mode: &Input,
    kind: &str,
) -> std::result::Result<(), String> {
    let parts: Vec<_> = repo.text.trim().split('/').collect();
    if parts.len() != 2
        || parts
            .iter()
            .any(|part| part.is_empty() || part.chars().any(char::is_whitespace))
    {
        return Err(format!("Enter a {kind} repository as owner/name."));
    }
    if !["private", "public"].contains(&visibility.text.trim()) {
        return Err("Visibility must be private or public.".into());
    }
    if !["existing", "new"].contains(&mode.text.trim()) {
        return Err("Choose existing or new for the repository.".into());
    }
    Ok(())
}

fn optional(text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.into())
    }
}
fn clean(text: &str) -> String {
    let mut input = Input::default();
    input.insert(text);
    input.text
}

fn expand_home(text: &str) -> String {
    if let Some(suffix) = text.strip_prefix("~/") {
        let home = crate::platform::home_dir();
        if !home.as_os_str().is_empty() {
            return home.join(suffix).to_string_lossy().into_owned();
        }
    }
    text.into()
}

struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            DisableBracketedPaste,
            LeaveAlternateScreen,
            Show
        );
    }
}

pub fn wizard(options: WizardOptions) -> Result<Option<WizardResult>> {
    enable_raw_mode()?;
    let _guard = TerminalGuard;
    execute!(io::stdout(), EnterAlternateScreen, EnableBracketedPaste)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;
    let mut form = Form::from_options(&options);
    let mut preview = PreviewClient::new(options.python);
    let mut redraw = true;
    loop {
        if let Some(result) = preview.poll() {
            if let Some(request) = form.loading_for.take() {
                form.accept_preview(request, result);
                redraw = true;
            }
        }
        if redraw {
            terminal.draw(|frame| render(frame, &form))?;
            redraw = false;
        }
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) => match form.key(key) {
                Action::Continue => redraw = true,
                Action::Cancel => return Ok(None),
                Action::Start(result) => return Ok(Some(*result)),
                Action::Preview(request) => {
                    preview.set_token(optional(&form.hf_token.text));
                    if let Err(error) = preview.start(request) {
                        form.loading_for = None;
                        form.error = clean(&error.to_string());
                    }
                    redraw = true;
                }
            },
            Event::Paste(text) => {
                let previous_token = form.hf_token.text.clone();
                if let Some(input) = form.input() {
                    input.insert(&text);
                    form.invalidate_preview();
                    form.token_changed(&previous_token);
                    redraw = true;
                }
            }
            Event::Resize(_, _) => redraw = true,
            _ => (),
        }
        if form.loading_for.is_none() {
            preview.cancel();
        }
    }
}

fn render(frame: &mut Frame, form: &Form) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(INK).fg(IVORY)),
        area,
    );
    if area.width < 36 || area.height < 14 {
        frame.render_widget(
            Paragraph::new("VESPASIAN\nEnlarge the terminal to continue.\nCtrl-C cancel")
                .style(Style::default().fg(COPPER)),
            area,
        );
        return;
    }
    let wide = area.width >= 88;
    let margin = if wide { 3 } else { 1 };
    let body = Rect::new(
        area.x + margin,
        area.y + 1,
        area.width.saturating_sub(margin * 2),
        area.height.saturating_sub(2),
    );
    let rows = Layout::vertical([
        Constraint::Length(if area.height < 24 { 3 } else { 4 }),
        Constraint::Min(1),
        Constraint::Length(3),
    ])
    .split(body);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "VESPASIAN",
                    Style::default().fg(COPPER).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("   /   v{}", env!("CARGO_PKG_VERSION")),
                    Style::default().fg(MUTED),
                ),
            ]),
            Line::styled(
                "github:chrisvoncsefalvay/vespasian",
                Style::default().fg(MUTED),
            ),
        ])
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(RULE)),
        ),
        rows[0],
    );
    let content = if wide {
        let cols = Layout::horizontal([
            Constraint::Length(22),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(rows[1]);
        render_steps(frame, cols[0], form);
        cols[2]
    } else {
        rows[1]
    };
    render_step(frame, content, form);
    let hint = if form.step == 10 {
        "Enter start run · ↑ ↓ scroll review"
    } else if form.selection_is_active() {
        "↑ ↓ choose  ·  Enter continue"
    } else if form.step == 2 && form.remote() {
        if form.mapping_confirmed() {
            "Tab fields · Enter continue · Ctrl-P refresh"
        } else if form.preview_ready() {
            "Tab fields · Enter confirm mapping · Ctrl-P refresh"
        } else {
            "Tab fields · Enter preview · Ctrl-P refresh"
        }
    } else if form.step == 5 {
        "Space select · Tab count/limit · Enter continue"
    } else if form.step == 7 {
        "Tab fields · ↑ ↓ models · type ID · Enter continue"
    } else if matches!(form.step, 8 | 9) && form.typing {
        "Tab fields · ↑ ↓ toggle choices · Enter continue"
    } else if matches!(form.step, 2 | 3) {
        "Tab fields · Enter continue · ← → move cursor"
    } else {
        "Enter continue  ·  ← → move cursor"
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(form.error.as_str(), Style::default().fg(RED)),
            Line::from(vec![Span::styled(hint, Style::default().fg(COPPER))]),
            Line::styled(
                if form.step == 5 {
                    "↑ ↓ checks · PgUp/PgDn page · Esc back · Ctrl-C"
                } else if form.step == 2 && form.remote() {
                    if area.width < 80 {
                        "PgUp/PgDn sample · Ctrl-↑/↓ scroll · Esc back · Ctrl-C"
                    } else {
                        "PgUp/PgDn samples · Ctrl-↑/↓ scroll · Esc back · Ctrl-C cancel"
                    }
                } else {
                    "Esc back · Ctrl-C cancel"
                },
                Style::default().fg(MUTED),
            ),
        ])
        .wrap(Wrap { trim: true }),
        rows[2],
    );
}

impl Form {
    fn selection_is_active(&self) -> bool {
        matches!(self.step, 0 | 6)
            || (self.step == 4 && self.dataset_mode == 0)
            || matches!(self.step, 1 | 8 | 9) && !self.typing
    }
}

fn render_steps(frame: &mut Frame, area: Rect, form: &Form) {
    let mut lines = vec![Line::from("")];
    for (index, step) in STEPS.iter().enumerate() {
        let marker = if index < form.step {
            "✓"
        } else if index == form.step {
            "›"
        } else {
            "·"
        };
        let colour = if index == form.step {
            COPPER
        } else if index < form.step {
            GREEN
        } else {
            MUTED
        };
        lines.push(Line::styled(
            format!("{marker}  {:02}  {step}", index + 1),
            Style::default().fg(colour),
        ));
        if area.height as usize >= STEPS.len() * 2 {
            lines.push(Line::from(""));
        }
    }
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::RIGHT)
                .border_style(Style::default().fg(RULE)),
        ),
        area,
    );
}

fn render_step(frame: &mut Frame, area: Rect, form: &Form) {
    let (title, subtitle) = match form.step {
        0 => (
            "What should change?",
            "Choose the kind of behaviour you want to develop.",
        ),
        1 => (
            "Choose the model",
            "Select the checkpoint whose weights will be edited.",
        ),
        2 => (
            DATA_MODES[form.dataset_mode].1,
            "Use a local JSONL file or a Hugging Face dataset.",
        ),
        3 => (
            "Set the editing goal",
            "State the desired behaviour and what should stay intact.",
        ),
        4 if form.dataset_mode != 0 => (
            "Score the behaviour",
            "Your goal defines the rubric used to compare generated text.",
        ),
        4 => (
            "Choose the measure",
            "This score guides which candidate changes are retained.",
        ),
        5 => (
            "Choose health checks",
            "Track capabilities on a fixed set of examples.",
        ),
        6 => (
            "Choose your editor",
            "A frontier model reflects on results and proposes changes.",
        ),
        7 => (
            "Choose the model and budget",
            "Set the frontier model and its inference budget, in USD.",
        ),
        8 => (
            "Choose where to save",
            "Keep the checkpoint locally or upload it after verification.",
        ),
        9 => (
            "Save the editor trajectory",
            "Keep the editor exchanges, decisions and measurements.",
        ),
        _ => (
            "Review the run",
            "Enter starts dataset assessment with these settings.",
        ),
    };
    let heading_height = if area.height < 14 || form.step == 2 {
        1
    } else {
        3
    };
    let rows =
        Layout::vertical([Constraint::Length(heading_height), Constraint::Min(0)]).split(area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("{:02} / {}  ", form.step + 1, STEPS.len()),
                    Style::default().fg(MUTED),
                ),
                Span::styled(
                    title,
                    Style::default().fg(IVORY).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    if form.step == 2 {
                        " · F2 data kind"
                    } else {
                        ""
                    },
                    Style::default().fg(COPPER),
                ),
            ]),
            Line::from(""),
            Line::styled(subtitle, Style::default().fg(MUTED)),
        ])
        .wrap(Wrap { trim: true }),
        rows[0],
    );
    let area = rows[1];
    match form.step {
        0 => render_options(frame, area, &TARGETS, form.target, true),
        1 => {
            let height = if form.typing && area.height < 10 {
                1
            } else {
                options_height(area, MODELS.len())
            };
            let sections = Layout::vertical([
                Constraint::Length(height),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);
            if form.typing && area.height < 10 {
                frame.render_widget(
                    Paragraph::new(MODELS[form.model].1).style(Style::default().fg(MUTED)),
                    sections[0],
                );
            } else {
                render_options(frame, sections[0], &MODELS, form.model, !form.typing);
            }
            if form.model == 2 {
                render_input(
                    frame,
                    sections[2],
                    "Model ID or local directory",
                    &form.custom_model,
                    form.typing,
                    "organisation/model or /path/checkpoint",
                );
            }
        }
        2 => render_dataset(frame, area, form),
        3 => {
            let sections =
                Layout::vertical([Constraint::Percentage(50), Constraint::Min(0)]).split(area);
            render_input(
                frame,
                sections[0],
                "Objective",
                &form.goal,
                form.field == 0,
                "Describe what a successful response should do.",
            );
            render_input(
                frame,
                sections[1],
                "Keep intact",
                &form.controls,
                form.field == 1,
                "Behaviour the edited model should preserve.",
            );
        }
        4 if form.dataset_mode == 0 => render_options(frame, area, &METRICS, form.metric, true),
        4 => {
            let description = if form.dataset_mode == 1 {
                "The editor scores generated text against your goal and the style of the corpus. Behaviours to preserve remain separate controls."
            } else {
                "The editor scores each generated response against your goal. No reference answer is required. Behaviours to preserve remain separate controls."
            };
            frame.render_widget(
                Paragraph::new(vec![
                    Line::styled(
                        "Goal score",
                        Style::default().fg(COPPER).add_modifier(Modifier::BOLD),
                    ),
                    Line::raw(""),
                    Line::raw(description),
                ])
                .wrap(Wrap { trim: true }),
                area,
            );
        }
        5 => render_health_setup(frame, area, form),
        6 => render_options(frame, area, &EDITORS, form.editor, true),
        7 => {
            let sections = Layout::vertical([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(0),
            ])
            .split(area);
            render_input(
                frame,
                sections[0],
                "Budget · USD",
                &form.budget,
                form.field == 0,
                "5.00",
            );
            render_input(
                frame,
                sections[1],
                "Frontier model",
                &form.editor_model,
                form.field == 1,
                form.default_editor_model(),
            );
            let explanation = if form.editor < 2 {
                "Estimated cost is checked between calls."
            } else {
                "Requests reserve budget before inference."
            };
            frame.render_widget(
                Paragraph::new(format!("The budget covers assessment, proposals, goal scoring and control checks.\n{explanation}"))
                    .style(Style::default().fg(MUTED))
                    .wrap(Wrap { trim: true }),
                sections[2],
            );
        }
        8 => render_publish(frame, area, form),
        9 => render_trajectory_setup(frame, area, form),
        _ => render_review(frame, area, form),
    }
}

fn dataset_fields(form: &Form) -> [(&str, &Input, &str); 7] {
    [
        ("Dataset", &form.dataset, "hf://organisation/dataset"),
        ("Configuration", &form.dataset_config, "auto"),
        ("Source split", &form.dataset_split, "train"),
        if form.dataset_mode == 1 {
            ("Text field", &form.text_field, "auto-detect")
        } else {
            ("Prompt field", &form.prompt_field, "auto-detect")
        },
        ("Target field", &form.target_field, "auto-detect"),
        ("Example limit", &form.limit, "100"),
        (
            "HF token · optional",
            &form.hf_token,
            "Public data or HF login",
        ),
    ]
}

fn render_dataset(frame: &mut Frame, area: Rect, form: &Form) {
    if !form.remote() {
        let fields = dataset_fields(form);
        let indices = form.dataset_field_indices();
        let all_visible = area.height >= indices.len() as u16 * 3 + 2;
        let shown: &[usize] = if all_visible {
            indices
        } else {
            std::slice::from_ref(&form.field)
        };
        let mut constraints = vec![Constraint::Length(3); shown.len()];
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Min(0));
        let sections = Layout::vertical(constraints).split(area);
        for (position, &index) in shown.iter().enumerate() {
            let (label, input, placeholder) = fields[index];
            if index == 6 {
                render_secret_input(frame, sections[position], input, form.field == index);
            } else {
                render_input(
                    frame,
                    sections[position],
                    label,
                    input,
                    form.field == index,
                    if index == 0 {
                        "dataset.jsonl or hf://organisation/dataset"
                    } else {
                        placeholder
                    },
                );
            }
        }
        let description = match form.dataset_mode {
            1 => "Local JSONL contains text examples. Choose their Text field, or leave it empty for detection. The editor scores generated text against their style and your goal.",
            2 => "Local JSONL contains prompts. Choose their Prompt field, or leave it empty for detection. The editor scores generated responses against your goal.",
            _ => "Use hf://organisation/dataset to browse Hub records. Local JSONL uses its supplied prompt–target pairs and splits.",
        };
        frame.render_widget(
            Paragraph::new(description)
                .style(Style::default().fg(MUTED))
                .wrap(Wrap { trim: true }),
            sections[shown.len() + 1],
        );
        return;
    }
    let fields = dataset_fields(form);
    if area.width >= 66 && area.height >= form.dataset_field_indices().len() as u16 * 3 {
        let cols = Layout::horizontal([
            Constraint::Percentage(42),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(area);
        let indices = form.dataset_field_indices();
        let rows = Layout::vertical(vec![Constraint::Length(3); indices.len()]).split(cols[0]);
        for (position, &index) in indices.iter().enumerate() {
            let (label, input, placeholder) = fields[index];
            if index == 6 {
                render_secret_input(frame, rows[position], input, form.field == index);
            } else {
                render_input(
                    frame,
                    rows[position],
                    label,
                    input,
                    form.field == index,
                    placeholder,
                );
            }
        }
        render_preview(frame, cols[2], form);
    } else {
        let sections = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);
        let (label, input, placeholder) = fields[form.field];
        if form.field == 6 {
            render_secret_input(frame, sections[0], input, true);
        } else {
            render_input(frame, sections[0], label, input, true, placeholder);
        }
        let selection = match form.dataset_mode {
            1 => format!(
                "Text {}",
                if form.text_field.text.is_empty() {
                    "auto"
                } else {
                    &form.text_field.text
                }
            ),
            2 => format!(
                "Prompt {}",
                if form.prompt_field.text.is_empty() {
                    "auto"
                } else {
                    &form.prompt_field.text
                }
            ),
            _ => format!(
                "{} → {}",
                if form.prompt_field.text.is_empty() {
                    "auto"
                } else {
                    &form.prompt_field.text
                },
                if form.target_field.text.is_empty() {
                    "auto"
                } else {
                    &form.target_field.text
                }
            ),
        };
        let indices = form.dataset_field_indices();
        let position = indices
            .iter()
            .position(|index| *index == form.field)
            .unwrap_or(0)
            + 1;
        frame.render_widget(
            Paragraph::new(format!(
                "Field {position}/{} · {selection} · limit {}",
                indices.len(),
                form.limit.text
            ))
            .style(Style::default().fg(MUTED)),
            sections[1],
        );
        render_preview(frame, sections[2], form);
    }
}

fn render_health_setup(frame: &mut Frame, area: Rect, form: &Form) {
    let selected = form.health_items.iter().filter(|item| item.enabled).count();
    let total: u32 = form
        .health_items
        .iter()
        .filter(|item| item.enabled)
        .filter_map(|item| item.count.text.parse::<u32>().ok())
        .sum();
    let sections = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
    frame.render_widget(
        Paragraph::new(format!(
            "{selected} checks selected · {total} examples per evaluation"
        ))
        .style(Style::default().fg(MUTED)),
        sections[0],
    );
    let wide = area.width >= 62 && area.height >= 16;
    let (list_area, settings_area) = if wide {
        let columns = Layout::horizontal([
            Constraint::Percentage(57),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(sections[1]);
        (columns[0], columns[2])
    } else {
        let rows = Layout::vertical([
            Constraint::Min(2),
            Constraint::Length(if area.height >= 15 { 7 } else { 4 }),
        ])
        .split(sections[1]);
        (rows[0], rows[1])
    };
    let visible = list_area.height.saturating_sub(2).max(1) as usize;
    let first = form
        .health_index
        .saturating_sub(visible / 2)
        .min(form.health_items.len().saturating_sub(visible));
    let lines: Vec<Line> = form
        .health_items
        .iter()
        .enumerate()
        .skip(first)
        .take(visible)
        .map(|(index, item)| {
            let active = form.field == 0 && index == form.health_index;
            let label = item.label.clone();
            Line::from(vec![
                Span::styled(
                    if index == form.health_index {
                        "› "
                    } else {
                        "  "
                    },
                    Style::default().fg(COPPER),
                ),
                Span::styled(
                    if item.enabled { "☑ " } else { "☐ " },
                    Style::default().fg(if item.enabled { GREEN } else { MUTED }),
                ),
                Span::styled(
                    label,
                    Style::default().fg(if active { COPPER } else { IVORY }),
                ),
                Span::styled(
                    if item.enabled {
                        format!("  {}", item.count.text)
                    } else {
                        String::new()
                    },
                    Style::default().fg(MUTED),
                ),
            ])
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if form.field == 0 { COPPER } else { RULE }))
        .title(Line::styled(
            format!(
                " Checks · {}/{} ",
                form.health_index + 1,
                form.health_items.len()
            ),
            Style::default().fg(MUTED),
        ));
    frame.render_widget(Paragraph::new(lines).block(block), list_area);
    let Some(item) = form.health_items.get(form.health_index) else {
        return;
    };
    if wide || settings_area.height >= 7 {
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(settings_area);
        render_input(
            frame,
            rows[0],
            "Examples for this check",
            &item.count,
            form.field == 1,
            "20",
        );
        render_input(
            frame,
            rows[1],
            "Maximum drop · pp",
            &form.health_drop,
            form.field == 2,
            "Blank = monitor only",
        );
        frame.render_widget(Paragraph::new(format!("{}\n{}\n\nScores are measured on the same examples each turn. A drop limit rejects regressions from baseline.\n\nLocal model execution only.", item.label, item.group))
            .style(Style::default().fg(MUTED)).wrap(Wrap { trim: true }), rows[2]);
    } else {
        let rows =
            Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(settings_area);
        if form.field == 2 {
            render_input(
                frame,
                rows[0],
                "Maximum drop · percentage points",
                &form.health_drop,
                true,
                "Blank = monitor only",
            );
        } else {
            render_input(
                frame,
                rows[0],
                "Examples for selected check",
                &item.count,
                form.field == 1,
                "20",
            );
        }
        frame.render_widget(
            Paragraph::new(if form.health_drop.text.is_empty() {
                "Monitor only · no frontier inference"
            } else {
                "Reject drops beyond the chosen limit"
            })
            .style(Style::default().fg(MUTED)),
            rows[1],
        );
    }
}

fn render_trajectory_setup(frame: &mut Frame, area: Rect, form: &Form) {
    if !form.typing {
        render_options(frame, area, &TRAJECTORIES, form.trajectory, true);
        return;
    }
    let fields = [
        (
            "Dataset repository",
            &form.trajectory_repo,
            "owner/trajectories",
        ),
        (
            "Visibility",
            &form.trajectory_visibility,
            "private or public",
        ),
        ("Repository", &form.trajectory_mode, "existing or new"),
        (
            "HF token · optional",
            &form.hf_token,
            "Public data or existing HF login",
        ),
    ];
    if area.height >= 13 {
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);
        for (index, (label, input, placeholder)) in fields.iter().enumerate() {
            if index == 3 {
                render_secret_input(frame, rows[index], input, form.field == index);
            } else {
                render_input(
                    frame,
                    rows[index],
                    label,
                    input,
                    form.field == index,
                    placeholder,
                );
            }
        }
        frame.render_widget(
            Paragraph::new(
                "Uploads contain editor prompts and responses, including supplied examples.",
            )
            .style(Style::default().fg(MUTED))
            .wrap(Wrap { trim: true }),
            rows[4],
        );
    } else {
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);
        let (label, input, placeholder) = fields[form.field];
        if form.field == 3 {
            render_secret_input(frame, rows[0], input, true);
        } else {
            render_input(frame, rows[0], label, input, true, placeholder);
        }
        frame.render_widget(
            Paragraph::new(format!(
                "Field {}/4 · {} · {}",
                form.field + 1,
                form.trajectory_visibility.text,
                form.trajectory_mode.text
            ))
            .style(Style::default().fg(MUTED)),
            rows[1],
        );
        frame.render_widget(
            Paragraph::new(
                "Save JSONL locally and upload editor prompts, responses and measured outcomes.",
            )
            .style(Style::default().fg(MUTED))
            .wrap(Wrap { trim: true }),
            rows[2],
        );
    }
}

fn render_publish(frame: &mut Frame, area: Rect, form: &Form) {
    if !form.typing {
        render_options(frame, area, &PUBLISH, form.publish, true);
        return;
    }
    let fields = [
        ("Model repository", &form.publish_repo, "owner/name"),
        ("Visibility", &form.publish_visibility, "private or public"),
        ("Repository", &form.publish_mode, "existing or new"),
        (
            "HF token · optional",
            &form.hf_token,
            "Public data or HF login",
        ),
    ];
    if area.height >= 13 {
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);
        for (index, (label, input, placeholder)) in fields.iter().enumerate() {
            if index == 3 {
                render_secret_input(frame, rows[index], input, form.field == index);
            } else {
                render_input(
                    frame,
                    rows[index],
                    label,
                    input,
                    form.field == index,
                    placeholder,
                );
            }
        }
        frame.render_widget(
            Paragraph::new(if form.loading_for.is_some() {
                "Checking dataset access with the updated token…"
            } else {
                "The verified checkpoint will be uploaded after the run."
            })
            .style(Style::default().fg(MUTED))
            .wrap(Wrap { trim: true }),
            rows[4],
        );
    } else {
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);
        let (label, input, placeholder) = fields[form.field];
        if form.field == 3 {
            render_secret_input(frame, rows[0], input, true);
        } else {
            render_input(frame, rows[0], label, input, true, placeholder);
        }
        frame.render_widget(
            Paragraph::new(format!(
                "Field {}/4 · {} · {} repository",
                form.field + 1,
                form.publish_visibility.text,
                form.publish_mode.text
            ))
            .style(Style::default().fg(MUTED)),
            rows[1],
        );
        frame.render_widget(
            Paragraph::new(format!(
                "Upload to {}\n{}",
                if form.publish_repo.text.is_empty() {
                    "owner/name"
                } else {
                    &form.publish_repo.text
                },
                if form.loading_for.is_some() {
                    "Checking dataset access…"
                } else {
                    "The verified checkpoint is uploaded after the run."
                }
            ))
            .style(Style::default().fg(MUTED))
            .wrap(Wrap { trim: true }),
            rows[2],
        );
    }
}

fn render_secret_input(frame: &mut Frame, area: Rect, input: &Input, active: bool) {
    let masked = Input::new(if input.text.is_empty() {
        ""
    } else {
        "••••••••"
    });
    render_input(
        frame,
        area,
        "HF token · optional",
        &masked,
        active,
        "Public data or HF login",
    );
}

fn mapped_text(row: &Value, field: &str) -> Option<String> {
    if field.is_empty() {
        return None;
    }
    let mut value = row;
    for key in field.split('.') {
        value = value.get(key)?;
    }
    Some(preview_text(
        value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string())
            .as_str(),
    ))
}

fn choice_text(value: &Value, key: &str) -> String {
    value[key]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(Value::as_str)
                .map(clean)
                .collect::<Vec<_>>()
                .join(" · ")
        })
        .unwrap_or_default()
}

fn preview_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .split('\n')
        .map(clean)
        .collect::<Vec<_>>()
        .join("\n")
}

fn value_lines(value: &Value) -> Vec<Line<'static>> {
    let text = value.as_str().map(str::to_owned).unwrap_or_else(|| {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    });
    preview_text(&text)
        .split('\n')
        .map(|line| Line::raw(line.to_owned()))
        .collect()
}

fn record_lines(preview: &Value, row: &Value, roomy: bool) -> Vec<Line<'static>> {
    let Some(fields) = row.as_object() else {
        return value_lines(row);
    };
    let mut names: Vec<&str> = preview["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|column| column["name"].as_str())
        .filter(|name| fields.contains_key(*name))
        .collect();
    for name in fields.keys() {
        if !names.contains(&name.as_str()) {
            names.push(name);
        }
    }
    let mut lines = Vec::new();
    for name in names {
        if roomy && !lines.is_empty() {
            lines.push(Line::raw(""));
        }
        lines.push(Line::styled(
            clean(name),
            Style::default().fg(COPPER).add_modifier(Modifier::BOLD),
        ));
        lines.extend(value_lines(&fields[name]));
    }
    lines
}

fn mapped_pair(preview: &Value, form: &Form) -> Option<(String, String)> {
    if let Some(pairs) = preview["mapped_rows"].as_array() {
        let pair = pairs.get(form.preview_row)?;
        return Some((
            preview_text(pair["prompt"].as_str()?),
            preview_text(pair["target"].as_str()?),
        ));
    }
    let row = preview["rows"].as_array()?.get(form.preview_row)?;
    match form.dataset_mode {
        1 => Some((String::new(), mapped_text(row, &form.text_field.text)?)),
        2 => Some((mapped_text(row, &form.prompt_field.text)?, String::new())),
        _ => Some((
            mapped_text(row, &form.prompt_field.text)?,
            mapped_text(row, &form.target_field.text)?,
        )),
    }
}

fn render_preview(frame: &mut Frame, area: Rect, form: &Form) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let count = form.preview_count();
    let confirmed = form.mapping_confirmed();
    let title = if form.loading_for.is_some() {
        " Fetching from Hugging Face… ".into()
    } else if count > 0 {
        format!(
            " Live preview · {} {}/{} ",
            if confirmed {
                if form.dataset_mode == 0 {
                    "pair"
                } else {
                    "example"
                }
            } else {
                "record"
            },
            form.preview_row + 1,
            count
        )
    } else {
        " Hugging Face preview ".into()
    };
    let block = Block::default()
        .title(title)
        .borders(if area.height < 8 {
            Borders::TOP
        } else {
            Borders::ALL
        })
        .border_style(Style::default().fg(if confirmed { GREEN } else { RULE }));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let Some(preview) = form.preview.as_ref() else {
        let text = if form.loading_for.is_some() {
            "Reading the dataset revision, fields and first records…"
        } else {
            "Enter the dataset, then press Enter to preview.\nLeave field names empty for suggestions."
        };
        frame.render_widget(
            Paragraph::new(text)
                .style(Style::default().fg(MUTED))
                .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    };
    let roomy = inner.height >= 9;
    let mut lines = Vec::new();
    if roomy {
        let revision: String = preview["revision"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(8)
            .collect();
        let shortened = if preview["truncated"].as_bool() == Some(true) {
            " · shortened"
        } else {
            ""
        };
        lines.push(Line::styled(
            format!(
                "{} / {} · {}{}",
                clean(preview["config"].as_str().unwrap_or("auto")),
                clean(preview["split"].as_str().unwrap_or("")),
                clean(&revision),
                shortened
            ),
            Style::default().fg(MUTED),
        ));
        lines.push(Line::raw(""));
    }
    if confirmed {
        if let Some((prompt, target)) = mapped_pair(preview, form) {
            let pair = preview["mapped_rows"]
                .as_array()
                .and_then(|rows| rows.get(form.preview_row));
            let target_source = pair
                .and_then(|row| row["target_source"].as_str())
                .unwrap_or(&form.target_field.text);
            let sections = match form.dataset_mode {
                1 => vec![("Text", form.text_field.text.as_str(), target)],
                2 => vec![("Prompt", form.prompt_field.text.as_str(), prompt)],
                _ => vec![
                    ("Prompt", form.prompt_field.text.as_str(), prompt),
                    ("Target", target_source, target),
                ],
            };
            for (label, field, value) in sections {
                if roomy && lines.last().is_some_and(|line| line.width() > 0) {
                    lines.push(Line::raw(""));
                }
                lines.push(Line::from(vec![
                    Span::styled(
                        label,
                        Style::default().fg(COPPER).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!(" · {}", clean(field)), Style::default().fg(MUTED)),
                ]));
                lines.extend(value.split('\n').map(|line| Line::raw(line.to_owned())));
            }
            if pair.is_some_and(|row| row["transformation"]["kind"] == "binary_choice_scores") {
                lines.push(Line::styled(
                    "Target format · choice labels",
                    Style::default().fg(MUTED),
                ));
            }
        } else {
            lines.push(Line::styled(
                "No mapped pair is available. Refresh the preview.",
                Style::default().fg(RED),
            ));
        }
    } else if let Some(row) = preview["rows"]
        .as_array()
        .and_then(|rows| rows.get(form.preview_row))
    {
        lines.extend(record_lines(preview, row, roomy));
        if roomy
            && row.as_object().is_some_and(|fields| {
                fields
                    .values()
                    .any(|value| value.is_array() || value.is_object())
            })
        {
            lines.push(Line::raw(""));
            lines.push(Line::styled(
                "Nested fields: examples[].input",
                Style::default().fg(MUTED),
            ));
        }
    } else {
        lines.push(Line::styled(
            preview["error"]
                .as_str()
                .map(clean)
                .unwrap_or_else(|| "No records were returned.".into()),
            Style::default().fg(RED),
        ));
        for (label, key) in [("Configurations", "configs"), ("Splits", "splits")] {
            let choices = choice_text(preview, key);
            if !choices.is_empty() {
                lines.push(Line::raw(""));
                lines.push(Line::styled(label, Style::default().fg(COPPER)));
                lines.push(Line::raw(choices));
            }
        }
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((form.preview_scroll, 0)),
        inner,
    );
}

fn options_height(area: Rect, count: usize) -> u16 {
    if area.height as usize >= count * 3 + 4 {
        count as u16 * 3
    } else {
        count as u16
    }
}

fn render_options(
    frame: &mut Frame,
    area: Rect,
    options: &[(&str, &str, &str)],
    selected: usize,
    active: bool,
) {
    let expanded = area.height as usize >= options.len() * 3;
    let mut lines = Vec::new();
    for (index, (_, label, description)) in options.iter().enumerate() {
        let chosen = index == selected;
        lines.push(Line::from(vec![
            Span::styled(
                if chosen { "› " } else { "  " },
                Style::default().fg(COPPER),
            ),
            Span::styled(
                *label,
                Style::default()
                    .fg(if chosen && active { COPPER } else { IVORY })
                    .add_modifier(if chosen {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ]));
        if expanded {
            lines.push(Line::styled(
                format!("  {description}"),
                Style::default().fg(MUTED),
            ));
            lines.push(Line::from(""));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_input(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    input: &Input,
    active: bool,
    placeholder: &str,
) {
    if area.height == 0 {
        return;
    }
    let block = Block::default()
        .title(Line::styled(
            format!(" {label} "),
            Style::default().fg(if active { COPPER } else { MUTED }),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if active { COPPER } else { RULE }));
    let inner = block.inner(area);
    let line = if input.text.is_empty() && !active {
        Line::styled(placeholder, Style::default().fg(MUTED))
    } else {
        input.line(
            (inner.width as usize * inner.height.max(1) as usize).max(1),
            active,
        )
    };
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(line).wrap(Wrap { trim: false }), inner);
}

fn render_review(frame: &mut Frame, area: Rect, form: &Form) {
    let result = form.result();
    let validation = (result.limit / 5).max(1);
    let editing = result.limit - validation;
    let upload = result.publish_repo.as_ref().map_or_else(
        || "Local checkpoint only".to_owned(),
        |repo| {
            format!(
                "{repo} · {} · {}",
                result.publish_visibility,
                if result.publish_create {
                    "new"
                } else {
                    "existing"
                }
            )
        },
    );
    let mut lines = vec![
        review_line("Model", &result.model),
        review_line("Dataset", &result.dataset),
        review_line("Data kind", DATA_MODES[form.dataset_mode].1),
        review_line(
            "Editor",
            &format!(
                "{} · {}",
                EDITORS[form.editor].1,
                result
                    .editor_model
                    .as_deref()
                    .unwrap_or(form.default_editor_model())
            ),
        ),
        review_line("Budget", &format!("${:.2} USD", result.budget)),
        review_line("Upload", &upload),
        review_line(
            "Measure",
            if form.dataset_mode == 0 {
                METRICS[form.metric].1
            } else {
                METRICS[3].1
            },
        ),
    ];
    if let Some(category) = &form.custom_target_type {
        lines.insert(0, review_line("Category", &clean(category)));
    }
    if form.remote() {
        lines.push(review_line(
            "Source",
            &format!(
                "{} / {}",
                result.dataset_config.as_deref().unwrap_or("auto"),
                result.dataset_split
            ),
        ));
        let fields = match form.dataset_mode {
            1 => result.text_field.clone(),
            2 => result.prompt_field.clone(),
            _ => format!("{} → {}", result.prompt_field, result.target_field),
        };
        lines.push(review_line("Fields", &fields));
        lines.push(review_line(
            "Examples",
            &format!(
                "≤ {} · ~{} edit / {} validation",
                result.limit, editing, validation
            ),
        ));
        lines.push(review_line(
            "Revision",
            result.dataset_revision.as_deref().unwrap_or(""),
        ));
    }
    let trajectory = result.trajectory_repo.as_ref().map_or_else(
        || {
            if result.save_trajectory {
                "Local trajectory.jsonl".to_owned()
            } else {
                "Off".to_owned()
            }
        },
        |repo| {
            format!(
                "{repo} · {} · {}",
                result.trajectory_visibility,
                if result.trajectory_create {
                    "new"
                } else {
                    "existing"
                }
            )
        },
    );
    lines.push(review_line("Trajectory", &trajectory));
    if result.health_checks.is_empty() {
        lines.push(review_line("Health", "No checks selected"));
    } else {
        lines.push(review_line(
            "Health",
            &format!(
                "{} checks · {}",
                result.health_checks.len(),
                result.health_max_drop.map_or_else(
                    || "monitor only".to_owned(),
                    |value| format!("maximum drop {:.1} pp", value * 100.0)
                )
            ),
        ));
        for selection in &result.health_checks {
            lines.push(review_line(
                "",
                &format!("{} · {} examples", selection.check, selection.examples),
            ));
        }
    }
    lines.push(review_line("Objective", &result.goal));
    if !result.controls.is_empty() {
        lines.push(review_line("Keep intact", &result.controls));
    }
    if result.publish_repo.is_some() {
        lines.push(Line::styled(
            "The verified checkpoint will be uploaded to this repository after the run.",
            Style::default().fg(MUTED),
        ));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .scroll((form.review_scroll, 0)),
        area,
    );
}

fn review_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<11} "), Style::default().fg(MUTED)),
        Span::raw(value.to_owned()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, FromArgMatches};
    use ratatui::backend::TestBackend;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn preview_payload() -> Value {
        serde_json::json!({"repo":"example/questions","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","config":"main","split":"train","configs":["main"],"splits":["train","test"],"columns":[{"name":"question","dtype":"string"},{"name":"answer","dtype":"string"}],"rows":[{"question":"What colour is the sky?","answer":"Blue."},{"question":"What is two plus two?","answer":"Four."}],"prompt_field":"question","target_field":"answer","error":null,"mapping_error":null,"truncated":false})
    }

    fn prepared() -> Form {
        let mut form = Form {
            step: 2,
            dataset: Input::new("hf://example/questions"),
            ..Default::default()
        };
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(preview_payload()));
        form
    }

    fn rendered_preview(form: &Form, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| render_preview(frame, frame.area(), form))
            .unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn raw_records_stack_field_names_above_contents_until_confirmed() {
        let mut form = prepared();
        assert!(form.preview_ready());
        assert!(!form.mapping_confirmed());
        for (width, height) in [(45, 22), (56, 7)] {
            let lines = rendered_preview(&form, width, height);
            let question = lines
                .iter()
                .position(|line| line.trim_matches(|ch| ch == ' ' || ch == '│') == "question")
                .unwrap();
            assert!(lines[question + 1].contains("What colour is the sky?"));
            let answer = lines
                .iter()
                .position(|line| line.trim_matches(|ch| ch == ' ' || ch == '│') == "answer")
                .unwrap();
            assert!(lines[answer + 1].contains("Blue."));
            assert!(!lines.join("\n").contains("Raw row:"));
            assert!(!lines.join("\n").contains("Prompt ·"));
        }
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 2);
        assert!(form.mapping_confirmed());
        let text = rendered_preview(&form, 56, 7).join("\n");
        assert!(text.contains("Prompt · question"));
        assert!(text.contains("Target · answer"));
        assert!(text.contains("Blue."));
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 3);
    }

    #[test]
    fn confirmation_resets_on_mapping_source_token_and_refresh_changes() {
        let mut form = prepared();
        form.key(key(KeyCode::Enter));
        form.prompt_field = Input::new("other");
        form.invalidate_preview();
        assert!(!form.mapping_confirmed());
        assert!(!form.preview_ready());
        assert!(form.preview.is_some());
        assert!(rendered_preview(&form, 56, 7).join("\n").contains("Blue."));
        form.dataset_split = Input::new("test");
        form.invalidate_preview();
        assert!(form.preview.is_none());
        let mut form = prepared();
        form.key(key(KeyCode::Enter));
        form.request_preview(true);
        assert!(!form.mapping_confirmed());
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(preview_payload()));
        assert!(!form.mapping_confirmed());
        form.key(key(KeyCode::Enter));
        form.hf_token = Input::new("changed-token");
        form.token_changed("");
        assert!(!form.mapping_confirmed());
        assert!(form.preview.is_none());
    }

    #[test]
    fn nested_pairs_share_a_record_and_navigate_after_confirmation() {
        let mut payload = preview_payload();
        payload["columns"] = serde_json::json!([{"name":"examples","dtype":"list"}]);
        payload["rows"] = serde_json::json!([{"examples":[{"input":"First question", "target":"First answer"},{"input":"Second question", "target":"Second answer"}]}]);
        payload["prompt_field"] = Value::String("examples[].input".into());
        payload["target_field"] = Value::String("examples[].target".into());
        payload["mapped_rows"] = serde_json::json!([
            {"row_index":0,"example_index":0,"prompt":"First question","target":"First answer"},
            {"row_index":0,"example_index":1,"prompt":"Second question","target":"Second answer"}
        ]);
        let mut form = Form {
            step: 2,
            dataset: Input::new("hf://example/questions"),
            ..Default::default()
        };
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(payload));
        assert_eq!(form.preview_count(), 1);
        assert!(form.validate().is_ok());
        form.key(key(KeyCode::Enter));
        assert!(form.mapping_confirmed());
        assert_eq!(form.preview_count(), 2);
        form.key(key(KeyCode::PageDown));
        assert_eq!(form.preview_row, 1);
        let text = rendered_preview(&form, 56, 10).join("\n");
        assert!(text.contains("Second question"));
        assert!(text.contains("Second answer"));
        assert!(!text.contains("First question"));
        form.key(key(KeyCode::PageDown));
        assert_eq!(form.preview_row, 0);
    }

    #[test]
    fn confirming_mapping_keeps_the_selected_source_record() {
        let mut form = prepared();
        let mut payload = preview_payload();
        payload["mapped_rows"] = serde_json::json!([
            {"row_index":0,"example_index":0,"prompt":"First","target":"A"},
            {"row_index":0,"example_index":1,"prompt":"Second","target":"B"},
            {"row_index":1,"example_index":0,"prompt":"Third","target":"C"}
        ]);
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(payload));
        form.key(key(KeyCode::PageDown));
        assert_eq!(form.preview_row, 1);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.preview_row, 2);
        assert!(rendered_preview(&form, 56, 7).join("\n").contains("Third"));
    }

    #[test]
    fn corpus_and_prompt_modes_confirm_only_their_relevant_fields() {
        for (mode, field, value, label) in [
            ("corpus", "body", "A quiet, measured passage.", "Text"),
            ("prompts", "question", "Write a short greeting.", "Prompt"),
        ] {
            let options = WizardOptions {
                dataset_mode: mode.into(),
                ..Default::default()
            };
            let mut form = Form::from_options(&options);
            form.step = 2;
            form.dataset = Input::new("hf://example/questions");
            let request = form.preview_request().unwrap();
            let mut payload = preview_payload();
            payload["dataset_mode"] = Value::String(mode.into());
            payload["columns"] = serde_json::json!([{"name":field,"dtype":"string"}]);
            payload["rows"] = serde_json::json!([{field:value}]);
            payload["prompt_field"] = if mode == "prompts" {
                Value::String(field.into())
            } else {
                Value::Null
            };
            payload["text_field"] = if mode == "corpus" {
                Value::String(field.into())
            } else {
                Value::Null
            };
            payload["target_field"] = Value::Null;
            payload["mapped_rows"] = if mode == "corpus" {
                serde_json::json!([{"row_index":0,"example_index":0,"prompt":"","target":value}])
            } else {
                serde_json::json!([{"row_index":0,"example_index":0,"prompt":value,"target":""}])
            };
            form.accept_preview(request, Ok(payload));
            assert!(form.preview_ready());
            assert!(!form.mapping_confirmed());
            form.key(key(KeyCode::Enter));
            assert!(form.mapping_confirmed());
            let text = rendered_preview(&form, 56, 7).join("\n");
            assert!(text.contains(&format!("{label} · {field}")));
            assert!(text.contains(value));
            assert!(!text.contains("Target ·"));
            if mode == "corpus" {
                assert!(!text.contains("Prompt ·"));
            }
            form.key(key(KeyCode::Enter));
            assert_eq!(form.step, 3);
            assert_eq!(form.result().metric, "semantic");
            assert_eq!(form.result().dataset_mode, mode);
            assert!(form.result().target_field.is_empty());
            form.step = 4;
            assert!(form.selection().is_none());
            let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
            terminal.draw(|frame| render(frame, &form)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("Goal score"));
            assert!(!text.contains("Exact match"));
        }
    }

    #[test]
    fn changing_data_kind_invalidates_mapping_and_skips_unused_target_input() {
        let mut form = prepared();
        form.key(key(KeyCode::Enter));
        assert!(form.mapping_confirmed());
        form.field = 4;
        assert!(matches!(form.key(key(KeyCode::F(2))), Action::Preview(_)));
        assert_eq!(form.dataset_mode(), "corpus");
        assert_eq!(form.field, 3);
        assert!(!form.mapping_confirmed());
        assert!(!form.preview_ready());
        assert!(form.preview.is_some());
        assert_eq!(form.preview_request().unwrap().dataset_mode, "corpus");
        assert_eq!(form.preview_request().unwrap().prompt_field, None);
        form.key(key(KeyCode::Tab));
        assert_eq!(form.field, 5);
        form.key(key(KeyCode::Tab));
        assert_eq!(form.field, 6);
        form.key(key(KeyCode::BackTab));
        assert_eq!(form.field, 5);
        form.key(key(KeyCode::BackTab));
        assert_eq!(form.field, 3);
        form.key(key(KeyCode::F(2)));
        assert_eq!(form.dataset_mode(), "prompts");
        form.key(key(KeyCode::F(2)));
        assert_eq!(form.dataset_mode(), "paired");
        assert!(!form.mapping_confirmed());
    }

    #[test]
    fn local_text_and_prompt_field_selectors_are_reachable() {
        for (mode, expected_label) in [("corpus", "Text field"), ("prompts", "Prompt field")] {
            let mut form = Form::from_options(&WizardOptions {
                dataset_mode: mode.into(),
                ..Default::default()
            });
            form.step = 2;
            form.dataset = Input::new("records.jsonl");
            form.key(key(KeyCode::Tab));
            assert_eq!(form.field, 3);
            form.input().unwrap().insert("document.body");
            if mode == "corpus" {
                assert_eq!(form.result().text_field, "document.body");
            } else {
                assert_eq!(form.result().prompt_field, "document.body");
            }
            for (width, height) in [(100, 32), (60, 18)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| render(frame, &form)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect();
                assert!(text.contains(expected_label));
                assert!(!text.contains("Target field"));
            }
            form.key(key(KeyCode::Tab));
            assert_eq!(form.field, 6);
            form.key(key(KeyCode::Tab));
            assert_eq!(form.field, 0);
        }
    }

    #[test]
    fn data_kind_and_text_options_survive_and_render_at_both_sizes() {
        let options = WizardOptions {
            dataset_mode: "corpus".into(),
            text_field: Some("document.text".into()),
            goal: Some("Write in the corpus style.".into()),
            ..Default::default()
        };
        let mut form = Form::from_options(&options);
        form.step = 2;
        form.dataset = Input::new("hf://example/documents");
        assert_eq!(
            form.preview_request().unwrap().text_field.as_deref(),
            Some("document.text")
        );
        assert_eq!(form.result().text_field, "document.text");
        form.prefill_goal();
        assert_eq!(form.goal.text, "Write in the corpus style.");
        for (width, height) in [(100, 32), (60, 18)] {
            for &field in form.dataset_field_indices() {
                form.field = field;
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| render(frame, &form)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect();
                assert!(text.contains("Text corpus"));
                assert!(text.contains("F2 data kind"));
                assert!(!text.contains("Target field"));
                assert!(text.contains(dataset_fields(&form)[field].0));
            }
        }
    }

    #[test]
    fn compact_preview_scroll_reaches_the_end_of_raw_and_mapped_content() {
        let mut form = prepared();
        let mut payload = preview_payload();
        let prompt = (1..=10)
            .map(|n| format!("Prompt line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        payload["rows"] = serde_json::json!([{"question":prompt,"answer":"Last target line"}]);
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(payload));
        assert!(rendered_preview(&form, 58, 6)
            .join("\n")
            .contains("question"));
        for _ in 0..10 {
            form.key(KeyEvent::new(KeyCode::Down, KeyModifiers::CONTROL));
        }
        let raw = rendered_preview(&form, 58, 6).join("\n");
        assert!(raw.contains("answer"));
        assert!(raw.contains("Last target line"));
        form.key(key(KeyCode::Enter));
        assert_eq!(form.preview_scroll, 0);
        assert!(rendered_preview(&form, 58, 6)
            .join("\n")
            .contains("Prompt ·"));
        for _ in 0..10 {
            form.key(KeyEvent::new(KeyCode::Down, KeyModifiers::CONTROL));
        }
        let mapped = rendered_preview(&form, 58, 6).join("\n");
        assert!(mapped.contains("Target ·"));
        assert!(mapped.contains("Last target line"));
        form.key(key(KeyCode::PageDown));
        assert_eq!(form.preview_scroll, 0);
    }

    #[test]
    fn multiline_values_keep_lines_and_strip_terminal_controls() {
        let value = serde_json::json!("first\n  second\r\n\u{001b}[2Jthird");
        let text = value_lines(&value)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert_eq!(text, vec!["first", "  second", "[2Jthird"]);
        let rows = record_lines(
            &preview_payload(),
            &serde_json::json!({"question":"line one\nline two", "answer":0}),
            false,
        );
        let text = rows.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(
            text,
            vec!["question", "line one", "line two", "answer", "0"]
        );
    }

    #[test]
    fn unicode_editing_never_splits_utf8() {
        let mut input = Input::new("café東京");
        input.key(key(KeyCode::Left));
        input.key(key(KeyCode::Backspace));
        input.key(key(KeyCode::Char('λ')));
        assert_eq!(input.text, "caféλ京");
        input.key(key(KeyCode::Home));
        input.key(key(KeyCode::Delete));
        assert_eq!(input.text, "aféλ京");
        assert_eq!(input.cursor, 0);
    }

    #[test]
    fn navigation_requires_valid_fields_and_preserves_backtracking() {
        let mut form = Form::default();
        form.key(key(KeyCode::Down));
        assert_eq!(form.target, 1);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 1);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 2);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 2);
        assert!(form.error.contains("JSONL"));
        form.dataset = Input::new("hf://example/questions");
        form.key(key(KeyCode::Tab));
        assert_eq!(form.field, 1);
        assert!(!form.preview_ready());
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(preview_payload()));
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 2);
        assert!(form.mapping_confirmed());
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 3);
        form.key(key(KeyCode::Esc));
        assert_eq!(form.step, 2);
        assert_eq!(form.dataset.text, "hf://example/questions");
    }

    #[test]
    fn hf_requires_a_fresh_preview_and_preserves_explicit_limits() {
        let mut form = Form {
            step: 2,
            dataset: Input::new("hf://example/questions"),
            ..Default::default()
        };
        assert!(matches!(form.key(key(KeyCode::Enter)), Action::Preview(_)));
        assert_eq!(form.step, 2);
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(preview_payload()));
        assert!(form.preview_ready());
        assert_eq!(form.prompt_field.text, "question");
        assert_eq!(form.target_field.text, "answer");
        assert!(!form.mapping_confirmed());
        form.key(key(KeyCode::Enter));
        assert!(form.mapping_confirmed());
        form.limit = Input::new("50");
        assert!(form.mapping_confirmed());
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 3);
        let result = form.result();
        assert_eq!(result.limit, 50);
        assert_eq!(result.dataset_config.as_deref(), Some("main"));
        assert_eq!(
            result.dataset_revision.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }

    #[test]
    fn stale_results_cannot_authorise_changed_source_or_mapping() {
        let mut form = prepared();
        let old = form.preview_request().unwrap();
        form.prompt_field = Input::new("missing");
        form.invalidate_preview();
        assert!(!form.preview_ready());
        form.accept_preview(old, Ok(preview_payload()));
        assert!(form.preview.is_some());
        assert!(!form.preview_ready());
        assert!(!form.mapping_confirmed());
        let request = form.preview_request().unwrap();
        let mut invalid = preview_payload();
        invalid["prompt_field"] = Value::Null;
        invalid["mapping_error"] = Value::String("Missing prompt field".into());
        form.accept_preview(request, Ok(invalid));
        assert!(!form.preview_ready());
        assert_eq!(form.prompt_field.text, "missing");
        assert!(!form.preview.as_ref().unwrap()["rows"]
            .as_array()
            .unwrap()
            .is_empty());
        form.dataset = Input::new("hf://example/another");
        form.invalidate_preview();
        assert!(form.preview.is_none());
    }

    #[test]
    fn hf_limit_and_all_seven_fields_are_validated_and_reachable() {
        let mut form = prepared();
        for n in ["0", "4", "100001", "1.5", "NaN", ""] {
            form.limit = Input::new(n);
            assert!(form.validate().is_err());
        }
        for n in ["5", "50", "100000"] {
            form.limit = Input::new(n);
            assert!(form.validate().is_ok());
        }
        form.field = 0;
        for expected in [1, 2, 3, 4, 5, 6, 0] {
            form.key(key(KeyCode::Tab));
            assert_eq!(form.field, expected);
        }
        form.key(key(KeyCode::BackTab));
        assert_eq!(form.field, 6);
    }

    #[test]
    fn options_and_explicit_mapping_survive_preview() {
        let options = WizardOptions {
            limit: 75,
            dataset_config: Some("custom".into()),
            dataset_split: "test".into(),
            dataset_revision: Some("revision".into()),
            prompt_field: Some("nested.prompt".into()),
            target_field: Some("nested.target".into()),
            ..Default::default()
        };
        let mut form = Form::from_options(&options);
        form.dataset = Input::new("hf://example/questions");
        let request = form.preview_request().unwrap();
        assert_eq!(request.revision.as_deref(), Some("revision"));
        form.accept_preview(request, Ok(preview_payload()));
        assert_eq!(form.prompt_field.text, "nested.prompt");
        assert_eq!(form.target_field.text, "nested.target");
        assert_eq!(form.dataset_config.text, "custom");
        assert_eq!(form.result().limit, 75);
        assert_eq!(form.result().dataset_split, "test");
    }

    #[test]
    fn fetched_cells_are_sanitised_and_samples_navigate() {
        let mut form = prepared();
        form.key(key(KeyCode::PageDown));
        assert_eq!(form.preview_row, 1);
        form.key(key(KeyCode::PageUp));
        assert_eq!(form.preview_row, 0);
        assert_eq!(
            mapped_text(
                &serde_json::json!({"a":{"b":"\u{001b}[2J\u{202e}safe"}}),
                "a.b"
            )
            .unwrap(),
            "[2Jsafe"
        );
    }

    #[test]
    fn hf_preview_and_release_identity_render_at_both_sizes() {
        let form = prepared();
        for (width, height) in [(100, 32), (60, 18)] {
            for field in 0..7 {
                let mut form = Form {
                    field,
                    ..prepared()
                };
                form.limit = Input::new("50");
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| render(frame, &form)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect();
                assert!(text.contains("github:chrisvoncsefalvay/vespasian"));
                assert!(text.contains(env!("CARGO_PKG_VERSION")));
                assert!(text.contains("Live preview"));
                assert!(text.contains("Blue."));
                assert!(text.contains(dataset_fields(&form)[field].0));
            }
        }
        assert!(form.preview_ready());
    }

    #[test]
    fn nonfinite_or_nonpositive_budgets_are_rejected() {
        let mut form = Form {
            step: 7,
            editor_model: Input::new("gpt-6.1-sol"),
            ..Default::default()
        };
        for value in ["0", "-1", "NaN", "inf", "USD 5"] {
            form.budget = Input::new(value);
            assert!(form.validate().is_err(), "accepted {value}");
        }
        form.budget = Input::new("2.50");
        assert!(form.validate().is_ok());
    }

    #[test]
    fn budget_selects_a_frontier_model_and_review_returns_explicit_choices() {
        let mut form = Form {
            step: 6,
            editor: 2,
            ..Default::default()
        };
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 7);
        assert_eq!(form.editor_model.text, "gpt-6.1-sol");
        form.key(key(KeyCode::Tab));
        form.key(key(KeyCode::Down));
        assert_eq!(form.editor_model.text, "gpt-6-astra");
        form.editor_model = Input::default();
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 7);
        assert!(form.error.contains("model ID"));
        form.editor_model = Input::new("chosen-frontier-model");
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 8);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 9);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 10);
        match form.key(key(KeyCode::Enter)) {
            Action::Start(result) => {
                assert_eq!(result.editor, "openai");
                assert_eq!(
                    result.editor_model.as_deref(),
                    Some("chosen-frontier-model")
                );
                assert_eq!(result.budget, 5.0);
                assert!(result.publish_repo.is_none());
            }
            _ => panic!("review did not start the run"),
        }
    }

    #[test]
    fn custom_model_and_cancellation_work() {
        let mut form = Form {
            step: 1,
            model: 2,
            ..Default::default()
        };
        form.key(key(KeyCode::Enter));
        assert!(form.typing);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 1);
        form.custom_model = Input::new("org/model");
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 2);
        assert!(matches!(
            form.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Cancel
        ));
    }

    #[test]
    fn screens_render_at_compact_and_full_sizes() {
        let mut form = Form::default();
        for (width, height) in [(60, 18), (100, 32), (20, 5)] {
            for (step, name) in STEPS.iter().enumerate() {
                form.step = step;
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| render(frame, &form)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect();
                assert!(text.contains("VESPASIAN"));
                if width >= 60 {
                    assert!(text.contains(name) || text.contains(" / 11"));
                }
            }
        }
    }

    #[test]
    fn objective_uses_target_column_and_preserves_user_edits() {
        let mut form = prepared();
        form.key(key(KeyCode::Enter));
        form.key(key(KeyCode::Enter));
        assert_eq!(
            form.goal.text,
            "Minimise answer errors against the answer column."
        );
        form.target_field = Input::new("gold");
        form.prefill_goal();
        assert_eq!(
            form.goal.text,
            "Minimise answer errors against the gold column."
        );
        form.target_field = Input::new("answer");
        form.prefill_goal();
        form.goal = Input::new("Answer arithmetic questions accurately.");
        form.key(key(KeyCode::Tab));
        form.input().unwrap().insert("Keep explanations concise.");
        form.key(key(KeyCode::Esc));
        form.key(key(KeyCode::Enter));
        assert_eq!(form.goal.text, "Answer arithmetic questions accurately.");
        assert_eq!(form.result().controls, "Keep explanations concise.");
    }

    fn guided_cli_form(arguments: &[&str]) -> (crate::Cli, Form) {
        let matches = crate::Cli::command()
            .try_get_matches_from(arguments)
            .unwrap();
        let cli = crate::Cli::from_arg_matches(&matches).unwrap();
        let form = Form::from_options(&WizardOptions::from((&cli, &matches)));
        (cli, form)
    }

    fn guided_cli_round_trip(arguments: &[&str]) -> crate::Cli {
        let (mut cli, form) = guided_cli_form(arguments);
        crate::apply_guided(&mut cli, form.result());
        cli
    }

    #[test]
    fn guided_cli_defaults_survive_form_round_trip() {
        let cli = guided_cli_round_trip(&["vespasian"]);
        assert_eq!(cli.target_type, "factual");
        assert_eq!(cli.model, "Qwen3-0.8B");
        assert_eq!(cli.editor, "codex");
        assert_eq!(cli.metric, "auto");
        assert_eq!(cli.editor_model, None);
        assert_eq!(cli.budget, 2.0);
    }

    #[test]
    fn guided_cli_explicit_default_target_type_survives_form_round_trip() {
        let cli = guided_cli_round_trip(&["vespasian", "--target-type", "custom"]);
        assert_eq!(cli.target_type, "custom");
    }

    #[test]
    fn guided_cli_custom_category_survives_form_round_trip() {
        let (mut cli, form) = guided_cli_form(&["vespasian", "--target-type", "summarisation"]);
        assert_eq!(TARGETS[form.target].0, "custom");
        crate::apply_guided(&mut cli, form.result());
        assert_eq!(cli.target_type, "summarisation");
    }

    #[test]
    fn guided_cli_custom_category_is_replaced_by_a_menu_choice() {
        for (choices, expected) in [
            (vec![KeyCode::Up], "classification"),
            (vec![KeyCode::Up, KeyCode::Down], "custom"),
        ] {
            let (mut cli, mut form) =
                guided_cli_form(&["vespasian", "--target-type", "summarisation"]);
            for choice in choices {
                form.key(key(choice));
            }
            crate::apply_guided(&mut cli, form.result());
            assert_eq!(cli.target_type, expected);
        }
    }

    #[test]
    fn guided_cli_custom_category_is_visible_in_review() {
        let (_, form) = guided_cli_form(&["vespasian", "--target-type", "summarisation"]);
        for (width, height) in [(60, 18), (100, 32)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| render_review(frame, frame.area(), &form))
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("Category"));
            assert!(text.contains("summarisation"));
        }
    }

    #[test]
    fn guided_cli_standard_choices_survive_form_round_trip() {
        for (model, editor, metric, target_type) in [
            ("Gemma3-1B", "anthropic", "token_f1", "style"),
            ("Qwen3-0.8B", "openai", "exact_match", "format"),
            ("Gemma3-1B", "claude", "semantic", "classification"),
            ("Qwen3-0.8B", "codex", "auto", "factual"),
            ("Gemma3-1B", "anthropic", "exact_match", "custom"),
        ] {
            let cli = guided_cli_round_trip(&[
                "vespasian",
                "--model",
                model,
                "--editor",
                editor,
                "--metric",
                metric,
                "--target-type",
                target_type,
            ]);
            assert_eq!(cli.model, model);
            assert_eq!(cli.editor, editor);
            assert_eq!(cli.metric, metric);
            assert_eq!(cli.target_type, target_type);
        }
    }

    #[test]
    fn guided_cli_custom_models_survive_form_round_trip() {
        for model in [
            "owner/custom-checkpoint",
            "Qwen/Qwen3.5-0.8B",
            "Qwen3.5-0.8B",
            "/tmp/local checkpoint",
        ] {
            let cli = guided_cli_round_trip(&["vespasian", "--model", model]);
            assert_eq!(cli.model, model);
        }
    }

    #[test]
    fn guided_cli_provider_and_explicit_model_survive_form_round_trip() {
        for (editor, editor_model) in [
            ("anthropic", "claude-opus-5-5"),
            ("claude", "chosen-claude-model"),
            ("openai", "gpt-6-astra"),
            ("codex", "chosen-codex-model"),
        ] {
            let cli = guided_cli_round_trip(&[
                "vespasian",
                "--editor",
                editor,
                "--editor-model",
                editor_model,
            ]);
            assert_eq!(cli.editor, editor);
            assert_eq!(cli.editor_model.as_deref(), Some(editor_model));
        }
    }

    #[test]
    fn guided_cli_provider_controls_the_default_editor_model() {
        for (editor, editor_model) in [
            ("anthropic", "claude-sonnet-5-5"),
            ("claude", "claude-sonnet-5-5"),
            ("openai", "gpt-6.1-sol"),
            ("codex", "gpt-6.1-sol"),
        ] {
            let (mut cli, mut form) = guided_cli_form(&["vespasian", "--editor", editor]);
            form.step = 6;
            form.key(key(KeyCode::Enter));
            assert_eq!(form.step, 7);
            crate::apply_guided(&mut cli, form.result());
            assert_eq!(cli.editor_model.as_deref(), Some(editor_model));
            assert_eq!(cli.editor, editor);
        }
    }

    #[test]
    fn cli_objective_budget_controls_and_publish_choices_survive() {
        let options = WizardOptions {
            goal: Some("Be concise.".into()),
            editor_model: Some("chosen-model".into()),
            budget: 2.5,
            controls: "Preserve factual accuracy.".into(),
            hf_token: Some("hf_session_secret".into()),
            publish_repo: Some("owner/checkpoint".into()),
            publish_visibility: "public".into(),
            publish_create: true,
            ..Default::default()
        };
        let mut form = Form::from_options(&options);
        form.prefill_goal();
        let result = form.result();
        assert_eq!(result.goal, "Be concise.");
        assert_eq!(result.editor_model.as_deref(), Some("chosen-model"));
        assert_eq!(result.budget, 2.5);
        assert_eq!(result.controls, "Preserve factual accuracy.");
        assert_eq!(result.hf_token.as_deref(), Some("hf_session_secret"));
        assert_eq!(result.publish_repo.as_deref(), Some("owner/checkpoint"));
        assert_eq!(result.publish_visibility, "public");
        assert!(result.publish_create);
    }

    #[test]
    fn token_edit_invalidates_preview_and_is_masked_everywhere() {
        let mut form = prepared();
        form.field = 6;
        form.key(key(KeyCode::Char('s')));
        assert!(!form.preview_ready());
        assert!(form.preview.is_none());
        form.hf_token = Input::new("hf_session_secret");
        for (width, height) in [(100, 32), (60, 18)] {
            for (step, field, typing) in [(2, 6, false), (8, 3, true), (10, 0, false)] {
                form.step = step;
                form.field = field;
                form.typing = typing;
                form.publish = 1;
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| render(frame, &form)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect();
                assert!(!text.contains("hf_session_secret"));
                if step != 10 {
                    assert!(text.contains("••••••••"));
                }
            }
        }
    }

    #[test]
    fn publish_is_local_by_default_and_validates_explicit_upload() {
        let mut form = Form {
            step: 8,
            ..Default::default()
        };
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 9);
        assert!(form.result().publish_repo.is_none());
        form.key(key(KeyCode::Esc));
        form.key(key(KeyCode::Down));
        form.key(key(KeyCode::Enter));
        assert!(form.typing);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 8);
        assert!(form.error.contains("owner/name"));
        form.publish_repo = Input::new("owner/checkpoint");
        form.key(key(KeyCode::Tab));
        form.key(key(KeyCode::Right));
        assert_eq!(form.publish_visibility.text, "public");
        form.key(key(KeyCode::Tab));
        form.key(key(KeyCode::Right));
        assert_eq!(form.publish_mode.text, "new");
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 9);
        assert_eq!(
            form.result().publish_repo.as_deref(),
            Some("owner/checkpoint")
        );
        assert!(form.result().publish_create);
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 10);
        for (width, height) in [(100, 32), (60, 18)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| render(frame, &form)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("owner/checkpoint"));
            assert!(text.contains("public"));
        }
    }

    #[test]
    fn health_checks_keep_individual_counts_and_optional_drop_limits() {
        let mut form = Form {
            step: 5,
            ..Default::default()
        };
        assert!(form.validate().is_ok());
        assert!(form.result().health_checks.is_empty());
        assert!(form.result().health_max_drop.is_none());
        assert!(form.result().hf_token.is_none());
        form.key(key(KeyCode::Char(' ')));
        form.key(key(KeyCode::Tab));
        form.input().unwrap().text = "7".into();
        form.key(key(KeyCode::Tab));
        form.input().unwrap().text = "5".into();
        form.key(key(KeyCode::Tab));
        form.key(key(KeyCode::Down));
        form.key(key(KeyCode::Char(' ')));
        form.key(key(KeyCode::Tab));
        form.input().unwrap().text = "12".into();
        let result = form.result();
        assert_eq!(result.health_checks[0].check, "copying");
        assert_eq!(result.health_checks[0].examples, 7);
        assert_eq!(result.health_checks[1].examples, 12);
        assert_eq!(result.health_max_drop, Some(0.05));
        assert!(form.validate().is_ok());
        form.health_items[0].count = Input::new("1001");
        assert!(form.validate().is_err());
        form.health_items[0].count = Input::new("7");
        form.health_drop = Input::new("NaN");
        assert!(form.validate().is_err());
    }

    #[test]
    fn trajectory_is_opt_in_and_hub_upload_does_not_require_a_typed_token() {
        let mut form = Form {
            step: 9,
            ..Default::default()
        };
        assert!(!form.result().save_trajectory);
        assert!(form.result().trajectory_repo.is_none());
        form.key(key(KeyCode::Down));
        assert!(form.result().save_trajectory);
        assert!(form.result().trajectory_repo.is_none());
        form.key(key(KeyCode::Down));
        form.key(key(KeyCode::Enter));
        assert!(form.typing);
        assert!(form.validate().is_err());
        form.trajectory_repo = Input::new("owner/editor-runs");
        assert!(form.validate().is_ok());
        assert!(form.result().hf_token.is_none());
        form.key(key(KeyCode::Tab));
        form.key(key(KeyCode::Right));
        form.key(key(KeyCode::Tab));
        form.key(key(KeyCode::Right));
        form.key(key(KeyCode::Enter));
        assert_eq!(form.step, 10);
        let result = form.result();
        assert_eq!(result.trajectory_repo.as_deref(), Some("owner/editor-runs"));
        assert_eq!(result.trajectory_visibility, "public");
        assert!(result.trajectory_create);
    }

    #[test]
    fn dataset_keeps_the_section_sidebar_on_wide_terminals() {
        let form = prepared();
        let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
        terminal.draw(|frame| render(frame, &form)).unwrap();
        let buffer = terminal.backend().buffer();
        let left: String = (6..28)
            .flat_map(|y| (3..25).map(move |x| (x, y)))
            .map(|position| buffer[position].symbol())
            .collect();
        for label in [
            "Behaviour",
            "Model",
            "Dataset",
            "Health checks",
            "Trajectory",
            "Review",
        ] {
            assert!(left.contains(label), "missing sidebar label {label}");
        }
    }

    #[test]
    fn local_dataset_exposes_a_session_token() {
        let mut form = Form {
            step: 2,
            dataset: Input::new("data.jsonl"),
            ..Default::default()
        };
        form.key(key(KeyCode::Tab));
        assert_eq!(form.field, 6);
        form.input().unwrap().insert("hf_local_session");
        form.key(key(KeyCode::Tab));
        assert_eq!(form.field, 0);
        assert_eq!(form.result().hf_token.as_deref(), Some("hf_local_session"));
    }

    #[test]
    #[ignore = "requires VESPASIAN_SNAPSHOT_DIR for a visual capture"]
    fn render_intro() {
        save_capture(&Form::default(), "wizard");
    }

    #[test]
    #[ignore = "requires a real VESPASIAN_PREVIEW_JSON receipt and VESPASIAN_SNAPSHOT_DIR"]
    fn render_hf_preview() {
        let path = std::env::var_os("VESPASIAN_PREVIEW_JSON").expect("set VESPASIAN_PREVIEW_JSON");
        let payload: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut form = Form {
            step: 2,
            dataset_mode: DATA_MODES
                .iter()
                .position(|(id, _)| *id == payload["dataset_mode"].as_str().unwrap_or("paired"))
                .unwrap_or(0),
            dataset: Input::new(&format!("hf://{}", payload["repo"].as_str().unwrap())),
            limit: Input::new("50"),
            dataset_split: Input::new(payload["split"].as_str().unwrap()),
            ..Default::default()
        };
        let request = form.preview_request().unwrap();
        form.accept_preview(request, Ok(payload));
        form.field = 3;
        save_capture(&form, "dataset-record");
        form.key(key(KeyCode::Enter));
        assert!(form.mapping_confirmed());
        save_capture(&form, "dataset-mapped");
        if form.preview_count() > 2 {
            form.preview_row = 2;
            save_capture(&form, "dataset-choice");
            form.preview_row = 0;
        }
        form.step = 3;
        form.prefill_goal();
        form.controls = Input::new("Keep answers concise and preserve factual accuracy.");
        form.field = 1;
        save_capture(&form, "objective");
        form.step = 5;
        form.health_items[0].enabled = true;
        form.health_items[0].count = Input::new("20");
        form.health_items[1].enabled = true;
        form.health_items[1].count = Input::new("10");
        form.health_items[3].enabled = true;
        form.health_items[3].count = Input::new("5");
        save_capture(&form, "health-checks");
        form.step = 7;
        form.editor_model = Input::new(form.default_editor_model());
        save_capture(&form, "budget");
        form.step = 8;
        form.publish = 1;
        form.publish_repo = Input::new("your-name/edited-model");
        form.typing = true;
        form.field = 0;
        save_capture(&form, "publish");
        form.step = 9;
        form.trajectory = 2;
        form.trajectory_repo = Input::new("your-name/editor-trajectories");
        form.typing = true;
        form.field = 0;
        save_capture(&form, "trajectory");
        form.step = 10;
        form.typing = false;
        save_capture(&form, "review");
    }

    fn save_capture(form: &Form, stem: &str) {
        let output = std::path::PathBuf::from(
            std::env::var_os("VESPASIAN_SNAPSHOT_DIR").expect("set VESPASIAN_SNAPSHOT_DIR"),
        );
        std::fs::create_dir_all(&output).unwrap();
        let width = std::env::var("VESPASIAN_SNAPSHOT_WIDTH")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(100);
        let height = std::env::var("VESPASIAN_SNAPSHOT_HEIGHT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(32);
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| render(frame, form)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut text = String::new();
        let mut svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\"><rect width=\"100%\" height=\"100%\" fill=\"#14171b\"/><g font-family=\"DejaVu Sans Mono,monospace\" font-size=\"14\">", width as f64 * 8.5, height as u32 * 20);
        for y in 0..height {
            for x in 0..width {
                let cell = &buffer[(x, y)];
                text.push_str(cell.symbol());
                let (r, g, b) = match cell.fg {
                    Color::Rgb(r, g, b) => (r, g, b),
                    _ => (236, 226, 210),
                };
                let symbol = cell
                    .symbol()
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;");
                let weight = if cell.modifier.contains(Modifier::BOLD) {
                    "bold"
                } else {
                    "normal"
                };
                svg.push_str(&format!("<text x=\"{}\" y=\"{}\" fill=\"#{r:02x}{g:02x}{b:02x}\" font-weight=\"{weight}\">{symbol}</text>", x as f64 * 8.5, y as u32 * 20 + 16));
            }
            text.push('\n');
        }
        svg.push_str("</g></svg>");
        std::fs::write(output.join(format!("{stem}.txt")), text).unwrap();
        std::fs::write(output.join(format!("{stem}.svg")), svg).unwrap();
    }
}
