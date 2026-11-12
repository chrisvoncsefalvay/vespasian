use std::{
    cell::Cell,
    io::{self, Stdout},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, TryRecvError},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::Result;
use crossterm::{
    cursor::Show,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Wrap},
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

/// Owns terminal restoration, including early errors and unwinding.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
    }
}

#[derive(Clone)]
struct Turn {
    number: u64,
    accepted: bool,
    score: Option<f64>,
    loss: Option<f64>,
    health: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetailPane {
    Answers,
    Editor,
    Genealogy,
}
impl DetailPane {
    fn index(self) -> usize {
        match self {
            Self::Answers => 0,
            Self::Editor => 1,
            Self::Genealogy => 2,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Answers => "Answers",
            Self::Editor => "Editor",
            Self::Genealogy => "Genealogy",
        }
    }
    fn next(self) -> Self {
        match self {
            Self::Answers => Self::Editor,
            Self::Editor => Self::Genealogy,
            Self::Genealogy => Self::Answers,
        }
    }
}

#[derive(Clone)]
struct Answer {
    label: String,
    phase: String,
    turn: Option<u64>,
    current: Option<u64>,
    total: Option<u64>,
    example: Value,
    receipt: Value,
}

#[derive(Clone)]
struct EditorUpdate {
    status: String,
    text: String,
    elapsed_ms: Option<u64>,
}
#[derive(Clone)]
struct EditorCall {
    id: String,
    provider: String,
    model: String,
    purpose: String,
    status: String,
    turn: Option<u64>,
    updates: Vec<EditorUpdate>,
}

#[derive(Clone)]
struct Change {
    turn: u64,
    parent: u64,
    accepted: Option<bool>,
    proposal: Value,
    mutation: Value,
    metrics: Value,
    rationale: String,
    decision_reason: String,
}

struct Dashboard {
    started: Instant,
    model: String,
    editor: String,
    goal: String,
    output: String,
    phase: String,
    message: String,
    metric: String,
    baseline: Value,
    best: Value,
    health_baseline: Value,
    health_current: Value,
    health_best: Value,
    health_index: usize,
    budget_usd: f64,
    budget_mode: String,
    spent_usd: f64,
    input_tokens: u64,
    output_tokens: u64,
    max_turns: u64,
    turn: u64,
    current: u64,
    total: u64,
    turns: Vec<Turn>,
    plan: String,
    done: bool,
    failed: bool,
    cancelling: bool,
    plot_loss: bool,
    progress_known: bool,
    progress_label: String,
    retained_turn: u64,
    answers: Vec<Answer>,
    editor_calls: Vec<EditorCall>,
    changes: Vec<Change>,
    assessment_reason: String,
    detail_pane: DetailPane,
    detail_indices: [usize; 3],
    detail_scroll: [usize; 3],
    follow_latest: [bool; 3],
    expanded: bool,
    detail_geometry: Cell<(u16, u16)>,
    model_structure: Value,
    loop_stages: Vec<Value>,
    tensor_activity: Value,
    layer_offset: Option<usize>,
    layer_geometry: Cell<(usize, usize)>,
}

impl Default for Dashboard {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            model: String::new(),
            editor: String::new(),
            goal: String::new(),
            output: String::new(),
            phase: "Preparing".into(),
            message: "Opening session".into(),
            metric: "exact_match".into(),
            baseline: Value::Null,
            best: Value::Null,
            health_baseline: Value::Null,
            health_current: Value::Null,
            health_best: Value::Null,
            health_index: 0,
            budget_usd: 0.0,
            budget_mode: String::new(),
            spent_usd: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            max_turns: 0,
            turn: 0,
            current: 0,
            total: 0,
            turns: Vec::new(),
            plan: String::new(),
            done: false,
            failed: false,
            cancelling: false,
            plot_loss: false,
            progress_known: false,
            progress_label: String::new(),
            retained_turn: 0,
            answers: Vec::new(),
            editor_calls: Vec::new(),
            changes: Vec::new(),
            assessment_reason: String::new(),
            detail_pane: DetailPane::Answers,
            detail_indices: [0; 3],
            detail_scroll: [0; 3],
            follow_latest: [true; 3],
            expanded: false,
            detail_geometry: Cell::new((30, 18)),
            model_structure: Value::Null,
            loop_stages: vec![],
            tensor_activity: Value::Null,
            layer_offset: None,
            layer_geometry: Cell::new((0, 1)),
        }
    }
}

impl Dashboard {
    fn item_count(&self, pane: DetailPane) -> usize {
        match pane {
            DetailPane::Answers => self.answers.len(),
            DetailPane::Editor => self.editor_calls.len(),
            DetailPane::Genealogy => self.changes.len() + 1,
        }
    }

    fn follow_item(&mut self, pane: DetailPane) {
        let index = pane.index();
        if self.follow_latest[index] {
            self.detail_indices[index] = self.item_count(pane).saturating_sub(1);
            if self.detail_pane == pane {
                self.detail_scroll[index] = 0;
            }
        }
    }

    fn record_change(&mut self, event: &Value, accepted: Option<bool>) {
        let Some(turn) = event["turn"].as_u64().filter(|turn| *turn > 0) else {
            return;
        };
        let index = self
            .changes
            .iter()
            .position(|change| change.turn == turn)
            .unwrap_or_else(|| {
                self.changes.push(Change {
                    turn,
                    parent: event["parent_turn"].as_u64().unwrap_or(self.retained_turn),
                    accepted: None,
                    proposal: Value::Null,
                    mutation: Value::Null,
                    metrics: Value::Null,
                    rationale: String::new(),
                    decision_reason: String::new(),
                });
                self.changes.len() - 1
            });
        let change = &mut self.changes[index];
        if let Some(parent) = event["parent_turn"].as_u64() {
            change.parent = parent;
        }
        if event["proposal"].is_object() {
            change.proposal = event["proposal"].clone();
        }
        if !event["mutation"].is_null() {
            change.mutation = event["mutation"].clone();
        }
        if event["metrics"].is_object() {
            change.metrics = event["metrics"].clone();
        }
        change.rationale = change.proposal["rationale"]
            .as_str()
            .or_else(|| event["rationale"].as_str())
            .map(clean_text)
            .unwrap_or_else(|| change.rationale.clone());
        if let Some(value) = accepted {
            change.accepted = Some(value);
        }
        if let Some(reason) = event["decision_reason"].as_str() {
            change.decision_reason = clean_text(reason);
        }
        self.follow_item(DetailPane::Genealogy);
    }

    fn record_editor_activity(&mut self, event: &Value) {
        let id = event["call_id"]
            .as_str()
            .map(clean)
            .unwrap_or_else(|| format!("call-{}", self.editor_calls.len() + 1));
        let index = self
            .editor_calls
            .iter()
            .position(|call| call.id == id)
            .unwrap_or_else(|| {
                self.editor_calls.push(EditorCall {
                    id,
                    provider: String::new(),
                    model: String::new(),
                    purpose: String::new(),
                    status: String::new(),
                    turn: None,
                    updates: Vec::new(),
                });
                self.editor_calls.len() - 1
            });
        let call = &mut self.editor_calls[index];
        for (key, target) in [
            ("provider", &mut call.provider),
            ("model", &mut call.model),
            ("purpose", &mut call.purpose),
            ("status", &mut call.status),
        ] {
            if let Some(value) = event[key].as_str() {
                *target = clean(value);
            }
        }
        if event["turn"].is_u64() {
            call.turn = event["turn"].as_u64();
        }
        let text = event["summary"]
            .as_str()
            .or_else(|| event["text"].as_str())
            .map(clean_text)
            .unwrap_or_default();
        call.updates.push(EditorUpdate {
            status: call.status.clone(),
            text,
            elapsed_ms: event["elapsed_ms"].as_u64(),
        });
        self.follow_item(DetailPane::Editor);
    }

    fn key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        if key.code == KeyCode::Char('q')
            || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
        {
            return true;
        }
        let pane = self.detail_pane.index();
        match key.code {
            KeyCode::Tab => self.detail_pane = self.detail_pane.next(),
            KeyCode::BackTab => self.detail_pane = self.detail_pane.next().next(),
            KeyCode::Char('m') => self.plot_loss = !self.plot_loss,
            KeyCode::Char('l') => {
                if let Some(count) = self.model_structure["num_layers"]
                    .as_u64()
                    .filter(|count| *count > 0)
                {
                    let (start, visible) = self.layer_geometry.get();
                    self.layer_offset = Some((start + visible) % count as usize);
                }
            }
            KeyCode::Char('h') | KeyCode::Char(']') | KeyCode::Char('[') => {
                let count = health_ids(self).len();
                if count > 0 {
                    self.health_index = if key.code == KeyCode::Char('[') {
                        (self.health_index + count - 1) % count
                    } else {
                        (self.health_index + 1) % count
                    };
                }
            }
            KeyCode::Enter | KeyCode::Char('f') => self.expanded = !self.expanded,
            KeyCode::Esc => self.expanded = false,
            KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End => {
                self.layer_offset = None;
                let count = self.item_count(self.detail_pane);
                if count > 0 {
                    self.detail_indices[pane] = match key.code {
                        KeyCode::Left => self.detail_indices[pane].saturating_sub(1),
                        KeyCode::Right => (self.detail_indices[pane] + 1).min(count - 1),
                        KeyCode::Home => 0,
                        _ => count - 1,
                    };
                    self.follow_latest[pane] = key.code == KeyCode::End;
                    self.detail_scroll[pane] = 0;
                }
            }
            KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown => {
                let distance = if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) {
                    8
                } else {
                    1
                };
                let (width, height) = self.detail_geometry.get();
                let maximum = detail_lines(self, width.max(1))
                    .len()
                    .saturating_sub(height.max(1) as usize);
                self.detail_scroll[pane] = if matches!(key.code, KeyCode::Up | KeyCode::PageUp) {
                    self.detail_scroll[pane].saturating_sub(distance)
                } else {
                    (self.detail_scroll[pane] + distance).min(maximum)
                };
                self.follow_latest[pane] = false;
            }
            _ => {}
        }
        false
    }

    fn apply(&mut self, event: &Value) {
        match event["type"].as_str().unwrap_or("") {
            "start" => {
                self.model = field(event, "model");
                self.editor = field(event, "editor");
                self.goal = field(event, "goal");
                self.output = field(event, "output");
                self.budget_usd = finite_number(event, "budget_usd").unwrap_or(0.0);
                self.budget_mode = field(event, "budget_mode");
                self.max_turns = event["max_turns"].as_u64().unwrap_or(0);
            }
            "loop_stage" => {
                self.phase = format!(
                    "{} · {}",
                    title(&field(event, "stage")),
                    field(event, "status")
                );
                self.message = field(event, "summary");
                self.loop_stages.push(event.clone());
                if self.loop_stages.len() > 256 {
                    self.loop_stages.remove(0);
                }
            }
            "agent_tool" => {
                let status = field(event, "status");
                if matches!(status.as_str(), "started" | "completed" | "failed") {
                    // Only the public controller receipt belongs in this pane.
                    // Tool arguments, returned payloads and private reasoning do not.
                    self.message = field(event, "summary");
                    self.record_editor_activity(&serde_json::json!({
                        "call_id":format!("tool-{}-{}", event["turn"], field(event, "call_id")),
                        "provider":"local tool", "model":field(event, "tool"),
                        "purpose":"Read-only strategy discovery", "status":status,
                        "turn":event["turn"], "summary":field(event, "summary")
                    }));
                }
            }
            "budget_notice" => {
                self.message = field(event, "message");
            }
            "phase" | "progress" => {
                if let Some(phase) = event["phase"].as_str() {
                    self.phase = title(&clean(phase));
                }
                if let Some(message) = event["message"].as_str() {
                    self.message = clean(message);
                }
                self.current = event["current"].as_u64().unwrap_or(0);
                self.total = event["total"].as_u64().unwrap_or(0);
                self.progress_known = event["current"].as_u64().is_some() && self.total > 0;
                self.progress_label = field(event, "label");
            }
            "assessment" => {
                if let Some(metric) = event["metric"].as_str() {
                    self.metric = clean(metric);
                }
                self.plan = event["plan"].as_str().map(clean_text).unwrap_or_default();
                self.assessment_reason =
                    event["reason"].as_str().map(clean_text).unwrap_or_default();
                self.message = field(event, "reason");
                self.phase = if event["suitable"].as_bool().unwrap_or(false) {
                    "Dataset assessed".into()
                } else {
                    "Dataset needs attention".into()
                };
            }
            "baseline" => {
                self.baseline = event["metrics"].clone();
                self.best = self.baseline.clone();
                self.health_baseline = event["health"].clone();
                self.health_current = self.health_baseline.clone();
                self.health_best = self.health_baseline.clone();
            }
            "model_structure" => {
                self.model_structure = event.clone();
            }
            "tensor_edit" => {
                self.tensor_activity = event.clone();
                self.layer_offset = None;
            }
            "answer" => {
                let mut example = event["example"].clone();
                if example.is_object() && event["truncated_fields"].is_array() {
                    example["truncated_fields"] = event["truncated_fields"].clone();
                }
                if example.is_object() {
                    let label = field(event, "label");
                    let turn = event["turn"].as_u64();
                    let id = example["id"].clone();
                    let answer = Answer {
                        label: label.clone(),
                        phase: field(event, "phase"),
                        turn,
                        current: event["current"].as_u64(),
                        total: event["total"].as_u64(),
                        example,
                        receipt: event["receipt"].clone(),
                    };
                    if let Some(index) = self.answers.iter().position(|row| {
                        !id.is_null()
                            && row.label == label
                            && row.turn == turn
                            && row.example["id"] == id
                    }) {
                        self.answers[index] = answer;
                    } else {
                        self.answers.push(answer);
                    }
                    self.follow_item(DetailPane::Answers);
                }
            }
            "answer_score" => {
                for answer in self.answers.iter_mut().filter(|answer| {
                    answer.label == event["label"].as_str().unwrap_or("")
                        && answer.example["id"] == event["example_id"]
                }) {
                    if event["semantic"]
                        .as_f64()
                        .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
                    {
                        answer.example["semantic"] = event["semantic"].clone();
                        answer.example["judgement"] = event["judgement"].clone();
                    }
                }
            }
            "proposal" => {
                self.record_change(event, None);
            }
            "editor_activity" => {
                self.record_editor_activity(event);
            }
            "turn" => {
                self.turn = event["turn"].as_u64().unwrap_or(self.turn);
                self.max_turns = event["max_turns"].as_u64().unwrap_or(self.max_turns);
                let accepted = event["accepted"].as_bool().unwrap_or(false);
                self.record_change(event, Some(accepted));
                self.tensor_activity = Value::Null;
                let metrics = event["metrics"].clone();
                self.turns.push(Turn {
                    number: self.turn,
                    accepted,
                    score: score(&metrics, "validation", &self.metric),
                    loss: score(&metrics, "validation", "loss"),
                    health: event["health"].clone(),
                });
                if !event["health"].is_null() {
                    self.health_current = event["health"].clone();
                }
                if accepted {
                    self.retained_turn = self.turn;
                    self.best = metrics;
                    if !event["health"].is_null() {
                        self.health_best = event["health"].clone();
                    }
                }
                if !event["best_health"].is_null() {
                    self.health_best = event["best_health"].clone();
                }
                self.current = 0;
                self.total = 0;
                self.progress_known = false;
                self.phase = "Evaluated".into();
                self.message = if accepted {
                    format!("Turn {} retained", self.turn)
                } else {
                    format!("Turn {} reverted", self.turn)
                };
            }
            "cost" => {
                self.spent_usd = finite_number(event, "spent_usd").unwrap_or(self.spent_usd);
                self.budget_usd = finite_number(event, "budget_usd").unwrap_or(self.budget_usd);
                self.input_tokens = event["input_tokens"].as_u64().unwrap_or(self.input_tokens);
                self.output_tokens = event["output_tokens"]
                    .as_u64()
                    .unwrap_or(self.output_tokens);
            }
            "complete" => {
                self.done = true;
                self.current = 0;
                self.total = 0;
                self.progress_known = false;
                self.phase = "Complete".into();
                self.message = field(event, "message");
                if !event["metrics"].is_null() {
                    self.best = event["metrics"].clone();
                }
                if !event["health"].is_null() {
                    self.health_best = event["health"].clone();
                }
                if let Some(output) = event["output"].as_str() {
                    self.output = clean(output);
                }
            }
            "error" => {
                self.done = true;
                self.failed = true;
                self.phase = if self.cancelling {
                    "Stopped"
                } else {
                    "Stopped with an error"
                }
                .into();
                self.message = field(event, "message");
            }
            _ => {}
        }
    }
}

/// Runs on the UI thread. The coordinator owns inference and observes `cancel`.
pub fn run(rx: Receiver<Value>, cancel: Arc<AtomicBool>) -> Result<()> {
    dashboard(rx, cancel, false)
}

pub fn replay(rx: Receiver<Value>, cancel: Arc<AtomicBool>) -> Result<()> {
    dashboard(rx, cancel, true)
}

fn dashboard(rx: Receiver<Value>, cancel: Arc<AtomicBool>, browse_on_finish: bool) -> Result<()> {
    enable_raw_mode()?;
    let _guard = TerminalGuard;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal: Terminal<CrosstermBackend<Stdout>> = Terminal::new(backend)?;
    terminal.clear()?;
    let mut state = Dashboard::default();
    loop {
        let mut disconnected = false;
        for _ in 0..128 {
            match rx.try_recv() {
                Ok(message) => state.apply(&message),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        terminal.draw(|frame| render(frame, &state))?;
        if (state.done || disconnected) && !browse_on_finish {
            return Ok(());
        }
        if disconnected {
            state.done = true;
        }
        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(key) = event::read()? {
                if state.key(key) {
                    if state.done || browse_on_finish {
                        return Ok(());
                    }
                    cancel.store(true, Ordering::SeqCst);
                    state.cancelling = true;
                    state.phase = "Stopping".into();
                    state.message = "Finishing the current operation".into();
                }
            }
        }
    }
}

fn render(frame: &mut Frame, state: &Dashboard) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(INK).fg(IVORY)),
        area,
    );
    if area.width < 32 || area.height < 10 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::styled("VESPASIAN", Style::default().fg(COPPER)),
                Line::raw(state.phase.clone()),
                Line::raw(format!(
                    "{} / {}",
                    money(state.spent_usd),
                    money(state.budget_usd)
                )),
                Line::raw("Enlarge terminal · q stop"),
            ]),
            area,
        );
        return;
    }
    let wide = area.width >= 90 && area.height >= 24;
    let body = Rect::new(
        area.x + 1,
        area.y,
        area.width.saturating_sub(2),
        area.height,
    );
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(if wide { 3 } else { 2 }),
        Constraint::Min(0),
        Constraint::Length(2),
    ])
    .split(body);
    render_header(frame, rows[0], state, wide, false);
    render_goal(frame, rows[1], state, false);
    render_telemetry(frame, rows[2], state);
    if state.expanded {
        render_detail(frame, rows[3], state);
    } else if wide {
        let cols =
            Layout::horizontal([Constraint::Percentage(65), Constraint::Min(0)]).split(rows[3]);
        render_graphs(frame, cols[0], state);
        render_detail(frame, cols[1], state);
    } else {
        let cols = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(rows[3]);
        render_graphs(frame, cols[0], state);
        let count = state.item_count(state.detail_pane);
        frame.render_widget(
            Paragraph::new(format!(
                "{} · {}{} · Enter expand",
                state.detail_pane.label(),
                if count > 0 {
                    format!(
                        "{}/{}",
                        state.detail_indices[state.detail_pane.index()] + 1,
                        count
                    )
                } else {
                    "waiting".into()
                },
                if state.follow_latest[state.detail_pane.index()] {
                    " · latest"
                } else {
                    ""
                }
            ))
            .style(Style::default().fg(COPPER)),
            cols[1],
        );
    }
    render_footer(frame, rows[4], state);
}

fn render_telemetry(frame: &mut Frame, area: Rect, state: &Dashboard) {
    let tall = area.height >= 3;
    let phase = format!("{}  ·  {}", state.phase, state.message);
    if tall {
        frame.render_widget(
            Paragraph::new(phase).style(Style::default().fg(MUTED)),
            Rect::new(area.x, area.y, area.width, 1),
        );
    }
    let bars = Layout::horizontal([
        Constraint::Percentage(35),
        Constraint::Percentage(25),
        Constraint::Min(0),
    ])
    .spacing(1)
    .split(Rect::new(
        area.x,
        area.y + u16::from(tall),
        area.width,
        area.height - u16::from(tall),
    ));
    let progress_label = if state.phase.eq_ignore_ascii_case("editing") {
        "Tensors"
    } else {
        "Examples"
    };
    let progress = if state.progress_known {
        format!("{progress_label} {}/{}", state.current, state.total)
    } else {
        format!("{} · —", state.phase)
    };
    let turn_label = if state.max_turns > 0 {
        format!("Turns {}/{}", state.turn, state.max_turns)
    } else {
        format!("Turns {}", state.turn)
    };
    let estimated = if state.budget_mode.contains("estimated") {
        " est."
    } else {
        ""
    };
    let budget = format!(
        "{} / {}{}",
        money(state.spent_usd),
        money(state.budget_usd),
        estimated
    );
    for (index, label, value, color) in [
        (
            0,
            progress,
            state
                .progress_known
                .then(|| ratio(state.current as f64, state.total as f64)),
            COPPER,
        ),
        (
            1,
            turn_label,
            (state.max_turns > 0).then(|| ratio(state.turn as f64, state.max_turns as f64)),
            GREEN,
        ),
        (
            2,
            budget,
            (state.budget_usd > 0.0).then(|| ratio(state.spent_usd, state.budget_usd)),
            COPPER,
        ),
    ] {
        let row = bars[index];
        frame.render_widget(
            Paragraph::new(label).style(Style::default().fg(IVORY)),
            Rect::new(row.x, row.y, row.width, 1),
        );
        if row.height > 1 {
            let gauge = Rect::new(row.x, row.y + 1, row.width, 1);
            if let Some(value) = value {
                frame.render_widget(
                    Gauge::default()
                        .ratio(value)
                        .label("")
                        .gauge_style(Style::default().fg(color).bg(RULE)),
                    gauge,
                );
            } else {
                frame.render_widget(
                    Paragraph::new("─".repeat(row.width as usize)).style(Style::default().fg(RULE)),
                    gauge,
                );
            }
        }
    }
}

fn render_graphs(frame: &mut Frame, area: Rect, state: &Dashboard) {
    let layer_height = if state.model_structure.is_null() {
        0
    } else if area.height >= 15 {
        4
    } else {
        2
    };
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(layer_height)]).split(area);
    if selected_health(state).is_some() {
        let parts =
            Layout::vertical([Constraint::Percentage(65), Constraint::Min(0)]).split(rows[0]);
        render_chart(frame, parts[0], state);
        render_health_chart(frame, parts[1], state);
    } else {
        render_chart(frame, rows[0], state);
    }
    if layer_height > 0 {
        render_layers(frame, rows[1], state);
    }
}

fn render_detail(frame: &mut Frame, area: Rect, state: &Dashboard) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let pane = state.detail_pane.index();
    let mut spans = Vec::new();
    for item in [
        DetailPane::Answers,
        DetailPane::Editor,
        DetailPane::Genealogy,
    ] {
        spans.push(Span::styled(
            format!(" {} ", item.label()),
            Style::default()
                .fg(if item == state.detail_pane {
                    COPPER
                } else {
                    MUTED
                })
                .add_modifier(if item == state.detail_pane {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ));
    }
    let count = state.item_count(state.detail_pane);
    let position = if count > 0 {
        format!("{}/{}", state.detail_indices[pane] + 1, count)
    } else {
        "waiting".into()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(RULE))
        .title(Line::from(spans));
    let inner = block.inner(area);
    let viewport = Rect::new(
        inner.x,
        inner.y + u16::from(inner.height > 0),
        inner.width,
        inner.height.saturating_sub(1),
    );
    state.detail_geometry.set((viewport.width, viewport.height));
    let lines = detail_lines(state, viewport.width);
    let scroll =
        state.detail_scroll[pane].min(lines.len().saturating_sub(viewport.height as usize));
    let bottom = format!(
        " {}–{} / {} · {} ",
        if lines.is_empty() { 0 } else { scroll + 1 },
        (scroll + viewport.height as usize).min(lines.len()),
        lines.len(),
        if state.expanded {
            "Esc return"
        } else {
            "Enter expand"
        }
    );
    frame.render_widget(
        block.title_bottom(Line::styled(bottom, Style::default().fg(MUTED))),
        area,
    );
    frame.render_widget(
        Paragraph::new(format!(
            "{position}  ·  {}",
            if state.follow_latest[pane] {
                "following latest"
            } else {
                "browsing history"
            }
        ))
        .style(Style::default().fg(MUTED)),
        Rect::new(inner.x, inner.y, inner.width, u16::from(inner.height > 0)),
    );
    frame.render_widget(
        Paragraph::new(
            lines
                .into_iter()
                .skip(scroll)
                .take(viewport.height as usize)
                .collect::<Vec<_>>(),
        ),
        viewport,
    );
}

fn layer_number(name: &str) -> Option<usize> {
    let (_, suffix) = name
        .split_once(".layers.")
        .or_else(|| name.strip_prefix("layers.").map(|suffix| ("", suffix)))?;
    suffix.split('.').next()?.parse().ok()
}

fn target_layers(proposal: &Value) -> std::collections::BTreeSet<usize> {
    proposal["targets"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|target| target.as_str().and_then(layer_number))
        .collect()
}

fn layer_states(
    state: &Dashboard,
) -> (
    std::collections::BTreeSet<usize>,
    std::collections::BTreeSet<usize>,
    Option<usize>,
    std::collections::BTreeSet<usize>,
) {
    let retained = state
        .changes
        .iter()
        .filter(|change| change.accepted == Some(true))
        .flat_map(|change| target_layers(&change.proposal))
        .collect();
    let queued = state
        .changes
        .last()
        .filter(|change| change.accepted.is_none())
        .map(|change| target_layers(&change.proposal))
        .unwrap_or_default();
    let active = (state.tensor_activity["status"] == "started")
        .then(|| {
            state.tensor_activity["tensor"]
                .as_str()
                .and_then(layer_number)
        })
        .flatten();
    let selected = if state.detail_pane == DetailPane::Genealogy {
        state.detail_indices[2]
            .checked_sub(1)
            .and_then(|index| state.changes.get(index))
            .map(|change| target_layers(&change.proposal))
            .unwrap_or_default()
    } else {
        std::collections::BTreeSet::new()
    };
    (retained, queued, active, selected)
}

fn render_layers(frame: &mut Frame, area: Rect, state: &Dashboard) {
    let count = state.model_structure["num_layers"]
        .as_u64()
        .filter(|count| *count > 0 && *count <= 4096)
        .map(|count| count as usize);
    let (retained, queued, active, selected) = layer_states(state);
    let heading = format!(
        "LAYERS{}",
        count.map_or_else(String::new, |count| format!(" · {count}"))
    );
    let mut lines = vec![Line::from(vec![
        Span::styled(heading, Style::default().fg(MUTED)),
        Span::styled("  ▪ active", Style::default().fg(IVORY)),
        Span::styled(" ▪ proposed", Style::default().fg(COPPER)),
        Span::styled(" ▪ kept", Style::default().fg(GREEN)),
        Span::styled(
            " ▪ selected",
            Style::default().fg(Color::Rgb(164, 181, 218)),
        ),
    ])];
    if let Some(count) = count {
        let cell_width = if count > 100 { 4 } else { 3 };
        let capacity = (area.width as usize / cell_width).max(1);
        let rows = area.height.saturating_sub(1).max(1) as usize;
        let visible = capacity * rows;
        let focus = active
            .or_else(|| selected.first().copied())
            .or_else(|| queued.first().copied())
            .or_else(|| retained.first().copied())
            .unwrap_or(0)
            .min(count - 1);
        let start = if count > visible {
            (focus / capacity).saturating_sub(rows / 2) * capacity
        } else {
            0
        };
        let start = state
            .layer_offset
            .unwrap_or(start)
            .min(count.saturating_sub(visible));
        state.layer_geometry.set((start, visible));
        let end = (start + visible).min(count);
        for chunk in (start..end).collect::<Vec<_>>().chunks(capacity) {
            let spans = chunk
                .iter()
                .map(|layer| {
                    let color = if active == Some(*layer) {
                        IVORY
                    } else if selected.contains(layer) {
                        Color::Rgb(164, 181, 218)
                    } else if queued.contains(layer) {
                        COPPER
                    } else if retained.contains(layer) {
                        GREEN
                    } else {
                        MUTED
                    };
                    Span::styled(
                        format!("{layer:02} "),
                        Style::default()
                            .fg(color)
                            .add_modifier(if active == Some(*layer) {
                                Modifier::BOLD | Modifier::REVERSED
                            } else if selected.contains(layer) {
                                Modifier::UNDERLINED
                            } else {
                                Modifier::empty()
                            }),
                    )
                })
                .collect::<Vec<_>>();
            lines.push(Line::from(spans));
        }
        if lines.len() < area.height as usize && !state.tensor_activity.is_null() {
            lines.push(Line::styled(
                format!(
                    "{} · {}",
                    title(&field(&state.tensor_activity, "status")),
                    field(&state.tensor_activity, "tensor")
                ),
                Style::default().fg(IVORY),
            ));
        }
        if count > visible {
            lines[0] = Line::from(vec![
                Span::styled(
                    format!("LAYERS {start:02}–{:02}/{count}", end - 1),
                    Style::default().fg(MUTED),
                ),
                Span::styled(" active", Style::default().fg(IVORY)),
                Span::styled(" proposed", Style::default().fg(COPPER)),
                Span::styled(" kept", Style::default().fg(GREEN)),
                Span::styled(" selected", Style::default().fg(Color::Rgb(164, 181, 218))),
            ]);
        }
    } else {
        let names = state.model_structure["editable"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|tensor| tensor["name"].as_str())
            .collect::<Vec<_>>();
        lines[0] = Line::styled(
            format!(
                "TENSORS · {} available · layer count unavailable",
                names.len()
            ),
            Style::default().fg(MUTED),
        );
        lines.extend(
            wrap_detail(&names.join(" · "), area.width, MUTED)
                .into_iter()
                .take(area.height.saturating_sub(1) as usize),
        );
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn clean_text(text: &str) -> String {
    text.lines().map(clean).collect::<Vec<_>>().join("\n")
}

fn wrap_detail(text: &str, width: u16, color: Color) -> Vec<Line<'static>> {
    let width = usize::from(width.max(1));
    let mut output = Vec::new();
    for logical in clean_text(text).split('\n') {
        if logical.is_empty() {
            output.push(Line::raw(""));
            continue;
        }
        let mut line = String::new();
        for word in logical.split_inclusive(char::is_whitespace) {
            if !line.is_empty() && Line::raw(format!("{line}{word}")).width() > width {
                output.push(Line::styled(
                    line.trim_end().to_owned(),
                    Style::default().fg(color),
                ));
                line.clear();
            }
            for ch in word.chars() {
                if Line::raw(format!("{line}{ch}")).width() > width && !line.is_empty() {
                    output.push(Line::styled(
                        std::mem::take(&mut line),
                        Style::default().fg(color),
                    ));
                }
                line.push(ch);
            }
        }
        if !line.is_empty() {
            output.push(Line::styled(
                line.trim_end().to_owned(),
                Style::default().fg(color),
            ));
        }
    }
    output
}

fn detail_section(lines: &mut Vec<Line<'static>>, title: &str, text: &str, width: u16) {
    if text.is_empty() {
        return;
    }
    if !lines.is_empty() {
        lines.push(Line::raw(""));
    }
    lines.extend(wrap_detail(title, width, COPPER));
    lines.extend(wrap_detail(text, width, IVORY));
}

fn value_text(value: &Value) -> String {
    value.as_str().map(clean_text).unwrap_or_else(|| {
        if value.is_null() {
            String::new()
        } else {
            clean_text(&serde_json::to_string_pretty(value).unwrap_or_default())
        }
    })
}

fn detail_lines(state: &Dashboard, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match state.detail_pane {
        DetailPane::Answers => {
            let Some(answer) = state.answers.get(state.detail_indices[0]) else {
                return wrap_detail("Generated answers appear as each example finishes. Use ← → to browse examples and Enter to expand.",width,MUTED);
            };
            let example = &answer.example;
            let mut label = answer.label.clone();
            if let Some(turn) = answer.turn {
                label.push_str(&format!(" · turn {turn}"));
            }
            if let (Some(current), Some(total)) = (answer.current, answer.total) {
                label.push_str(&format!(" · {current}/{total}"));
            }
            lines.extend(wrap_detail(&label, width, COPPER));
            lines.extend(wrap_detail(
                &[
                    title(&answer.phase),
                    field(example, "split"),
                    field(example, "check"),
                ]
                .into_iter()
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
                .join(" · "),
                width,
                MUTED,
            ));
            let mut measures = Vec::new();
            if let Some(passed) = example["passed"].as_bool() {
                measures.push(format!(
                    "{} · {}",
                    if passed { "PASS" } else { "FAIL" },
                    field(example, "scoring").replace('_', " ")
                ));
            }
            if state.metric == "numeric_match" {
                if let Some(numeric) = example["numeric_match"].as_f64().filter(|v| v.is_finite()) {
                    measures.push(format!("Final answer {:.0}%", numeric * 100.0));
                }
            } else {
                if let Some(exact) = example["exact_match"].as_bool() {
                    measures.push(format!("Exact {}", if exact { "match" } else { "miss" }));
                } else if let Some(exact) = example["exact_match"].as_f64() {
                    measures.push(format!("Exact {:.0}%", exact * 100.0));
                }
                if let Some(f1) = example["token_f1"].as_f64().filter(|v| v.is_finite()) {
                    measures.push(format!("F1 {:.1}%", f1 * 100.0));
                }
            }
            if let Some(loss) = example["loss"].as_f64().filter(|v| v.is_finite()) {
                measures.push(format!("Loss {loss:.4}"));
            }
            lines.extend(wrap_detail(&measures.join(" · "), width, MUTED));
            if let Some(semantic) = example["semantic"].as_f64() {
                detail_section(
                    &mut lines,
                    "GOAL SCORE",
                    &format!(
                        "{:.1}%\n{}",
                        semantic * 100.0,
                        value_text(&example["judgement"])
                    ),
                    width,
                );
            }
            if state.metric == "semantic"
                && example["semantic"].is_null()
                && answer.phase != "health"
            {
                lines.extend(wrap_detail("Goal score pending", width, MUTED));
            }
            if example["mode"] == "corpus" {
                detail_section(&mut lines, "GENERATION MODE", "Unconditional text", width);
                detail_section(
                    &mut lines,
                    "STYLE EXAMPLE",
                    &value_text(&example["target"]),
                    width,
                );
            } else {
                detail_section(&mut lines, "PROMPT", &value_text(&example["prompt"]), width);
                if example["mode"] != "prompts" {
                    detail_section(
                        &mut lines,
                        "REFERENCE",
                        &value_text(&example["target"]),
                        width,
                    );
                }
            }
            let output = example
                .get("output")
                .or_else(|| example.get("prediction"))
                .unwrap_or(&Value::Null);
            detail_section(&mut lines, "GENERATED", &value_text(output), width);
            if !example["finish_reason"].is_null() {
                detail_section(
                    &mut lines,
                    "GENERATION",
                    &format!(
                        "{} · {} tokens",
                        field(example, "finish_reason"),
                        example["token_ids"]
                            .as_array()
                            .map_or_else(|| "—".into(), |ids| ids.len().to_string())
                    ),
                    width,
                );
            }
            if example["truncated_fields"]
                .as_array()
                .is_some_and(|fields| !fields.is_empty())
            {
                detail_section(
                    &mut lines,
                    "FULL OUTPUT",
                    &format!(
                        "Display shortened: {}\nFull output after evaluation: {}{} [{}]",
                        value_text(&example["truncated_fields"]),
                        field(&answer.receipt, "file"),
                        field(&answer.receipt, "pointer"),
                        answer.receipt["index"]
                    ),
                    width,
                );
            }
        }
        DetailPane::Editor => {
            if let Some(call) = state.editor_calls.get(state.detail_indices[1]) {
                lines.extend(wrap_detail(
                    &format!("{} · {}", title(&call.provider), call.model),
                    width,
                    COPPER,
                ));
                lines.extend(wrap_detail(
                    &format!("{} · {}", title(&call.purpose), title(&call.status)),
                    width,
                    MUTED,
                ));
                if let Some(turn) = call.turn {
                    lines.extend(wrap_detail(&format!("Turn {turn}"), width, MUTED));
                }
                for update in &call.updates {
                    let heading = format!(
                        "{}{}",
                        title(&update.status),
                        update.elapsed_ms.map_or_else(String::new, |ms| format!(
                            " · {:.1}s",
                            ms as f64 / 1000.0
                        ))
                    );
                    detail_section(&mut lines, &heading, &update.text, width);
                }
                let change = call
                    .turn
                    .and_then(|turn| state.changes.iter().find(|change| change.turn == turn));
                if let Some(change) = change {
                    detail_section(&mut lines, "EDIT RATIONALE", &change.rationale, width);
                }
            } else {
                lines.extend(wrap_detail("Waiting for editor activity", width, MUTED));
            }
            if state.input_tokens > 0 || state.output_tokens > 0 {
                detail_section(
                    &mut lines,
                    "SESSION USAGE",
                    &format!(
                        "{} input · {} output tokens\n{} / {} inference",
                        tokens(state.input_tokens),
                        tokens(state.output_tokens),
                        money(state.spent_usd),
                        money(state.budget_usd)
                    ),
                    width,
                );
            }
            detail_section(
                &mut lines,
                "DATASET ASSESSMENT",
                &state.assessment_reason,
                width,
            );
            detail_section(&mut lines, "EVALUATION PLAN", &state.plan, width);
        }
        DetailPane::Genealogy => {
            let selected = state.detail_indices[2];
            lines.extend(wrap_detail(
                &format!("{}● BASELINE", if selected == 0 { "› " } else { "  " }),
                width,
                if selected == 0 { COPPER } else { MUTED },
            ));
            for (index, change) in state.changes.iter().enumerate() {
                let (decision, color) = match change.accepted {
                    Some(true) => ("RETAINED", GREEN),
                    Some(false) => ("REVERTED", RED),
                    None => ("PROPOSED", COPPER),
                };
                let depth =
                    change_depth(state, change.turn).min((width as usize).saturating_sub(24) / 2);
                let branch = if change.accepted == Some(true) {
                    "└─"
                } else {
                    "├─"
                };
                lines.extend(wrap_detail(
                    &format!(
                        "{}{}{branch} {:02} ← {:02} {decision}",
                        if selected == index + 1 { "›" } else { " " },
                        "  ".repeat(depth),
                        change.turn,
                        change.parent
                    ),
                    width,
                    color,
                ));
            }
            if selected == 0 {
                detail_section(&mut lines, "ROOT CHECKPOINT", &state.model, width);
                detail_section(
                    &mut lines,
                    "VALIDATION",
                    &format!(
                        "{} {}",
                        metric_label(&state.metric),
                        percent(score(&state.baseline, "validation", &state.metric))
                    ),
                    width,
                );
                detail_section(
                    &mut lines,
                    "MODEL STRUCTURE",
                    &format!(
                        "{} · {} layers",
                        field(&state.model_structure, "architecture"),
                        state.model_structure["num_layers"]
                            .as_u64()
                            .map_or_else(|| "unknown".into(), |count| count.to_string())
                    ),
                    width,
                );
                detail_section(
                    &mut lines,
                    "AVAILABLE TENSORS",
                    &value_text(&state.model_structure["editable"]),
                    width,
                );
                if let Some(event) =
                    state.loop_stages.iter().rev().find(|event| {
                        event["turn"] == 0 && event["details"]["behavior"].is_object()
                    })
                {
                    detail_section(
                        &mut lines,
                        "OBSERVED BEHAVIOR (NOT ACCEPTANCE SCORES)",
                        &behavior_text(&event["details"]["behavior"]),
                        width,
                    );
                }
            } else if let Some(change) = state.changes.get(selected - 1) {
                detail_section(
                    &mut lines,
                    &format!("TURN {:02} · PARENT {:02}", change.turn, change.parent),
                    &format!(
                        "{} · strength {}",
                        field(&change.proposal, "operation"),
                        value_text(&change.proposal["strength"])
                    ),
                    width,
                );
                detail_section(&mut lines, "DECISION", &change.decision_reason, width);
                for (key, title) in [
                    ("strategy", "STRATEGY"),
                    ("plan", "APPLICATION PLAN"),
                    ("reflection", "REFLECTION ON MEASURED OUTCOMES"),
                ] {
                    if let Some(text) = change.proposal[key].as_str() {
                        detail_section(&mut lines, title, text, width);
                    }
                }
                detail_section(&mut lines, "EDITOR RATIONALE", &change.rationale, width);
                for event in state
                    .loop_stages
                    .iter()
                    .filter(|event| event["turn"].as_u64() == Some(change.turn))
                {
                    detail_section(
                        &mut lines,
                        &format!(
                            "{} · {}",
                            field(event, "stage").to_uppercase(),
                            field(event, "status")
                        ),
                        &field(event, "summary"),
                        width,
                    );
                    if event["details"]["behavior"].is_object() {
                        detail_section(
                            &mut lines,
                            "OBSERVED BEHAVIOR (NOT ACCEPTANCE SCORES)",
                            &behavior_text(&event["details"]["behavior"]),
                            width,
                        );
                    }
                    if event["stage"] == "inspect" && event["status"] == "completed" {
                        detail_section(
                            &mut lines,
                            "OBSERVED TOKEN ROWS",
                            &value_text(&event["details"]["inspection"]),
                            width,
                        );
                    }
                }
                if !change.proposal["token_ids"].is_null() {
                    detail_section(
                        &mut lines,
                        "TOKEN ROW SELECTION",
                        &format!(
                            "destinations {} · sources {}",
                            change.proposal["token_ids"], change.proposal["source_token_ids"]
                        ),
                        width,
                    );
                }
                let tensors = change.proposal["targets"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n");
                detail_section(&mut lines, "TENSORS", &tensors, width);
                let mut measured = Vec::new();
                for tensor in change.mutation["changed_tensors"]
                    .as_array()
                    .into_iter()
                    .flatten()
                {
                    measured.push(format!(
                        "{}\n{} elements · relative norm {}\nshape {}",
                        field(tensor, "name"),
                        value_text(&tensor["changed_elements"]),
                        tensor["relative_norm"]
                            .as_f64()
                            .map_or_else(|| "—".into(), |value| format!("{value:.6}")),
                        value_text(&tensor["shape"])
                    ));
                }
                detail_section(&mut lines, "MEASURED CHANGE", &measured.join("\n\n"), width);
                detail_section(
                    &mut lines,
                    "EDIT PARAMETERS",
                    &format!(
                        "Rank {} · seed {}",
                        value_text(&change.proposal["rank"]),
                        value_text(&change.proposal["seed"])
                    ),
                    width,
                );
                if !change.metrics.is_null() {
                    detail_section(
                        &mut lines,
                        "VALIDATION",
                        &format!(
                            "{} {}",
                            metric_label(&state.metric),
                            percent(score(&change.metrics, "validation", &state.metric))
                        ),
                        width,
                    );
                }
            }
        }
    }
    lines
}

fn change_depth(state: &Dashboard, turn: u64) -> usize {
    let mut parent = turn;
    let mut depth: usize = 0;
    for _ in 0..state.changes.len() {
        let Some(change) = state.changes.iter().find(|change| change.turn == parent) else {
            break;
        };
        if change.parent >= parent {
            break;
        }
        depth += 1;
        parent = change.parent;
    }
    depth.saturating_sub(1)
}

fn health_ids(state: &Dashboard) -> Vec<&str> {
    state
        .health_baseline
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, value)| health_score(value).is_some())
        .map(|(key, _)| key.as_str())
        .collect()
}

fn health_score(value: &Value) -> Option<f64> {
    value["score"]
        .as_f64()
        .filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
}

fn selected_health(state: &Dashboard) -> Option<&str> {
    let ids = health_ids(state);
    (!ids.is_empty()).then(|| ids[state.health_index % ids.len()])
}

fn health_label(state: &Dashboard, id: &str) -> String {
    state.health_baseline[id]["label"]
        .as_str()
        .map(clean)
        .unwrap_or_else(|| clean(id))
}

fn health_trajectory(state: &Dashboard, id: &str) -> Trajectory {
    let Some(baseline) = health_score(&state.health_baseline[id]) else {
        return Trajectory::default();
    };
    let mut current = baseline * 100.0;
    let mut path = Trajectory {
        incumbent: vec![(0.0, current)],
        ..Default::default()
    };
    for turn in &state.turns {
        if let Some(value) = health_score(&turn.health[id]) {
            let point = (turn.number as f64, value * 100.0);
            if turn.accepted {
                current = point.1;
                path.retained.push(point);
            } else {
                path.reverted.push(point);
            }
        }
        path.incumbent.push((turn.number as f64, current));
    }
    path
}

fn render_health_chart(frame: &mut Frame, area: Rect, state: &Dashboard) {
    if area.height == 0 {
        return;
    }
    let Some(id) = selected_health(state) else {
        return;
    };
    let label = health_label(state, id);
    let label: String = label
        .chars()
        .take(area.width.saturating_sub(13) as usize)
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(RULE))
        .title(Line::styled(
            format!(" Health · {label} "),
            Style::default().fg(GREEN),
        ))
        .title_bottom(Line::from(vec![
            Span::styled(
                format!(
                    " Base {} ",
                    percent(health_score(&state.health_baseline[id]))
                ),
                Style::default().fg(MUTED),
            ),
            Span::styled(
                format!("Kept {} ", percent(health_score(&state.health_best[id]))),
                Style::default().fg(GREEN),
            ),
            Span::styled(
                format!(
                    "Trial {} ",
                    percent(health_score(&state.health_current[id]))
                ),
                Style::default().fg(RED),
            ),
            Span::styled(
                health_score(&state.health_best[id])
                    .zip(health_score(&state.health_baseline[id]))
                    .map_or_else(String::new, |(after, before)| {
                        format!("Δ {:+.1}pp ", (after - before) * 100.0)
                    }),
                Style::default().fg(MUTED),
            ),
        ]));
    let path = health_trajectory(state, id);
    if path.incumbent.is_empty() {
        frame.render_widget(
            Paragraph::new("Awaiting health measurements").block(block),
            area,
        );
        return;
    }
    let last = state.turn.max(1) as f64;
    let reference = [(0.0, path.incumbent[0].1), (last, path.incumbent[0].1)];
    let datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(RULE))
            .data(&reference),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(GREEN))
            .data(&path.incumbent),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(COPPER))
            .data(&path.retained),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(RED))
            .data(&path.reverted),
    ];
    let y_labels = if area.height < 8 {
        vec![Line::from("0%"), Line::from("100%")]
    } else {
        vec![Line::from("0%"), Line::from("50%"), Line::from("100%")]
    };
    let mut x_axis = Axis::default()
        .style(Style::default().fg(RULE))
        .bounds([0.0, last]);
    let mut y_axis = Axis::default()
        .style(Style::default().fg(RULE))
        .bounds([0.0, 100.0]);
    if area.height >= 8 {
        x_axis = x_axis
            .title("turn")
            .labels([Line::from("0"), Line::from(state.turn.max(1).to_string())]);
        y_axis = y_axis.labels(y_labels);
    }
    frame.render_widget(
        Chart::new(datasets)
            .block(block)
            .x_axis(x_axis)
            .y_axis(y_axis)
            .style(Style::default().fg(MUTED)),
        area,
    );
}

fn render_header(frame: &mut Frame, area: Rect, state: &Dashboard, wide: bool, roomy: bool) {
    let halves = Layout::horizontal([Constraint::Min(0), Constraint::Length(18)]).split(area);
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "VESPASIAN",
            Style::default().fg(COPPER).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            if wide { "   /   WEIGHT EDITING" } else { "" },
            Style::default().fg(MUTED),
        ),
    ])];
    if roomy {
        lines.push(Line::from(""));
    }
    lines.push(Line::styled(
        if state.model.is_empty() {
            "Preparing model"
        } else {
            &state.model
        },
        Style::default().fg(IVORY),
    ));
    frame.render_widget(Paragraph::new(lines), halves[0]);
    let elapsed = state.started.elapsed().as_secs();
    let timing = if state.done {
        format!("{} turns", state.turn)
    } else {
        format!("{:02}:{:02}", elapsed / 60, elapsed % 60)
    };
    let phase = if state.done {
        if state.failed {
            "STOPPED"
        } else {
            "COMPLETE"
        }
    } else if state.cancelling {
        "STOPPING"
    } else {
        "IN PROGRESS"
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(
                phase,
                Style::default().fg(if state.failed { RED } else { COPPER }),
            ),
            Line::styled(timing, Style::default().fg(MUTED)),
        ])
        .alignment(ratatui::layout::Alignment::Right),
        halves[1],
    );
    if area.height >= 3 {
        frame.render_widget(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(RULE)),
            area,
        );
    }
}

fn render_goal(frame: &mut Frame, area: Rect, state: &Dashboard, roomy: bool) {
    let mut lines = vec![Line::from(vec![
        Span::styled("GOAL  ", Style::default().fg(COPPER)),
        Span::raw(if state.goal.is_empty() {
            "—"
        } else {
            &state.goal
        }),
    ])];
    if roomy && !state.plan.is_empty() {
        lines.push(Line::styled(
            state.plan.as_str(),
            Style::default().fg(MUTED),
        ));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
}

#[derive(Default)]
struct Trajectory {
    incumbent: Vec<(f64, f64)>,
    retained: Vec<(f64, f64)>,
    reverted: Vec<(f64, f64)>,
}

fn trajectory(state: &Dashboard) -> Trajectory {
    let metric = if state.plot_loss {
        "loss"
    } else {
        &state.metric
    };
    let scale = if state.plot_loss { 1.0 } else { 100.0 };
    let Some(baseline) = score(&state.baseline, "validation", metric) else {
        return Trajectory::default();
    };
    let mut current = baseline * scale;
    let mut incumbent = vec![(0.0, current)];
    let mut retained = Vec::new();
    let mut reverted = Vec::new();
    for turn in &state.turns {
        if let Some(value) = if state.plot_loss {
            turn.loss
        } else {
            turn.score
        } {
            let point = (turn.number as f64, value * scale);
            if turn.accepted {
                current = point.1;
                retained.push(point);
            } else {
                reverted.push(point);
            }
        }
        incumbent.push((turn.number as f64, current));
    }
    Trajectory {
        incumbent,
        retained,
        reverted,
    }
}

fn render_chart(frame: &mut Frame, area: Rect, state: &Dashboard) {
    if area.height == 0 {
        return;
    }
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(RULE))
        .title(Line::from(vec![
            Span::styled(
                " Hill climb ",
                Style::default().fg(COPPER).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if state.plot_loss {
                    "· Reference loss ↓ ".into()
                } else {
                    format!("· {} ↑ ", metric_label(&state.metric))
                },
                Style::default().fg(MUTED),
            ),
        ]));
    let metric = if state.plot_loss {
        "loss"
    } else {
        &state.metric
    };
    let format_value = |value: Option<f64>| {
        if state.plot_loss {
            value.map_or_else(|| "—".into(), |value| format!("{value:.4}"))
        } else {
            percent(value)
        }
    };
    let before = format_value(score(&state.baseline, "validation", metric));
    let after = format_value(score(&state.best, "validation", metric));
    block = block.title_bottom(Line::from(vec![
        Span::styled(format!(" {before} → {after} "), Style::default().fg(IVORY)),
        Span::styled("━ kept ", Style::default().fg(COPPER)),
        Span::styled("• trial ", Style::default().fg(RED)),
        Span::styled(
            if state.plot_loss {
                format!(
                    "· {} {} ",
                    metric_label(&state.metric),
                    percent(score(&state.best, "validation", &state.metric))
                )
            } else {
                String::new()
            },
            Style::default().fg(MUTED),
        ),
    ]));
    let Trajectory {
        incumbent,
        retained,
        reverted,
    } = trajectory(state);
    if incumbent.is_empty() {
        frame.render_widget(
            Paragraph::new(
                if state.plot_loss && state.baseline["validation"].is_object() {
                    "Reference loss is unavailable for this dataset. Press m for the objective."
                } else {
                    "The measured baseline starts the climb."
                },
            )
            .style(Style::default().fg(MUTED))
            .block(block),
            area,
        );
        return;
    }
    let last_turn = state.turn.max(1) as f64;
    let bounds = if state.plot_loss {
        let values = incumbent
            .iter()
            .chain(retained.iter())
            .chain(reverted.iter())
            .map(|(_, value)| *value);
        let (low, high) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
        let padding = ((high - low) * 0.12).max(1e-6);
        [(low - padding).max(0.0), high + padding]
    } else {
        [0.0, 100.0]
    };
    let y_labels: Vec<Line> = if state.plot_loss {
        [bounds[0], (bounds[0] + bounds[1]) / 2.0, bounds[1]]
            .map(|value| Line::styled(format!("{value:.4}"), Style::default().fg(MUTED)))
            .into()
    } else {
        ["0%", "50%", "100%"]
            .map(|value| Line::styled(value, Style::default().fg(MUTED)))
            .into()
    };
    let baseline = vec![(0.0, incumbent[0].1), (last_turn, incumbent[0].1)];
    let datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(RULE))
            .data(&baseline),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(COPPER))
            .data(&incumbent),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(GREEN))
            .data(&retained),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(RED))
            .data(&reverted),
    ];
    let x_labels = if state.turn >= 4 {
        vec![
            Line::styled("0", Style::default().fg(MUTED)),
            Line::styled(format!("{}", state.turn / 2), Style::default().fg(MUTED)),
            Line::styled(format!("{}", state.turn), Style::default().fg(MUTED)),
        ]
    } else {
        vec![
            Line::styled("0", Style::default().fg(MUTED)),
            Line::styled(format!("{}", state.turn.max(1)), Style::default().fg(MUTED)),
        ]
    };
    let mut x_axis = Axis::default()
        .style(Style::default().fg(RULE))
        .bounds([0.0, last_turn]);
    let mut y_axis = Axis::default()
        .style(Style::default().fg(RULE))
        .bounds(bounds);
    if area.height >= 8 {
        x_axis = x_axis.title("turn").labels(x_labels);
        y_axis = y_axis.labels(y_labels);
    }
    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(x_axis)
        .y_axis(y_axis)
        .style(Style::default().fg(MUTED));
    frame.render_widget(chart, area);
}

fn render_footer(frame: &mut Frame, area: Rect, state: &Dashboard) {
    let action = if state.done { "q close" } else { "q stop" };
    let help = if area.width >= 90 {
        format!(
            "Tab pane · ← → item · ↑ ↓ scroll · Enter expand · m score/loss · h health · l layers · {action}"
        )
    } else {
        format!("Tab pane · ← → item · ↑ ↓ scroll · Enter expand · {action}")
    };
    let second = if area.width >= 90 {
        format!(
            "OUTPUT  {}",
            tail(&state.output, area.width.saturating_sub(8) as usize)
        )
    } else {
        "m score/loss · h health · l layers · PgUp/PgDn · End latest".into()
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(help, Style::default().fg(MUTED)),
            Line::styled(second, Style::default().fg(MUTED)),
        ]),
        area,
    );
}

fn score(metrics: &Value, split: &str, metric: &str) -> Option<f64> {
    metrics[split][metric].as_f64().filter(|x| x.is_finite())
}

fn finite_number(value: &Value, key: &str) -> Option<f64> {
    value[key].as_f64().filter(|x| x.is_finite() && *x >= 0.0)
}

fn field(value: &Value, key: &str) -> String {
    clean(value[key].as_str().unwrap_or(""))
}

fn metric_label(metric: &str) -> &str {
    match metric {
        "token_f1" => "Token F1",
        "numeric_match" => "Final answer accuracy",
        "exact_match" => "Exact match",
        "semantic" => "Goal score",
        _ => metric,
    }
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |x| format!("{:.1}%", x * 100.0))
}

fn money(value: f64) -> String {
    if value == 0.0 || value >= 1.0 {
        format!("${value:.2}")
    } else {
        format!("${value:.4}")
    }
}

fn tokens(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 10_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 || !denominator.is_finite() || !numerator.is_finite() {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

fn title(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn tail(text: &str, count: usize) -> String {
    let length = text.chars().count();
    if length <= count {
        text.into()
    } else if count == 0 {
        String::new()
    } else {
        format!(
            "…{}",
            text.chars().skip(length - count + 1).collect::<String>()
        )
    }
}

/// Dataset and editor text are untrusted terminal content. Remove ANSI sequences,
/// C0/C1 controls, and bidi controls while preserving printable Unicode.
fn clean(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            match chars.next() {
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&c) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\u{7}' {
                            break;
                        }
                        if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                _ => {}
            }
        } else if ch == '\n' || ch == '\r' || ch == '\t' {
            output.push(' ');
        } else if !ch.is_control()
            && !matches!(ch, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        {
            output.push(ch);
        }
    }
    output
}

fn behavior_text(summary: &Value) -> String {
    let mut rows = vec![format!(
        "{} visible examples; sealed tests excluded.",
        summary["sample_count"].as_u64().unwrap_or(0)
    )];
    for (key, label) in [
        ("repetition", "Repetition flags"),
        ("length_truncation", "Output-limit endings"),
        ("empty_output", "Empty outputs"),
        ("reference_mismatch", "Reference mismatches"),
        ("format_only_mismatch", "Format-only mismatches"),
        ("preservation_mismatch", "Preservation mismatches"),
    ] {
        if let Some(count) = summary["signals"][key]["count"].as_u64() {
            rows.push(format!("{label}: {count}"));
        }
    }
    rows.push("Repetition and truncation are observations, not proof of incorrectness. Correctness and preservation still decide retention.".into());
    rows.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use serde_json::json;

    fn populated() -> Dashboard {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"start","model":"Qwen/Qwen3.5-0.8B","editor":"codex","goal":"Answer supplied facts accurately.","budget_usd":5.0,"max_turns":8,"output":"/tmp/run/checkpoint"}));
        state.apply(&json!({"type":"assessment","suitable":true,"reason":"Reference answers support the goal.","metric":"exact_match","plan":"Measure factual answers on unseen phrasings."}));
        state.apply(&json!({"type":"baseline","metrics":{"validation":{"exact_match":0.25,"loss":3.2,"count":4},"preserve":{"exact_match":1.0,"count":2}}}));
        state.apply(&json!({"type":"turn","turn":1,"max_turns":8,"accepted":true,"metrics":{"validation":{"exact_match":0.75,"count":4},"preserve":{"exact_match":1.0,"count":2}},"rationale":"Improved reference answers"}));
        state.apply(&json!({"type":"cost","spent_usd":0.1234,"budget_usd":5.0,"input_tokens":4500,"output_tokens":900}));
        state
    }

    fn snapshot(state: &Dashboard, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| render(frame, state)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn health_session() -> Dashboard {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"start","model":"Qwen/Qwen3.5-0.8B","editor":"codex","goal":"Answer the supplied questions.","budget_usd":2.0,"max_turns":4,"output":"/tmp/health-run"}));
        state.apply(&json!({"type":"baseline","metrics":{"validation":{"exact_match":0.25,"loss":3.2,"count":4}},"health":{"copying":{"label":"Copying","score":1.0,"count":10,"metric":"exact_copy"},"arithmetic":{"label":"Arithmetic","score":0.5,"count":10,"metric":"accuracy"}}}));
        state.apply(&json!({"type":"turn","turn":1,"accepted":true,"metrics":{"validation":{"exact_match":0.5,"loss":3.0,"count":4}},"health":{"copying":{"label":"Copying","score":0.9,"count":10,"metric":"exact_copy"},"arithmetic":{"label":"Arithmetic","score":0.6,"count":10,"metric":"accuracy"}}}));
        state.apply(&json!({"type":"turn","turn":2,"accepted":false,"metrics":{"validation":{"exact_match":0.75,"loss":2.0,"count":4}},"health":{"copying":{"label":"Copying","score":0.2,"count":10,"metric":"exact_copy"},"arithmetic":{"label":"Arithmetic","score":0.9,"count":10,"metric":"accuracy"}},"best_health":{"copying":{"label":"Copying","score":0.9,"count":10,"metric":"exact_copy"},"arithmetic":{"label":"Arithmetic","score":0.6,"count":10,"metric":"accuracy"}}}));
        state.health_index = health_ids(&state)
            .iter()
            .position(|id| *id == "copying")
            .unwrap();
        state
    }

    #[test]
    fn health_history_keeps_rejected_trials_separate_from_retained_weights() {
        let state = health_session();
        let path = health_trajectory(&state, "copying");
        assert_eq!(path.incumbent, vec![(0.0, 100.0), (1.0, 90.0), (2.0, 90.0)]);
        assert_eq!(path.retained, vec![(1.0, 90.0)]);
        assert_eq!(path.reverted, vec![(2.0, 20.0)]);
        assert_eq!(health_score(&state.health_current["copying"]), Some(0.2));
        assert_eq!(health_score(&state.health_best["copying"]), Some(0.9));
        assert_eq!(score(&state.best, "validation", "exact_match"), Some(0.5));
    }

    #[test]
    fn objective_and_health_remain_visible_at_both_terminal_sizes() {
        let state = health_session();
        for (width, height) in [(100, 32), (60, 18)] {
            let output = snapshot(&state, width, height);
            for expected in [
                "Hill climb",
                "Health",
                "Copying",
                "20.0%",
                "90.0%",
                "q stop",
            ] {
                assert!(
                    output.contains(expected),
                    "missing {expected} at {width}x{height}:\n{output}"
                );
            }
        }
    }

    #[test]
    fn equal_scores_share_one_plot_row_for_lines_and_trial_markers() {
        let mut state = health_session();
        state.baseline["validation"]["exact_match"] = json!(0.2);
        state.health_baseline["copying"]["score"] = json!(0.2);
        for turn in &mut state.turns {
            turn.score = Some(0.2);
            turn.health["copying"]["score"] = json!(0.2);
        }
        for (width, height) in [(48, 12), (28, 5)] {
            for health in [false, true] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal
                    .draw(|frame| {
                        if health {
                            render_health_chart(frame, frame.area(), &state);
                        } else {
                            render_chart(frame, frame.area(), &state);
                        }
                    })
                    .unwrap();
                let buffer = terminal.backend().buffer();
                let mut rows = std::collections::BTreeSet::new();
                let mut rejected_marker = false;
                for y in 0..height {
                    for x in 0..width {
                        let cell = &buffer[(x, y)];
                        let Some(symbol) = cell
                            .symbol()
                            .chars()
                            .next()
                            .filter(|symbol| ('\u{2801}'..='\u{28ff}').contains(symbol))
                        else {
                            continue;
                        };
                        let bits = symbol as u32 - 0x2800;
                        for (bit, row) in [
                            (0, 0),
                            (1, 1),
                            (2, 2),
                            (3, 0),
                            (4, 1),
                            (5, 2),
                            (6, 3),
                            (7, 3),
                        ] {
                            if bits & (1 << bit) != 0 {
                                rows.insert(y as u32 * 4 + row);
                            }
                        }
                        rejected_marker |= cell.fg == RED;
                    }
                }
                assert!(
                    rejected_marker,
                    "rejected marker did not use the shared plot grid"
                );
                assert_eq!(rows.len(),1,"equal values occupied different plot rows at {width}x{height}, health={health}: {rows:?}");
            }
        }
    }

    #[test]
    fn missing_health_measurements_are_not_plotted_as_zero() {
        let mut state = health_session();
        state.apply(&json!({"type":"turn","turn":3,"accepted":false,"metrics":{}}));
        let path = health_trajectory(&state, "copying");
        assert_eq!(path.incumbent.last(), Some(&(3.0, 90.0)));
        assert_eq!(path.reverted.len(), 1);
        assert!(health_trajectory(&state, "unknown").incumbent.is_empty());
        state.health_baseline["invalid"] = json!({"label":"Invalid","score":2.0});
        assert!(!health_ids(&state).contains(&"invalid"));
    }

    #[test]
    fn wide_dashboard_shows_budget_measured_scores_and_history() {
        let output = snapshot(&populated(), 100, 32);
        for expected in [
            "VESPASIAN",
            "Turns 1/8",
            "$0.1234",
            "$5.00",
            "25.0%",
            "75.0%",
            "Turn 1 retained",
            "Hill climb",
            "/tmp/run/checkpoint",
            "q stop",
        ] {
            assert!(output.contains(expected), "missing {expected}:\n{output}");
        }
    }

    #[test]
    fn compact_dashboard_preserves_goal_budget_and_scores() {
        let output = snapshot(&populated(), 60, 18);
        for expected in [
            "VESPASIAN",
            "Answer supplied facts",
            "$0.1234",
            "$5.00",
            "25.0%",
            "75.0%",
            "Hill climb",
            "q stop",
        ] {
            assert!(output.contains(expected), "missing {expected}:\n{output}");
        }
    }

    #[test]
    fn arbitrary_terminal_sizes_do_not_panic() {
        for state in [populated(), health_session()] {
            for (width, height) in [(1, 1), (20, 6), (32, 10), (40, 12), (82, 10), (120, 40)] {
                snapshot(&state, width, height);
            }
        }
    }

    #[test]
    fn rejected_turn_does_not_replace_best_metrics() {
        let mut state = populated();
        state.apply(&json!({"type":"turn","turn":2,"accepted":false,"metrics":{"validation":{"exact_match":0.0}},"rationale":"Preservation score fell"}));
        assert_eq!(score(&state.best, "validation", "exact_match"), Some(0.75));
        assert_eq!(state.turns.last().unwrap().score, Some(0.0));
        assert!(!state.turns.last().unwrap().accepted);
        let Trajectory {
            incumbent,
            retained,
            reverted,
        } = trajectory(&state);
        assert_eq!(incumbent, [(0.0, 25.0), (1.0, 75.0), (2.0, 75.0)]);
        assert_eq!(retained, [(1.0, 75.0)]);
        assert_eq!(reverted, [(2.0, 0.0)]);
    }

    #[test]
    fn external_text_cannot_inject_terminal_controls() {
        assert_eq!(
            clean("\x1b[2Jhello\x1b]52;c;private\x07\nworld\u{202e}"),
            "hello world"
        );
        let mut state = Dashboard::default();
        state.apply(
            &json!({"type":"phase","phase":"editing","message":"clean\x1b[31m text\x1b[0m"}),
        );
        assert_eq!(state.message, "clean text");
        assert_eq!(state.phase, "Editing");
    }

    #[test]
    fn loss_view_uses_raw_loss_and_labels_its_direction() {
        let mut state = populated();
        state.plot_loss = true;
        state.turns[0].loss = Some(2.0);
        let path = trajectory(&state);
        assert_eq!(path.incumbent, [(0.0, 3.2), (1.0, 2.0)]);
        let output = snapshot(&state, 100, 32);
        assert!(output.contains("Reference loss ↓"));
        assert!(!output.contains("320.0%"));
    }

    #[test]
    fn budget_gauge_is_bounded_and_missing_measurements_stay_empty() {
        assert_eq!(ratio(6.0, 5.0), 1.0);
        assert_eq!(ratio(1.0, 0.0), 0.0);
        assert_eq!(ratio(f64::NAN, 1.0), 0.0);
        assert_eq!(percent(None), "—");
        assert_eq!(tail("abcdef", 4), "…def");
        let output = snapshot(&Dashboard::default(), 100, 32);
        assert!(!output.contains("100.0%"));
        assert!(trajectory(&Dashboard::default()).incumbent.is_empty());
    }

    fn press(state: &mut Dashboard, key: KeyCode) {
        state.key(KeyEvent::new(key, KeyModifiers::NONE));
    }

    #[test]
    fn answers_show_only_fields_and_metrics_for_the_dataset_mode() {
        for mode in ["paired", "corpus", "prompts"] {
            let mut state = Dashboard {
                metric: "semantic".into(),
                ..Default::default()
            };
            let mut example = json!({"id":"example", "mode":mode, "prompt":"A prompt", "target":"An example", "output":"Generated text"});
            if mode == "corpus" {
                example["prompt"] = json!("");
                example["loss"] = json!(2.5);
            } else if mode == "prompts" {
                example["target"] = json!("");
            }
            state.apply(&json!({"type":"answer", "label":"baseline", "phase":"evaluating", "example":example}));
            let text = detail_lines(&state, 80)
                .iter()
                .map(Line::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("GENERATED"));
            assert_eq!(text.contains("STYLE EXAMPLE"), mode == "corpus");
            assert_eq!(text.contains("Unconditional text"), mode == "corpus");
            assert_eq!(text.contains("PROMPT"), mode != "corpus");
            assert_eq!(text.contains("REFERENCE"), mode == "paired");
            assert_eq!(text.contains("Loss 2.5000"), mode == "corpus");
            assert!(!text.contains("Exact "));
            assert!(!text.contains("F1 "));
        }
    }

    #[test]
    fn missing_reference_loss_does_not_look_like_a_pending_measurement() {
        let state = Dashboard {
            metric: "semantic".into(),
            plot_loss: true,
            baseline: json!({"validation":{"semantic":0.5,"count":2}}),
            ..Default::default()
        };
        let output = snapshot(&state, 100, 32);
        assert!(output.contains("Reference loss is unavailable"));
        assert!(trajectory(&state).incumbent.is_empty());
    }

    #[test]
    fn answers_keep_raw_and_semantic_scores_separate_by_label() {
        let mut state = Dashboard {
            metric: "semantic".into(),
            ..Default::default()
        };
        for label in ["baseline", "turn-001"] {
            state.apply(&json!({"type":"answer","label":label,"phase":"evaluating","turn":1,"current":1,"total":2,"example":{"id":"validation:abc","split":"validation","prompt":"Why?","target":"Because.","output":"A reason.","exact_match":false,"token_f1":0.2,"loss":2.5}}));
        }
        state.apply(&json!({"type":"answer_score","label":"turn-001","example_id":"validation:abc","semantic":0.8,"judgement":"Satisfies the request."}));
        assert!(state.answers[0].example["semantic"].is_null());
        assert_eq!(state.answers[1].example["semantic"], json!(0.8));
        assert_eq!(state.answers[1].example["exact_match"], json!(false));
        let text = detail_lines(&state, 80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        for expected in [
            "Exact miss",
            "F1 20.0%",
            "GOAL SCORE",
            "80.0%",
            "Satisfies the request.",
            "GENERATED",
            "A reason.",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
        state.apply(&json!({"type":"answer_score","label":"turn-001","example_id":"validation:abc","semantic":2.0}));
        assert_eq!(state.answers[1].example["semantic"], json!(0.8));
    }

    #[test]
    fn rejected_candidates_branch_from_last_retained_checkpoint() {
        let mut state = Dashboard::default();
        for (turn, accepted) in [(1, true), (2, false), (3, true)] {
            state.apply(&json!({"type":"proposal","turn":turn,"proposal":{"operation":"scale","strength":0.001,"targets":[format!("model.layers.{turn}.weight")],"rationale":"Measured perturbation"}}));
            state.apply(&json!({"type":"turn","turn":turn,"accepted":accepted,"metrics":{"validation":{"exact_match":0.5}},"decision_reason":if accepted {"Improved loss"} else {"Objective regressed"}}));
        }
        assert_eq!(
            state
                .changes
                .iter()
                .map(|change| (change.turn, change.parent))
                .collect::<Vec<_>>(),
            vec![(1, 0), (2, 1), (3, 1)]
        );
        assert_eq!(state.retained_turn, 3);
        assert_eq!(state.changes.len(), 3);
        state.detail_pane = DetailPane::Genealogy;
        state.detail_indices[2] = 2;
        let text = detail_lines(&state, 80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("02 ← 01 REVERTED"));
        assert!(text.contains("03 ← 01 RETAINED"));
        assert!(text.contains("Objective regressed"));
    }

    #[test]
    fn provider_activity_is_grouped_without_invented_reasoning() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"editor_activity","call_id":"one","status":"started","provider":"codex","model":"editor-model","purpose":"proposal","turn":2}));
        state.apply(&json!({"type":"editor_activity","call_id":"one","status":"activity","text":"Reading evaluation results\nComparing metrics"}));
        state.apply(&json!({"type":"editor_activity","call_id":"one","status":"complete","summary":"Proposed one tensor scaling","elapsed_ms":2300}));
        assert_eq!(state.editor_calls.len(), 1);
        assert_eq!(state.editor_calls[0].status, "complete");
        assert_eq!(state.editor_calls[0].updates.len(), 3);
        state.detail_pane = DetailPane::Editor;
        let text = detail_lines(&state, 80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Reading evaluation results\nComparing metrics"));
        assert!(text.contains("Proposed one tensor scaling"));
        assert!(text.contains("2.3s"));
    }

    #[test]
    fn local_tool_receipts_show_only_public_activity() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"agent_tool","call_id":"one","turn":2,
            "tool":"read_skill","status":"completed","summary":"Loaded the selected row-edit skill",
            "details":{"request":"PRIVATE_REQUEST","result":"PRIVATE_RESULT"},"reasoning":"PRIVATE_REASONING"}));
        assert_eq!(state.editor_calls.len(), 1);
        assert_eq!(state.editor_calls[0].provider, "local tool");
        assert_eq!(state.editor_calls[0].status, "completed");
        state.detail_pane = DetailPane::Editor;
        let text = detail_lines(&state, 80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Loaded the selected row-edit skill"));
        assert!(text.contains("Read-only strategy discovery"));
        for private in ["PRIVATE_REQUEST", "PRIVATE_RESULT", "PRIVATE_REASONING"] {
            assert!(!text.contains(private));
        }
        assert!(state.loop_stages.is_empty());
        state.apply(&json!({"type":"agent_tool","call_id":"two","turn":2,
            "tool":"read_skill","status":"failed","summary":"Skill unavailable"}));
        assert_eq!(state.editor_calls[1].status, "failed");
        state.apply(&json!({"type":"agent_tool","call_id":"three","turn":2,
            "tool":"read_skill","status":"imagined","summary":"Not a receipt"}));
        assert_eq!(state.editor_calls.len(), 2);
    }

    #[test]
    fn long_details_scroll_to_the_end_and_browsing_pauses_follow() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"answer","label":"baseline","example":{"id":"a","prompt":"A prompt","target":"Expected","output":format!("{}\nFINAL SENTENCE","An answer with many words. ".repeat(100))}}));
        state.expanded = true;
        snapshot(&state, 60, 18);
        for _ in 0..100 {
            press(&mut state, KeyCode::PageDown);
        }
        let end = snapshot(&state, 60, 18);
        assert!(end.contains("FINAL SENTENCE"), "{end}");
        assert!(!state.follow_latest[0]);
        state.apply(&json!({"type":"answer","label":"baseline","example":{"id":"b","output":"Another answer"}}));
        assert_eq!(state.detail_indices[0], 0);
        press(&mut state, KeyCode::End);
        assert_eq!(state.detail_indices[0], 1);
        assert!(state.follow_latest[0]);
        press(&mut state, KeyCode::Home);
        assert_eq!(state.detail_indices[0], 0);
        assert_eq!(state.detail_scroll[0], 0);
        press(&mut state, KeyCode::Esc);
        assert!(!state.expanded);
    }

    #[test]
    fn shortened_answers_point_to_the_complete_receipt() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"answer","label":"baseline","example":{"id":"a","output":"Preview"},"truncated_fields":["output"],"receipt":{"file":"baseline.json","pointer":"/evaluation/samples","index":0}}));
        let text = detail_lines(&state, 100)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Display shortened"));
        assert!(text.contains("Full output after evaluation: baseline.json/evaluation/samples [0]"));
    }

    #[test]
    fn missing_progress_remains_unknown_and_real_counts_are_used() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"progress","phase":"loading","message":"Opening checkpoint"}));
        assert!(!state.progress_known);
        let text = snapshot(&state, 100, 32);
        assert!(text.contains("Loading · —"));
        assert!(!text.contains("0/0"));
        state.apply(&json!({"type":"progress","phase":"evaluating","current":3,"total":12}));
        assert!(state.progress_known);
        assert!(snapshot(&state, 100, 32).contains("Examples 3/12"));
        state.apply(&json!({"type":"phase","phase":"saving"}));
        assert!(!state.progress_known);
    }

    #[test]
    fn layer_map_tracks_actual_active_tensor_and_rejected_branches() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"model_structure","num_layers":28,"architecture":"Qwen","editable":[{"name":"model.layers.4.weight","shape":[4,4]}]}));
        state.apply(
            &json!({"type":"proposal","turn":1,"proposal":{"targets":["model.layers.4.weight"]}}),
        );
        let (retained, queued, active, _) = layer_states(&state);
        assert!(retained.is_empty());
        assert!(queued.contains(&4));
        assert_eq!(active, None);
        state.apply(
            &json!({"type":"tensor_edit","status":"started","tensor":"model.layers.4.weight"}),
        );
        assert_eq!(layer_states(&state).2, Some(4));
        state.apply(
            &json!({"type":"tensor_edit","status":"applied","tensor":"model.layers.4.weight"}),
        );
        assert_eq!(layer_states(&state).2, None);
        state.apply(&json!({"type":"turn","turn":1,"accepted":false}));
        assert!(layer_states(&state).0.is_empty());
        assert!(layer_states(&state).1.is_empty());
        state.detail_pane = DetailPane::Genealogy;
        assert!(layer_states(&state).3.contains(&4));
        let text = snapshot(&state, 100, 32);
        for expected in ["LAYERS · 28", "proposed", "kept", "selected", "27"] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
    }

    #[test]
    fn compact_layer_map_focuses_retained_layers_and_pages_through_all_indices() {
        let mut state = Dashboard::default();
        state.apply(&json!({"type":"model_structure","num_layers":24}));
        state.apply(&json!({"type":"turn","turn":1,"accepted":true,"proposal":{"targets":["model.layers.20.weight","model.layers.23.weight"]}}));
        let output = snapshot(&state, 60, 18);
        assert!(output.contains("LAYERS 05–23/24"), "{output}");
        assert!(output.contains("20 21 22 23"), "{output}");
        press(&mut state, KeyCode::Char('l'));
        let output = snapshot(&state, 60, 18);
        assert!(output.contains("LAYERS 00–18/24"), "{output}");
        assert!(output.contains("00 01 02"), "{output}");
    }

    #[test]
    fn compact_expansion_keeps_all_detail_panes_accessible() {
        let mut state = health_session();
        for pane in [
            DetailPane::Answers,
            DetailPane::Editor,
            DetailPane::Genealogy,
        ] {
            state.detail_pane = pane;
            state.expanded = false;
            let graph = snapshot(&state, 60, 18);
            assert!(graph.contains("Hill climb"));
            assert!(graph.contains(pane.label()));
            press(&mut state, KeyCode::Enter);
            let detail = snapshot(&state, 60, 18);
            assert!(detail.contains("Esc return"));
            assert!(detail.contains(pane.label()));
        }
    }

    /// Renders the actual recorded session without adding sample measurements.
    /// VESPASIAN_EVENTS=/path/events.jsonl cargo test ui::tests::render_recorded_session -- --ignored
    #[test]
    #[ignore = "requires VESPASIAN_EVENTS pointing to a recorded session"]
    fn render_recorded_session() {
        let source = std::path::PathBuf::from(
            std::env::var_os("VESPASIAN_EVENTS").expect("set VESPASIAN_EVENTS"),
        );
        let content = std::fs::read_to_string(&source).unwrap();
        let mut state = Dashboard::default();
        let event_limit = std::env::var("VESPASIAN_EVENT_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        for line in content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .take(event_limit)
        {
            state.apply(&serde_json::from_str::<Value>(line).unwrap());
        }
        state.plot_loss = std::env::var("VESPASIAN_VIEW").as_deref() == Ok("loss");
        state.detail_pane = match std::env::var("VESPASIAN_PANE").as_deref() {
            Ok("editor") => DetailPane::Editor,
            Ok("genealogy") => DetailPane::Genealogy,
            _ => DetailPane::Answers,
        };
        state.expanded = std::env::var("VESPASIAN_EXPANDED").as_deref() == Ok("1");
        if let Ok(index) = std::env::var("VESPASIAN_ITEM") {
            state.detail_indices[state.detail_pane.index()] = index.parse().unwrap();
            state.follow_latest[state.detail_pane.index()] = false;
        }
        if let Ok(scroll) = std::env::var("VESPASIAN_SCROLL") {
            state.detail_scroll[state.detail_pane.index()] = scroll.parse().unwrap();
        }
        let output = std::env::var_os("VESPASIAN_SNAPSHOT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| source.parent().unwrap().to_path_buf());
        std::fs::create_dir_all(&output).unwrap();
        let width = std::env::var("VESPASIAN_SNAPSHOT_WIDTH")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(100);
        let height = std::env::var("VESPASIAN_SNAPSHOT_HEIGHT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(32);
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| render(frame, &state)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut plain = String::new();
        let mut ansi = String::from("\x1b[2J\x1b[H");
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><rect width=\"100%\" height=\"100%\" fill=\"#14171b\"/><g font-family=\"DejaVu Sans Mono,monospace\" font-size=\"14\">",
            width as f64 * 8.5, height as u32 * 20, width as f64 * 8.5, height as u32 * 20,
        );
        for y in 0..height {
            for x in 0..width {
                let cell = &buffer[(x, y)];
                plain.push_str(cell.symbol());
                let foreground = if cell.modifier.contains(Modifier::REVERSED) {
                    cell.bg
                } else {
                    cell.fg
                };
                let background = if cell.modifier.contains(Modifier::REVERSED) {
                    cell.fg
                } else {
                    cell.bg
                };
                if let Color::Rgb(br, bg, bb) = background {
                    if background != INK {
                        svg.push_str(&format!("<rect x=\"{}\" y=\"{}\" width=\"8.5\" height=\"20\" fill=\"#{br:02x}{bg:02x}{bb:02x}\"/>",x as f64*8.5,y as u32*20));
                    }
                }
                let (r, g, b) = match foreground {
                    Color::Rgb(r, g, b) => (r, g, b),
                    _ => (236, 226, 210),
                };
                ansi.push_str(&format!("\x1b[38;2;{r};{g};{b}m{}", cell.symbol()));
                if let Some(ch) = cell
                    .symbol()
                    .chars()
                    .next()
                    .filter(|ch| ('\u{2800}'..='\u{28ff}').contains(ch))
                {
                    let dots = ch as u32 - 0x2800;
                    for (bit, column, row) in [
                        (0, 0, 0),
                        (1, 0, 1),
                        (2, 0, 2),
                        (3, 1, 0),
                        (4, 1, 1),
                        (5, 1, 2),
                        (6, 0, 3),
                        (7, 1, 3),
                    ] {
                        if dots & (1 << bit) != 0 {
                            svg.push_str(&format!("<circle cx=\"{}\" cy=\"{}\" r=\"0.8\" fill=\"#{r:02x}{g:02x}{b:02x}\"/>", x as f64 * 8.5 + 2.75 + column as f64 * 4.0, y as f64 * 20.0 + 2.75 + row as f64 * 4.0));
                        }
                    }
                    continue;
                }
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
                svg.push_str(&format!(
                    "<text x=\"{}\" y=\"{}\" fill=\"#{r:02x}{g:02x}{b:02x}\" font-weight=\"{weight}\">{symbol}</text>",
                    x as f64 * 8.5, y as u32 * 20 + 16,
                ));
            }
            plain.push('\n');
            ansi.push_str("\x1b[0m\r\n");
        }
        svg.push_str("</g></svg>");
        std::fs::write(output.join("terminal.txt"), plain).unwrap();
        std::fs::write(output.join("terminal.ansi"), ansi).unwrap();
        std::fs::write(output.join("terminal.svg"), svg).unwrap();
    }
}
