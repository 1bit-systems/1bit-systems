//! eframe/egui `App` impl for 1bit-helm.
//!
//! Four panes, left-nav + top-bar switcher:
//!
//! * Status   — subscribes to `http://127.0.0.1:8190/_live/stats` (1bit-landing
//!   SSE). Renders loaded model + tok/s gauge + GPU temp/util +
//!   NPU up + shadow-burn % + per-service dots.
//! * Chat     — `POST /v1/chat/completions` against the lemonade gateway
//!   (default `http://127.0.0.1:8200`), SSE `stream: true`,
//!   token-by-token append.
//! * Models   — `GET /v1/models`. Card grid with a disabled "Load" button
//!   (server loads at startup today — future hook).
//! * Settings — bearer-token mgmt (+ about dialog).
//!
//! The Skills + Memory panes were retired in the 2026-04-25 cull when
//! `onebit-agents` was deleted (superseded by GAIA agent-core in
//! `1bit-services/agent-core/`). Re-add via the new agent-core surface
//! when the canonical port lands.
//!
//! Persistence: last-open pane + window size via eframe's `persistence`
//! feature (ron-backed).
//!
//! Runtime: `HelmApp::new` stays runtime-free (tests construct it without
//! pulling tokio up). The network wiring lives in `HelmApp::attach_runtime`,
//! called from `main` after eframe hands us a `tokio::runtime::Handle`.
//! Worker messages come back via `mpsc::UnboundedReceiver` drained each
//! frame; `ctx.request_repaint()` is issued when a new message lands.

use crate::bearer::Bearer;
use crate::conv_log::{default_root as default_log_root, write_session};
use crate::conversation::{Conversation, Role};
use crate::models::{ModelCard, fetch_models};
use crate::session::SessionConfig;
use crate::stream::{SseEvent, parse_sse_line};
use crate::telemetry::{self, LiveStats, ServiceDot, TelemetryMsg};
use crate::{BRAND, BRAND_DOMAIN};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::runtime::Handle;
use tokio::sync::mpsc;

/// Which pane is currently focused.
///
/// Persisted across restarts via eframe's storage (see `save`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pane {
    #[default]
    Status,
    Chat,
    Models,
    Settings,
}

impl Pane {
    pub const ALL: &'static [Pane] = &[
        Pane::Status,
        Pane::Chat,
        Pane::Models,
        Pane::Settings,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Pane::Status => "Status",
            Pane::Chat => "Chat",
            Pane::Models => "Models",
            Pane::Settings => "Settings",
        }
    }
}


/// Messages from background workers back to the UI.
#[derive(Debug)]
pub enum UiMsg {
    Telemetry(TelemetryMsg),
    ChatDelta(String),
    ChatDone,
    ChatError(String),
    Models(Result<Vec<ModelCard>, String>),
    Toast(String),
}

/// Top-level eframe app. The eframe-facing `App` impl lives below; this
/// struct itself is plain data so construction + default-state tests don't
/// require an open window.
pub struct HelmApp {
    /// Server + model + bearer config. Loaded from env via `main`.
    pub cfg: SessionConfig,
    /// Gateway base for `/v1/*` (default http://127.0.0.1:8200 — lemonade).
    pub gateway_url: String,
    /// Landing base for `/_live/*` (default http://127.0.0.1:8190).
    pub landing_url: String,

    /// Pane currently visible in the right-hand content area.
    pub current_pane: Pane,

    /// Conversation buffer for the Chat pane.
    pub chat_conv: Conversation,
    /// User's current half-typed input.
    pub chat_input: String,
    /// Streaming assistant reply, populated by the network worker. None
    /// when idle.
    pub chat_streaming: Option<String>,

    /// Latest /v1/models snapshot for the Models pane.
    pub models: Vec<ModelCard>,
    pub models_error: Option<String>,

    /// Status-pane live readout (mirrored from `/_live/stats`).
    pub live: LiveStats,
    pub live_connected: bool,
    pub live_last_error: Option<String>,

    /// Bearer-token storage.
    pub bearer: Bearer,
    /// True on first launch — opens the modal that prompts for the token.
    pub show_bearer_modal: bool,
    /// Textarea backing for the modal + Settings pane.
    pub bearer_input: String,

    /// Toast strip message (disappears on the next meaningful action).
    pub toast: Option<String>,

    /// Last error string shown in the status bar. Rendered in red on the
    /// bottom strip when set; clears on next successful action.
    pub last_error: Option<String>,

    /// Shared runtime handle, set by [`HelmApp::attach_runtime`]. None in
    /// tests so construction stays cheap.
    runtime: Option<Handle>,
    /// Drain this each frame for [`UiMsg`].
    ui_rx: Option<mpsc::UnboundedReceiver<UiMsg>>,
    /// Cloneable sender so background workers can post back.
    ui_tx: Option<mpsc::UnboundedSender<UiMsg>>,
    /// Shared HTTP client.
    http: Option<reqwest::Client>,
    /// Where `write_session` lands — overridable for tests.
    pub log_root: PathBuf,
}

impl HelmApp {
    /// Build a fresh app from env-derived config. Safe to call outside
    /// eframe — no network, no window, no filesystem I/O.
    pub fn new(cfg: SessionConfig) -> Self {
        let gateway_url = cfg.server_url.clone();
        Self {
            cfg,
            gateway_url,
            landing_url: "http://127.0.0.1:8190".to_string(),
            current_pane: Pane::default(),
            chat_conv: Conversation::new(),
            chat_input: String::new(),
            chat_streaming: None,
            models: Vec::new(),
            models_error: None,
            live: LiveStats::default(),
            live_connected: false,
            live_last_error: None,
            bearer: Bearer::new(),
            show_bearer_modal: false,
            bearer_input: String::new(),
            toast: None,
            last_error: None,
            runtime: None,
            ui_rx: None,
            ui_tx: None,
            http: None,
            log_root: default_log_root(),
        }
    }

    /// eframe entry point. Pulls persisted pane + window size out of
    /// `cc.storage` if available.
    pub fn from_cc(cc: &eframe::CreationContext<'_>, cfg: SessionConfig) -> Self {
        let mut app = Self::new(cfg);
        if let Some(storage) = cc.storage
            && let Some(pane) = eframe::get_value::<Pane>(storage, PERSIST_PANE_KEY)
        {
            app.current_pane = pane;
        }
        app
    }

    /// Wire the app to a live tokio runtime + HTTP client, then kick off
    /// the telemetry stream + initial refreshes. Call once from `main`.
    pub fn attach_runtime(&mut self, handle: Handle, http: reqwest::Client) {
        let (ui_tx, ui_rx) = mpsc::unbounded_channel();
        self.ui_tx = Some(ui_tx.clone());
        self.ui_rx = Some(ui_rx);
        self.http = Some(http.clone());
        self.runtime = Some(handle.clone());

        // Bearer load is fast enough to do synchronously.
        self.bearer.load();
        if self.bearer.get().is_none() && self.cfg.bearer.is_none() {
            self.show_bearer_modal = true;
        } else if let Some(tok) = self.bearer.get() {
            // Keyring/file copy beats the env-derived one — it's the
            // persistent truth; env vars are a one-session override.
            self.cfg.bearer = Some(tok.to_string());
        }

        // Kick off the SSE subscription. The worker posts back on ui_tx.
        let telemetry_rx = telemetry::spawn(&handle, http, self.landing_url.clone());
        let forwarder_tx = ui_tx;
        handle.spawn(async move {
            let mut rx = telemetry_rx;
            while let Some(msg) = rx.recv().await {
                if forwarder_tx.send(UiMsg::Telemetry(msg)).is_err() {
                    break;
                }
            }
        });

        // Seed Models now so first-paint isn't blank.
        self.refresh_models();
    }

    /// Fire a `GET /v1/models` on the ambient runtime. Result comes back
    /// via `UiMsg::Models`.
    pub fn refresh_models(&mut self) {
        let (Some(rt), Some(http), Some(tx)) = (&self.runtime, &self.http, &self.ui_tx) else {
            return;
        };
        let base = self.gateway_url.clone();
        let bearer = self.cfg.bearer.clone();
        let http = http.clone();
        let tx = tx.clone();
        rt.spawn(async move {
            let res = fetch_models(&http, &base, bearer.as_deref())
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(UiMsg::Models(res));
        });
    }

    /// Spawn the chat SSE streaming request. Tokens arrive as
    /// `UiMsg::ChatDelta(...)`, finishing with `UiMsg::ChatDone`.
    pub fn send_chat(&mut self) {
        if self.chat_input.trim().is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.chat_input);
        self.chat_conv.push_user(text);

        let (Some(rt), Some(http), Some(tx)) = (&self.runtime, &self.http, &self.ui_tx) else {
            self.last_error =
                Some("runtime not attached — chat requires HelmApp::attach_runtime".into());
            return;
        };
        self.chat_streaming = Some(String::new());

        // Rebuild a fresh body per request; we include the conversation
        // so the server sees the full context window.
        let body = build_chat_body(&self.cfg, &self.chat_conv);
        let url = format!(
            "{}/v1/chat/completions",
            self.gateway_url.trim_end_matches('/')
        );
        let bearer = self.cfg.bearer.clone();
        let http = http.clone();
        let tx = tx.clone();

        rt.spawn(async move {
            if let Err(e) = stream_chat(http, url, body, bearer, tx.clone()).await {
                let _ = tx.send(UiMsg::ChatError(e.to_string()));
            }
        });
    }

    /// Flush the conversation to `~/.halo/helm/conversations/<ts>.jsonl`
    /// (creating the dir if missing). Called from `save()` — i.e. when
    /// eframe persists state / closes.
    fn flush_conversation(&self) {
        if self.chat_conv.turns.is_empty() {
            return;
        }
        if let Err(e) = write_session(&self.log_root, &self.chat_conv) {
            tracing::warn!(err = %e, "conversation log write failed");
        }
    }
}

/// Storage key for the currently-selected pane.
const PERSIST_PANE_KEY: &str = "helm.current_pane";

/// Build the `/v1/chat/completions` JSON body. Pure — exposed for testing.
pub fn build_chat_body(cfg: &SessionConfig, conv: &Conversation) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::with_capacity(conv.turns.len() + 1);
    if let Some(sys) = &cfg.system_prompt {
        messages.push(serde_json::json!({ "role": "system", "content": sys }));
    }
    messages.extend(conv.to_openai_messages());
    serde_json::json!({
        "model": cfg.default_model,
        "messages": messages,
        "stream": true
    })
}

async fn stream_chat(
    http: reqwest::Client,
    url: String,
    body: serde_json::Value,
    bearer: Option<String>,
    tx: mpsc::UnboundedSender<UiMsg>,
) -> anyhow::Result<()> {
    let mut req = http.post(&url).json(&body);
    if let Some(tok) = bearer {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("chat: {status}: {text}");
    }
    let mut bytes = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk?;
        buf.extend_from_slice(&chunk);
        while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
            let line = buf.drain(..=nl).collect::<Vec<u8>>();
            let line = &line[..line.len() - 1];
            let line = std::str::from_utf8(line).unwrap_or("");
            match parse_sse_line(line) {
                SseEvent::Delta(s) => {
                    if tx.send(UiMsg::ChatDelta(s)).is_err() {
                        return Ok(());
                    }
                }
                SseEvent::Done => {
                    let _ = tx.send(UiMsg::ChatDone);
                    return Ok(());
                }
                SseEvent::Ignore => {}
            }
        }
    }
    let _ = tx.send(UiMsg::ChatDone);
    Ok(())
}

// --- eframe wiring ---------------------------------------------------------

impl eframe::App for HelmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain the UI channel. Collect then process so we don't hold the
        // borrow on `self.ui_rx` while mutating other fields.
        let mut drained: Vec<UiMsg> = Vec::new();
        if let Some(rx) = self.ui_rx.as_mut() {
            while let Ok(msg) = rx.try_recv() {
                drained.push(msg);
            }
        }
        let had_messages = !drained.is_empty();
        for msg in drained {
            self.apply_ui_msg(msg);
        }
        // Telemetry arrives on a 1.5 s cadence; repaint when something
        // actually lands so we don't burn CPU on idle frames.
        if had_messages {
            ctx.request_repaint();
        } else {
            // Idle repaint every ~1s so the bottom strip's "tok/s" readout
            // doesn't look frozen even between SSE events.
            ctx.request_repaint_after(std::time::Duration::from_millis(1000));
        }

        // First-launch bearer modal: blocks everything else.
        if self.show_bearer_modal {
            draw_bearer_modal(ctx, self);
        }

        // Top bar: brand + pane switcher buttons.
        egui::TopBottomPanel::top("helm-top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("1bit-helm");
                ui.label(egui::RichText::new(format!("  —  {BRAND}")).weak());
                ui.separator();
                for p in Pane::ALL {
                    let selected = self.current_pane == *p;
                    if ui.selectable_label(selected, p.label()).clicked() {
                        self.current_pane = *p;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.monospace(&self.gateway_url);
                });
            });
        });

        // Left nav: same panes (radio-style) + quick config readout.
        egui::SidePanel::left("helm-nav")
            .default_width(160.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Panes").strong());
                for p in Pane::ALL {
                    ui.radio_value(&mut self.current_pane, *p, p.label());
                }
                ui.separator();
                ui.label(egui::RichText::new("Model").strong());
                ui.monospace(&self.cfg.default_model);
                ui.separator();
                ui.label(egui::RichText::new("Landing").strong());
                ui.monospace(&self.landing_url);
            });

        // Bottom status strip: brand + live dot.
        egui::TopBottomPanel::bottom("helm-status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let dot_color = if self.live_connected {
                    egui::Color32::from_rgb(0, 170, 0)
                } else {
                    egui::Color32::from_rgb(160, 0, 0)
                };
                ui.colored_label(dot_color, "●");
                if let Some(toast) = &self.toast {
                    ui.label(toast);
                } else if let Some(err) = &self.last_error {
                    ui.colored_label(egui::Color32::RED, err);
                } else {
                    ui.label(format!(
                        "{BRAND}  |  {}  |  {:.1} tok/s  |  GPU {}°C {}%",
                        if self.live.loaded_model.is_empty() {
                            "(no model)"
                        } else {
                            &self.live.loaded_model
                        },
                        self.live.tok_s_decode,
                        self.live.gpu_temp_c as i32,
                        self.live.gpu_util_pct,
                    ));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(BRAND_DOMAIN);
                });
            });
        });

        // Main content.
        egui::CentralPanel::default().show(ctx, |ui| match self.current_pane {
            Pane::Status => draw_status(ui, self),
            Pane::Chat => draw_chat(ui, self),
            Pane::Models => draw_models(ui, self),
            Pane::Settings => draw_settings(ui, self),
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PERSIST_PANE_KEY, &self.current_pane);
        // Flush current conversation on shutdown / periodic save.
        self.flush_conversation();
    }
}

impl HelmApp {
    fn apply_ui_msg(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Telemetry(TelemetryMsg::Snapshot(s)) => {
                self.live = s;
                self.live_connected = true;
                self.live_last_error = None;
            }
            UiMsg::Telemetry(TelemetryMsg::Disconnected(e)) => {
                self.live_connected = false;
                self.live_last_error = Some(e);
            }
            UiMsg::ChatDelta(s) => {
                if let Some(buf) = self.chat_streaming.as_mut() {
                    buf.push_str(&s);
                }
            }
            UiMsg::ChatDone => {
                if let Some(text) = self.chat_streaming.take()
                    && !text.is_empty()
                {
                    self.chat_conv.push_assistant(text);
                }
            }
            UiMsg::ChatError(e) => {
                self.chat_streaming = None;
                self.last_error = Some(format!("chat: {e}"));
            }
            UiMsg::Models(Ok(cards)) => {
                self.models = cards;
                self.models_error = None;
            }
            UiMsg::Models(Err(e)) => {
                self.models_error = Some(e);
            }
            UiMsg::Toast(s) => {
                self.toast = Some(s);
            }
        }
    }
}

// --- pane renderers --------------------------------------------------------

fn draw_status(ui: &mut egui::Ui, app: &HelmApp) {
    ui.heading("Status");
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let c = if app.live_connected {
            egui::Color32::from_rgb(0, 170, 0)
        } else {
            egui::Color32::from_rgb(160, 0, 0)
        };
        ui.colored_label(c, "●");
        ui.label("telemetry:");
        ui.monospace(format!("{}/_live/stats", app.landing_url));
        if let Some(err) = &app.live_last_error {
            ui.colored_label(egui::Color32::from_rgb(180, 120, 0), err);
        }
    });
    ui.add_space(6.0);

    egui::Grid::new("status-grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            ui.label("loaded model");
            ui.monospace(if app.live.loaded_model.is_empty() {
                "—".to_string()
            } else {
                app.live.loaded_model.clone()
            });
            ui.end_row();

            ui.label("tok/s (decode)");
            let tok_s = app.live.tok_s_decode;
            ui.add(
                egui::ProgressBar::new((tok_s / 120.0).clamp(0.0, 1.0))
                    .text(format!("{tok_s:.1}"))
                    .desired_width(260.0),
            );
            ui.end_row();

            ui.label("GPU temp");
            ui.monospace(format!("{:.0} °C", app.live.gpu_temp_c));
            ui.end_row();

            ui.label("GPU util");
            ui.add(
                egui::ProgressBar::new((app.live.gpu_util_pct as f32 / 100.0).clamp(0.0, 1.0))
                    .text(format!("{} %", app.live.gpu_util_pct))
                    .desired_width(260.0),
            );
            ui.end_row();

            ui.label("NPU");
            if app.live.npu_up {
                ui.colored_label(egui::Color32::from_rgb(0, 170, 0), "up");
            } else {
                ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "down");
            }
            ui.end_row();

            ui.label("shadow-burn %");
            ui.monospace(format!("{:.1}", app.live.shadow_burn_exact_pct));
            ui.end_row();

            ui.label("stale");
            ui.monospace(if app.live.stale { "yes" } else { "no" });
            ui.end_row();
        });

    ui.separator();
    ui.label(egui::RichText::new("Services").strong());
    ui.add_space(2.0);
    if app.live.services.is_empty() {
        ui.label("(no service probes yet — waiting for first telemetry tick)");
    } else {
        ui.horizontal_wrapped(|ui| {
            for svc in &app.live.services {
                draw_service_dot(ui, svc);
            }
        });
    }
}

fn draw_service_dot(ui: &mut egui::Ui, svc: &ServiceDot) {
    let c = if svc.active {
        egui::Color32::from_rgb(0, 170, 0)
    } else {
        egui::Color32::from_rgb(160, 0, 0)
    };
    ui.colored_label(c, "●");
    ui.monospace(&svc.name);
    ui.add_space(8.0);
}

fn draw_chat(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Chat");
    ui.add_space(4.0);

    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .max_height(ui.available_height() - 60.0)
        .show(ui, |ui| {
            for turn in &app.chat_conv.turns {
                if matches!(turn.role, Role::System) {
                    continue;
                }
                let (label, color) = match turn.role {
                    Role::User => ("you", egui::Color32::from_rgb(0, 140, 200)),
                    Role::Assistant => ("halo", egui::Color32::from_rgb(0, 170, 0)),
                    Role::System => ("sys", egui::Color32::GRAY),
                };
                ui.colored_label(color, format!("{label}:"));
                ui.label(&turn.content);
                ui.add_space(6.0);
            }
            if let Some(partial) = &app.chat_streaming {
                ui.colored_label(egui::Color32::from_rgb(0, 170, 0), "halo:");
                ui.label(partial);
            }
        });

    ui.separator();
    ui.horizontal(|ui| {
        let text_edit = egui::TextEdit::singleline(&mut app.chat_input)
            .desired_width(ui.available_width() - 80.0)
            .hint_text("message…");
        let resp = ui.add(text_edit);
        let send_key = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let click = ui.button("Send").clicked();
        if send_key || click {
            app.send_chat();
        }
    });
}

fn draw_models(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Models");
    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            app.refresh_models();
        }
        ui.label(format!(
            "{} model(s) @ {}",
            app.models.len(),
            app.gateway_url
        ));
    });
    ui.separator();
    if let Some(err) = &app.models_error {
        ui.colored_label(egui::Color32::RED, err);
    }
    if app.models.is_empty() && app.models_error.is_none() {
        ui.label("(no models loaded — click Refresh, or the gateway may be down)");
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        // Render as a card grid; one card per row for now — a grid-wrap
        // layout in egui wants a fixed item width, which our model ids
        // don't uniformly fit into.
        let mut load_click_id: Option<String> = None;
        for card in &app.models {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(&card.id).strong().monospace());
                        if !card.owned_by.is_empty() {
                            ui.label(
                                egui::RichText::new(format!("owned_by: {}", card.owned_by)).weak(),
                            );
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Load").clicked() {
                            load_click_id = Some(card.id.clone());
                        }
                    });
                });
            });
            ui.add_space(4.0);
        }
        if let Some(id) = load_click_id {
            app.toast = Some(format!(
                "'{id}': halo-server loads models at startup today; model-swap API TBD"
            ));
        }
    });
}

fn draw_settings(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Settings");
    ui.add_space(6.0);

    ui.label(egui::RichText::new("Bearer token").strong());
    ui.horizontal(|ui| {
        ui.label("backend:");
        ui.monospace(app.bearer.backend().label());
    });
    ui.horizontal(|ui| {
        ui.label("status:");
        if app.bearer.get().is_some() {
            ui.colored_label(egui::Color32::from_rgb(0, 170, 0), "set");
        } else {
            ui.colored_label(egui::Color32::from_rgb(160, 0, 0), "unset");
        }
    });
    ui.horizontal(|ui| {
        let edit = egui::TextEdit::singleline(&mut app.bearer_input)
            .password(true)
            .hint_text("paste bearer …")
            .desired_width(320.0);
        ui.add(edit);
        if ui.button("Save").clicked() && !app.bearer_input.trim().is_empty() {
            match app.bearer.store(&app.bearer_input) {
                Ok(b) => {
                    app.cfg.bearer = app.bearer.get().map(str::to_owned);
                    app.toast = Some(format!("bearer stored in {}", b.label()));
                    app.bearer_input.clear();
                }
                Err(e) => app.last_error = Some(format!("bearer: {e}")),
            }
        }
        if ui.button("Reset").clicked() {
            if let Err(e) = app.bearer.clear() {
                app.last_error = Some(format!("bearer clear: {e}"));
            } else {
                app.cfg.bearer = None;
                app.toast = Some("bearer cleared".into());
            }
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Endpoints").strong());
    ui.horizontal(|ui| {
        ui.label("gateway:");
        ui.monospace(&app.gateway_url);
    });
    ui.horizontal(|ui| {
        ui.label("landing:");
        ui.monospace(&app.landing_url);
    });

    ui.separator();
    ui.label(egui::RichText::new("About").strong());
    ui.label(format!("{BRAND}  —  {BRAND_DOMAIN}"));
    ui.label(format!(
        "1bit-helm  v{} — native Rust desktop client",
        env!("CARGO_PKG_VERSION")
    ));
}

fn draw_bearer_modal(ctx: &egui::Context, app: &mut HelmApp) {
    egui::Window::new("Set bearer token")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.label(format!("{BRAND} — first-run setup"));
            ui.label(
                "Paste the bearer for lemond's `/v1/*` endpoints.\n\
                 Stored in your system keyring (or chmod-0600 XDG file).",
            );
            ui.add_space(4.0);
            let edit = egui::TextEdit::singleline(&mut app.bearer_input)
                .password(true)
                .hint_text("sk-…")
                .desired_width(320.0);
            ui.add(edit);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() && !app.bearer_input.trim().is_empty() {
                    match app.bearer.store(&app.bearer_input) {
                        Ok(b) => {
                            app.cfg.bearer = app.bearer.get().map(str::to_owned);
                            app.toast = Some(format!("bearer stored in {}", b.label()));
                            app.bearer_input.clear();
                            app.show_bearer_modal = false;
                            // Re-fire dependent fetches now that we have auth.
                            app.refresh_models();
                        }
                        Err(e) => app.last_error = Some(format!("bearer: {e}")),
                    }
                }
                if ui.button("Skip (no auth)").clicked() {
                    app.show_bearer_modal = false;
                }
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv_log::read_session;
    use tempfile::TempDir;

    fn cfg() -> SessionConfig {
        SessionConfig::new("http://127.0.0.1:8200", "1bit-monster-2b")
    }

    #[test]
    fn new_constructs_without_panic_and_defaults_are_sane() {
        let app = HelmApp::new(cfg());
        assert_eq!(app.cfg.server_url, "http://127.0.0.1:8200");
        assert_eq!(app.gateway_url, "http://127.0.0.1:8200");
        assert_eq!(app.landing_url, "http://127.0.0.1:8190");
        assert_eq!(app.cfg.default_model, "1bit-monster-2b");
        assert!(app.chat_conv.turns.is_empty());
        assert!(app.chat_input.is_empty());
        assert!(app.chat_streaming.is_none());
        assert!(app.models.is_empty());
        assert!(app.last_error.is_none());
    }

    #[test]
    fn default_pane_is_status() {
        assert_eq!(Pane::default(), Pane::Status);
        let app = HelmApp::new(cfg());
        assert_eq!(app.current_pane, Pane::Status);
    }

    #[test]
    fn pane_all_covers_four_panes_in_declared_order() {
        // Tripwire: if we add a pane to the enum but miss `Pane::ALL`, the
        // UI silently drops it.
        assert_eq!(Pane::ALL.len(), 4);
        assert_eq!(Pane::ALL[0], Pane::Status);
        assert_eq!(Pane::ALL[1], Pane::Chat);
        assert_eq!(Pane::ALL[2], Pane::Models);
        assert_eq!(Pane::ALL[3], Pane::Settings);
    }

    #[test]
    fn pane_labels_are_stable_strings() {
        assert_eq!(Pane::Status.label(), "Status");
        assert_eq!(Pane::Chat.label(), "Chat");
        assert_eq!(Pane::Models.label(), "Models");
        assert_eq!(Pane::Settings.label(), "Settings");
    }

    #[test]
    fn pane_round_trips_through_serde() {
        for p in Pane::ALL {
            let s = serde_json::to_string(p).unwrap();
            let back: Pane = serde_json::from_str(&s).unwrap();
            assert_eq!(back, *p);
        }
    }

    #[test]
    fn conversation_log_flushes_on_save_path() {
        // Simulate: app gets two turns, flush_conversation writes a JSONL,
        // and we can re-read the entries. Mirrors the path eframe's
        // `save()` runs.
        let td = TempDir::new().unwrap();
        let mut app = HelmApp::new(cfg());
        app.log_root = td.path().to_path_buf();
        app.chat_conv.push_user("hi".into());
        app.chat_conv.push_assistant("hello".into());
        app.flush_conversation();

        let files: Vec<_> = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(files.len(), 1);
        let entries = read_session(&files[0].path()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[0].content, "hi");
        assert_eq!(entries[1].role, "assistant");
    }

    #[test]
    fn build_chat_body_sets_stream_true_and_includes_turns() {
        let cfg = SessionConfig::new("http://x", "m");
        let mut conv = Conversation::new();
        conv.push_user("q".into());
        let body = build_chat_body(&cfg, &conv);
        assert_eq!(body["stream"], serde_json::Value::Bool(true));
        assert_eq!(body["model"], "m");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn apply_ui_msg_chat_delta_then_done_lands_assistant_turn() {
        let mut app = HelmApp::new(cfg());
        app.chat_streaming = Some(String::new());
        app.apply_ui_msg(UiMsg::ChatDelta("Hel".into()));
        app.apply_ui_msg(UiMsg::ChatDelta("lo".into()));
        assert_eq!(app.chat_streaming.as_deref(), Some("Hello"));
        app.apply_ui_msg(UiMsg::ChatDone);
        assert!(app.chat_streaming.is_none());
        assert_eq!(app.chat_conv.turns.len(), 1);
        assert_eq!(app.chat_conv.turns[0].role, Role::Assistant);
        assert_eq!(app.chat_conv.turns[0].content, "Hello");
    }

    #[test]
    fn apply_ui_msg_telemetry_disconnected_flips_state() {
        let mut app = HelmApp::new(cfg());
        app.apply_ui_msg(UiMsg::Telemetry(TelemetryMsg::Snapshot(LiveStats {
            loaded_model: "m".into(),
            tok_s_decode: 10.0,
            ..Default::default()
        })));
        assert!(app.live_connected);
        assert_eq!(app.live.loaded_model, "m");

        app.apply_ui_msg(UiMsg::Telemetry(TelemetryMsg::Disconnected("x".into())));
        assert!(!app.live_connected);
        assert_eq!(app.live_last_error.as_deref(), Some("x"));
    }
}
