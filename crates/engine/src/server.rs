//! OpenAI-compatible HTTP front end (`/v1/chat/completions`, `/v1/completions`).
//!
//! Inference is serialised behind one engine mutex and runs on a blocking
//! thread: a P100 decode step is 10-15 ms of pure GPU work, and batching is done
//! by the engine's column tile, not by the HTTP layer.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{Engine, Tuning};
use crate::sampler::SamplingParams;
use crate::tokenizer::Role;

#[derive(Clone)]
pub struct AppState {
    engine: Arc<Mutex<Engine>>,
    model_name: String,
}

#[derive(Deserialize, Default)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Deserialize, Default)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: Option<ChatMessage>,
    delta: Option<serde_json::Value>,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

fn id() -> String {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{ns:x}")
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn params(temperature: Option<f32>, top_p: Option<f32>, top_k: Option<usize>, seed: Option<u64>) -> SamplingParams {
    let mut p = SamplingParams::default();
    if let Some(t) = temperature {
        p.temperature = t;
    }
    if let Some(v) = top_p {
        p.top_p = v;
    }
    if let Some(v) = top_k {
        p.top_k = v;
    }
    if let Some(s) = seed {
        p.seed = s;
    }
    p
}

fn chat_prompt(eng: &Engine, msgs: &[ChatMessage]) -> Vec<u32> {
    let mut system: Option<String> = None;
    let mut turns: Vec<(Role, String)> = Vec::new();
    for m in msgs {
        match m.role.as_str() {
            "system" => system = Some(m.content.clone()),
            "assistant" => turns.push((Role::Assistant, m.content.clone())),
            _ => turns.push((Role::User, m.content.clone())),
        }
    }
    eng.model
        .tokenizer
        .render_chat(system.as_deref(), &turns, true)
}

async fn models(State(st): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [ModelCard { id: st.model_name.clone(), object: "model", owned_by: "trav-p100" }]
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn generate_response(
    engine: Arc<Mutex<Engine>>,
    prompt: Vec<u32>,
    model: String,
    p: SamplingParams,
    max_tokens: usize,
    stream: bool,
) -> axum::response::Response {
    let n_prompt = prompt.len();
    if stream {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        tokio::task::spawn_blocking(move || {
            let mut e = engine.blocking_lock();
            let ctx = e.tuning.ctx;
            let mut seq = match e.new_seq(ctx, p) {
                Ok(s) => s,
                Err(err) => {
                    let _ = tx.blocking_send(Ok(Event::default()
                        .data(serde_json::json!({"error": err.to_string()}).to_string())));
                    return;
                }
            };
            let r = e.generate(&mut seq, &prompt, max_tokens, |text| {
                let payload = serde_json::json!({
                    "id": "chatcmpl",
                    "object": "chat.completion.chunk",
                    "created": now(),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                });
                tx.blocking_send(Ok(Event::default().data(payload.to_string()))).is_ok()
            });
            if let Err(err) = r {
                let _ = tx.blocking_send(Ok(Event::default()
                    .data(serde_json::json!({"error": err.to_string()}).to_string())));
            }
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        });
        Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let out = tokio::task::spawn_blocking(move || -> Result<(String, usize)> {
            let mut e = engine.blocking_lock();
            let ctx = e.tuning.ctx;
            let mut seq = e.new_seq(ctx, p)?;
            let mut text = String::new();
            let n = e.generate(&mut seq, &prompt, max_tokens, |s| {
                text.push_str(s);
                true
            })?;
            Ok((text, n))
        })
        .await;

        match out {
            Ok(Ok((text, n))) => Json(ChatResponse {
                id: id(),
                object: "chat.completion",
                created: now(),
                model,
                choices: vec![Choice {
                    index: 0,
                    message: Some(ChatMessage { role: "assistant".into(), content: text }),
                    delta: None,
                    finish_reason: Some("stop".into()),
                }],
                usage: Usage {
                    prompt_tokens: n_prompt,
                    completion_tokens: n,
                    total_tokens: n_prompt + n,
                },
            })
            .into_response(),
            Ok(Err(e)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response(),
        }
    }
}

fn check_ctx(engine: &Engine, n_prompt: usize, max_tokens: usize) -> Option<axum::response::Response> {
    if n_prompt + max_tokens > engine.tuning.ctx {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": format!(
                    "prompt {n_prompt} + max_tokens {max_tokens} exceeds ctx {}",
                    engine.tuning.ctx)}})),
            )
                .into_response(),
        );
    }
    None
}

async fn chat(State(st): State<AppState>, Json(req): Json<ChatRequest>) -> axum::response::Response {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let model = req.model.clone().unwrap_or_else(|| st.model_name.clone());
    let engine = st.engine.clone();
    let (prompt, too_long) = {
        let e = engine.lock().await;
        let prompt = chat_prompt(&e, &req.messages);
        let n = prompt.len();
        (prompt, check_ctx(&e, n, max_tokens))
    };
    if let Some(resp) = too_long {
        return resp;
    }
    let p = params(req.temperature, req.top_p, req.top_k, req.seed);
    generate_response(engine, prompt, model, p, max_tokens, req.stream.unwrap_or(false)).await
}

async fn completions(
    State(st): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let model = req.model.clone().unwrap_or_else(|| st.model_name.clone());
    let engine = st.engine.clone();
    let (prompt, too_long) = {
        let e = engine.lock().await;
        let prompt = e.model.tokenizer.encode(&req.prompt);
        let n = prompt.len();
        (prompt, check_ctx(&e, n, max_tokens))
    };
    if let Some(resp) = too_long {
        return resp;
    }
    let p = params(req.temperature, req.top_p, None, req.seed);
    generate_response(engine, prompt, model, p, max_tokens, req.stream.unwrap_or(false)).await
}

pub struct ServeConfig {
    pub addr: String,
    pub ctx: usize,
}

pub async fn serve(engine: Engine, model_name: String, cfg: ServeConfig) -> Result<()> {
    let tuning: Tuning = engine.tuning.clone();
    if tuning.ctx != cfg.ctx {
        bail!("tuning/ctx mismatch");
    }
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        model_name,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cfg.addr)
        .await
        .map_err(|e| anyhow::anyhow!("bind {}: {e}", cfg.addr))?;
    println!("listening on http://{}/v1", cfg.addr);
    axum::serve(listener, app).await?;
    Ok(())
}
