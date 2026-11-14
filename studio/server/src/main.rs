//! Kimodo Studio: yerel web sunucusu.
//! - Python Kimodo worker'ını alt süreç olarak çalıştırır, izler, çökerse yeniden başlatır.
//! - İş kuyruğu tutar; worker'a her seferinde tek iş gönderir.
//! - Kütüphane (üretilen animasyonlar) listeleme/yeniden adlandırma/kaydetme/silme.
//! - SSE ile canlı durum, ilerleme ve log akışı.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Json, Router,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    convert::Infallible,
    path::PathBuf,
    process::Stdio,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{ChildStdin, Command},
    sync::{broadcast, Mutex, Notify},
};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tower_http::services::ServeDir;

// ---------- yapılandırma ----------

#[derive(Clone)]
struct Config {
    python: PathBuf,
    worker_script: PathBuf,
    kimodo_repo: PathBuf,
    library: PathBuf,
    legacy_output: PathBuf,
    web: PathBuf,
    port: u16,
    blender: PathBuf,
    blender_script: PathBuf,
    skin_npz: PathBuf,
}

impl Config {
    fn from_env() -> Self {
        let studio = std::env::var("KIMODO_STUDIO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."));
        let workspace = std::env::var("KIMODO_WORKSPACE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
                    .join("kimodo-rx6650")
            });
        Config {
            python: workspace.join(".venv").join("Scripts").join("python.exe"),
            worker_script: studio.join("worker").join("kimodo_worker.py"),
            kimodo_repo: workspace.join("kimodo"),
            legacy_output: workspace.join("kimodo").join("output"),
            library: std::env::var("KIMODO_LIBRARY")
                .map(PathBuf::from)
                .unwrap_or_else(|_| studio.join("library")),
            web: studio.join("web"),
            blender: std::env::var("BLENDER")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe")
                }),
            blender_script: studio.join("blender").join("kimodo_to_fbx.py"),
            skin_npz: workspace
                .join(r"kimodo\kimodo\assets\skeletons\somaskel77\skin_standard.npz"),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8787),
        }
    }
}

// ---------- durum ----------

#[derive(Clone, Serialize, Deserialize)]
struct Job {
    id: String,
    name: String,
    prompt: String,
    duration: f64,
    steps: u32,
    seed: Option<u64>,
    #[serde(default)]
    postprocess: bool,
    #[serde(default = "yes")]
    ground_fix: bool,
    #[serde(default)]
    segments: Option<Value>,
    #[serde(default)]
    constraints: Option<Value>,
    status: String, // queued | running | done | failed | cancelled
    message: Option<String>,
    created: u64,
    started: Option<u64>,
    gen_sec: Option<f64>,
    checks: Option<Value>,
}

struct Worker {
    status: String, // starting | ready | busy | dead
    info: Value,
    stdin: Option<ChildStdin>,
    busy: Option<String>,
    restarts: u32,
    last_rss: Option<f64>,
}

struct App {
    cfg: Config,
    jobs: Mutex<Vec<Job>>,
    worker: Mutex<Worker>,
    logs: Mutex<VecDeque<String>>,
    events: broadcast::Sender<String>,
    restart: Notify,
    counter: std::sync::atomic::AtomicU64,
}

type Shared = Arc<App>;

fn yes() -> bool {
    true
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl App {
    fn emit(&self, v: Value) {
        let _ = self.events.send(v.to_string());
    }
    async fn log(&self, line: String) {
        let mut l = self.logs.lock().await;
        if l.len() >= 400 {
            l.pop_front();
        }
        l.push_back(line.clone());
        drop(l);
        self.emit(json!({"type": "log", "line": line}));
    }
    async fn worker_snapshot(&self) -> Value {
        let w = self.worker.lock().await;
        json!({"status": w.status, "info": w.info, "busy": w.busy, "restarts": w.restarts, "rss_gib": w.last_rss})
    }
    async fn emit_worker(&self) {
        let s = self.worker_snapshot().await;
        self.emit(json!({"type": "worker", "worker": s}));
    }
    async fn update_job<F: FnOnce(&mut Job)>(&self, id: &str, f: F) {
        let mut jobs = self.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|j| j.id == id) {
            f(j);
            let j = j.clone();
            drop(jobs);
            self.emit(json!({"type": "job", "job": j}));
        }
    }
}

async fn send_line(w: &mut Worker, v: &Value) -> bool {
    if let Some(stdin) = w.stdin.as_mut() {
        let line = format!("{}\n", v);
        stdin.write_all(line.as_bytes()).await.is_ok() && stdin.flush().await.is_ok()
    } else {
        false
    }
}

/// Worker boştaysa sıradaki işi gönder.
async fn dispatch(app: &Shared) {
    let mut w = app.worker.lock().await;
    if w.status != "ready" || w.busy.is_some() {
        return;
    }
    let mut jobs = app.jobs.lock().await;
    let Some(job) = jobs.iter_mut().find(|j| j.status == "queued") else {
        return;
    };
    let msg = json!({
        "id": job.id, "prompt": job.prompt, "duration": job.duration, "steps": job.steps,
        "seed": job.seed, "name": job.name, "out_dir": app.cfg.library, "postprocess": job.postprocess, "ground_fix": job.ground_fix,
        "segments": job.segments, "constraints": job.constraints,
    });
    if send_line(&mut w, &msg).await {
        job.status = "running".into();
        job.started = Some(now());
        w.busy = Some(job.id.clone());
        w.status = "busy".into();
        let j = job.clone();
        drop(jobs);
        drop(w);
        app.emit(json!({"type": "job", "job": j}));
        app.emit_worker().await;
    }
}

async fn handle_proto(app: &Shared, v: Value) {
    let ev = v["event"].as_str().unwrap_or("");
    match ev {
        "ready" => {
            {
                let mut w = app.worker.lock().await;
                w.status = "ready".into();
                w.last_rss = v["rss_gib"].as_f64();
                w.info = v.clone();
                let msg = json!({"cmd": "export_skin", "out_dir": app.cfg.library});
                send_line(&mut w, &msg).await;
            }
            app.log(format!("[studio] worker hazır: {}", v)).await;
            app.emit_worker().await;
            dispatch(app).await;
        }
        "done" | "error" => {
            let id = v["id"].as_str().unwrap_or("").to_string();
            let ok = ev == "done" && v["ok"].as_bool().unwrap_or(false);
            let msg = if ev == "error" {
                v["message"].as_str().map(String::from)
            } else if !ok {
                Some("Doğrulama başarısız (checks)".into())
            } else {
                None
            };
            if ev == "error" {
                if let Some(t) = v["trace"].as_str() {
                    app.log(format!("[worker hata] {}", t)).await;
                }
            }
            app.update_job(&id, |j| {
                j.status = if ok { "done" } else { "failed" }.into();
                j.message = msg;
                j.gen_sec = v["gen_sec"].as_f64();
                j.checks = v.get("checks").cloned();
            })
            .await;
            {
                let mut w = app.worker.lock().await;
                if w.busy.as_deref() == Some(id.as_str()) || ev == "error" {
                    w.busy = None;
                    if w.status == "busy" {
                        w.status = "ready".into();
                    }
                }
                if let Some(r) = v["rss_gib"].as_f64() {
                    w.last_rss = Some(r);
                }
            }
            app.emit_worker().await;
            if ev == "done" {
                app.emit(json!({"type": "library"}));
            }
            dispatch(app).await;
        }
        "converted" | "skin_ready" => {
            app.emit(json!({"type": "library"}));
        }
        "progress" => app.emit(json!({"type": "progress", "data": v})),
        "fatal" => app.log(format!("[worker FATAL] {}", v["message"])).await,
        _ => {}
    }
}

/// Worker'ı başlatır, izler, çökerse yeniden başlatır.
async fn supervisor(app: Shared) {
    let mut backoff = 3u64;
    loop {
        {
            let mut w = app.worker.lock().await;
            w.status = "starting".into();
            w.busy = None;
            w.stdin = None;
        }
        app.emit_worker().await;
        app.log(format!(
            "[studio] worker başlatılıyor: {}",
            app.cfg.worker_script.display()
        ))
        .await;

        let child = Command::new(&app.cfg.python)
            .arg("-u")
            .arg(&app.cfg.worker_script)
            .current_dir(&app.cfg.kimodo_repo)
            .env("TEXT_ENCODER_DEVICE", "cpu")
            .env("TEXT_ENCODER_MODE", "local")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                app.log(format!("[studio] worker başlatılamadı: {e}")).await;
                app.worker.lock().await.status = "dead".into();
                app.emit_worker().await;
                tokio::time::sleep(Duration::from_secs(30)).await;
                continue;
            }
        };
        app.worker.lock().await.stdin = child.stdin.take();

        let out = child.stdout.take().unwrap();
        let err = child.stderr.take().unwrap();
        let a1 = app.clone();
        let t_out = tokio::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(rest) = line.strip_prefix("@@") {
                    if let Ok(v) = serde_json::from_str::<Value>(rest) {
                        handle_proto(&a1, v).await;
                        continue;
                    }
                }
                a1.log(line).await;
            }
        });
        let a2 = app.clone();
        let t_err = tokio::spawn(async move {
            // tqdm \r ile yazar; satırları \r'den de böl.
            let mut reader = BufReader::new(err);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let s = String::from_utf8_lossy(&buf);
                        for part in s.split('\r').map(str::trim).filter(|p| !p.is_empty()) {
                            if part.contains("it/s]") || part.contains("s/it]") {
                                a2.emit(json!({"type": "tqdm", "line": part}));
                            } else {
                                a2.log(part.to_string()).await;
                            }
                        }
                    }
                }
            }
        });

        let started = std::time::Instant::now();
        let manual = tokio::select! {
            r = child.wait() => { app.log(format!("[studio] worker çıktı: {:?}", r)).await; false }
            _ = app.restart.notified() => { let _ = child.kill().await; app.log("[studio] worker yeniden başlatılıyor (istek)".into()).await; true }
        };
        let _ = t_out.await;
        t_err.abort();

        let busy = {
            let mut w = app.worker.lock().await;
            w.status = "dead".into();
            w.stdin = None;
            w.restarts += 1;
            w.busy.take()
        };
        if let Some(id) = busy {
            // Çökme sırasında çalışan işi tekrar kuyruğa koyma: aynı iş tekrar çökertebilir.
            app.update_job(&id, |j| {
                j.status = "failed".into();
                j.message = Some(
                    if manual {
                        "Worker yeniden başlatıldı"
                    } else {
                        "Worker çöktü (OOM?)"
                    }
                    .into(),
                );
            })
            .await;
        }
        app.emit_worker().await;
        if started.elapsed() > Duration::from_secs(600) || manual {
            backoff = 3;
        } else {
            backoff = (backoff * 2).min(120);
        }
        tokio::time::sleep(Duration::from_secs(if manual { 1 } else { backoff })).await;
    }
}

// ---------- yardımcılar ----------

fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 64
        && !n.starts_with('_')
        && n.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn slug(prompt: &str) -> String {
    let mut s = String::new();
    for c in prompt.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c);
        } else if !s.ends_with('-') {
            s.push('-');
        }
        if s.len() >= 32 {
            break;
        }
    }
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "motion".into()
    } else {
        s
    }
}

const EXTS: [&str; 6] = [".json", ".npz", ".bvh", ".view.bin", ".fbx", ".glb"];

fn err(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({"error": msg}))).into_response()
}

// ---------- HTTP ----------

#[derive(Deserialize)]
struct NewJob {
    #[serde(default)]
    prompt: String,
    duration: Option<f64>,
    steps: Option<u32>,
    seed: Option<u64>,
    name: Option<String>,
    count: Option<u32>,
    postprocess: Option<bool>,
    ground_fix: Option<bool>,
    segments: Option<Vec<Segment>>,
    constraints: Option<Vec<Value>>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Segment {
    prompt: String,
    duration: f64,
}

async fn create_job(State(app): State<Shared>, Json(req): Json<NewJob>) -> Response {
    let segs = req.segments.clone().filter(|v| !v.is_empty());
    if let Some(sg) = &segs {
        if sg.len() > 8 {
            return err(StatusCode::BAD_REQUEST, "en fazla 8 segment");
        }
        for x in sg {
            if x.prompt.trim().is_empty()
                || x.prompt.len() > 500
                || !(0.5..=10.0).contains(&x.duration)
            {
                return err(
                    StatusCode::BAD_REQUEST,
                    "her segmentte prompt dolu ve süre 0.5-10 sn olmalı",
                );
            }
        }
        if sg.iter().map(|x| x.duration).sum::<f64>() > 20.0 {
            return err(StatusCode::BAD_REQUEST, "toplam süre en fazla 20 sn");
        }
    }
    let prompt = match &segs {
        Some(sg) => sg
            .iter()
            .map(|x| x.prompt.trim())
            .collect::<Vec<_>>()
            .join(" | "),
        None => req.prompt.trim().to_string(),
    };
    if prompt.is_empty() || prompt.len() > 4000 {
        return err(StatusCode::BAD_REQUEST, "prompt boş ya da çok uzun");
    }
    let duration = match &segs {
        Some(sg) => sg.iter().map(|x| x.duration).sum(),
        None => req.duration.unwrap_or(3.0),
    };
    if segs.is_none() && !(0.5..=10.0).contains(&duration) {
        return err(StatusCode::BAD_REQUEST, "süre 0.5-10 sn olmalı");
    }
    if req.constraints.as_ref().map_or(0, |c| c.len()) > 32 {
        return err(StatusCode::BAD_REQUEST, "en fazla 32 kısıt");
    }
    let steps = req.steps.unwrap_or(50).clamp(5, 200);
    if let Some(n) = &req.name {
        if !n.is_empty() && !valid_name(n) {
            return err(
                StatusCode::BAD_REQUEST,
                "isim yalnızca harf, rakam, - ve _ içerebilir",
            );
        }
    }
    let count = req.count.unwrap_or(1).clamp(1, 8);
    let mut created = vec![];
    for i in 0..count {
        let n = app
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let id = format!("{}{:03}", now(), n % 1000);
        let base = req
            .name
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| slug(&prompt));
        let segments = segs.as_ref().map(|sg| serde_json::to_value(sg).unwrap());
        let mut name = if count > 1 {
            format!("{base}-{i:02}")
        } else {
            base.clone()
        };
        let mut k = 1;
        while app.cfg.library.join(format!("{name}.json")).exists()
            || app
                .jobs
                .lock()
                .await
                .iter()
                .any(|j| j.name == name && j.status != "failed")
        {
            k += 1;
            name = format!(
                "{}-{k}",
                if count > 1 {
                    format!("{base}-{i:02}")
                } else {
                    base.clone()
                }
            );
        }
        let job = Job {
            id,
            name,
            prompt: prompt.clone(),
            duration,
            steps,
            seed: req.seed.map(|s| s + i as u64),
            postprocess: req.postprocess.unwrap_or(true),
            ground_fix: req.ground_fix.unwrap_or(true),
            segments,
            constraints: req.constraints.clone().map(Value::from),
            status: "queued".into(),
            message: None,
            created: now(),
            started: None,
            gen_sec: None,
            checks: None,
        };
        {
            let mut jobs = app.jobs.lock().await;
            jobs.push(job.clone());
            let excess = jobs.len().saturating_sub(300);
            jobs.drain(0..excess);
        }
        app.emit(json!({"type": "job", "job": job}));
        created.push(job);
    }
    dispatch(&app).await;
    Json(json!({"jobs": created})).into_response()
}

async fn cancel_job(State(app): State<Shared>, Path(id): Path<String>) -> Response {
    let mut found = false;
    app.update_job(&id, |j| {
        if j.status == "queued" {
            j.status = "cancelled".into();
            found = true;
        }
    })
    .await;
    if found {
        Json(json!({"ok": true})).into_response()
    } else {
        err(
            StatusCode::CONFLICT,
            "yalnızca sıradaki iş iptal edilebilir",
        )
    }
}

async fn status(State(app): State<Shared>) -> Json<Value> {
    let jobs = app.jobs.lock().await.clone();
    let logs: Vec<String> = app
        .logs
        .lock()
        .await
        .iter()
        .rev()
        .take(150)
        .rev()
        .cloned()
        .collect();
    Json(json!({"worker": app.worker_snapshot().await, "jobs": jobs, "logs": logs}))
}

async fn restart_worker(State(app): State<Shared>) -> Json<Value> {
    app.restart.notify_one();
    Json(json!({"ok": true}))
}

async fn library(State(app): State<Shared>) -> Json<Value> {
    let mut items = vec![];
    if let Ok(mut rd) = tokio::fs::read_dir(&app.cfg.library).await {
        while let Ok(Some(e)) = rd.next_entry().await {
            let fname = e.file_name().to_string_lossy().to_string();
            let Some(stem) = fname.strip_suffix(".json") else {
                continue;
            };
            if stem.starts_with('_') {
                continue;
            }
            if let Ok(txt) = tokio::fs::read_to_string(e.path()).await {
                if let Ok(mut v) = serde_json::from_str::<Value>(&txt) {
                    v["name"] = json!(stem);
                    v["has_view"] =
                        json!(app.cfg.library.join(format!("{stem}.view.bin")).exists());
                    v["has_bvh"] = json!(app.cfg.library.join(format!("{stem}.bvh")).exists());
                    v["has_fbx"] = json!(app.cfg.library.join(format!("{stem}.fbx")).exists());
                    items.push(v);
                }
            }
        }
    }
    items.sort_by(|a, b| {
        b["created"]
            .as_str()
            .unwrap_or("")
            .cmp(a["created"].as_str().unwrap_or(""))
    });
    Json(json!({"items": items}))
}

#[derive(Deserialize)]
struct Patch {
    saved: Option<bool>,
    rename: Option<String>,
    note: Option<String>,
}

async fn patch_item(
    State(app): State<Shared>,
    Path(name): Path<String>,
    Json(p): Json<Patch>,
) -> Response {
    if !valid_name(&name) {
        return err(StatusCode::BAD_REQUEST, "geçersiz isim");
    }
    let lib = &app.cfg.library;
    let meta_path = lib.join(format!("{name}.json"));
    let Ok(txt) = tokio::fs::read_to_string(&meta_path).await else {
        return err(StatusCode::NOT_FOUND, "bulunamadı");
    };
    let mut meta: Value = serde_json::from_str(&txt).unwrap_or(json!({}));
    if let Some(s) = p.saved {
        meta["saved"] = json!(s);
    }
    if let Some(n) = p.note {
        meta["note"] = json!(n);
    }
    let mut final_name = name.clone();
    if let Some(new) = p.rename.filter(|n| n != &name) {
        if !valid_name(&new) {
            return err(
                StatusCode::BAD_REQUEST,
                "isim yalnızca harf, rakam, - ve _ içerebilir",
            );
        }
        if lib.join(format!("{new}.json")).exists() {
            return err(StatusCode::CONFLICT, "bu isimde bir kayıt var");
        }
        for ext in EXTS {
            let from = lib.join(format!("{name}{ext}"));
            if from.exists() {
                if let Err(e) = tokio::fs::rename(&from, lib.join(format!("{new}{ext}"))).await {
                    return err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
                }
            }
        }
        final_name = new;
    }
    let path = lib.join(format!("{final_name}.json"));
    if let Err(e) = tokio::fs::write(&path, serde_json::to_string_pretty(&meta).unwrap()).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    app.emit(json!({"type": "library"}));
    Json(json!({"ok": true, "name": final_name})).into_response()
}

async fn delete_item(State(app): State<Shared>, Path(name): Path<String>) -> Response {
    if !valid_name(&name) {
        return err(StatusCode::BAD_REQUEST, "geçersiz isim");
    }
    for ext in EXTS {
        let _ = tokio::fs::remove_file(app.cfg.library.join(format!("{name}{ext}"))).await;
    }
    app.emit(json!({"type": "library"}));
    Json(json!({"ok": true})).into_response()
}

/// kimodo/output altındaki eski NPZ/BVH'leri kütüphaneye kopyalar.
async fn import_legacy(State(app): State<Shared>) -> Response {
    let mut imported = vec![];
    let Ok(mut rd) = tokio::fs::read_dir(&app.cfg.legacy_output).await else {
        return err(StatusCode::NOT_FOUND, "eski çıktı klasörü yok");
    };
    let mut w = app.worker.lock().await;
    if w.stdin.is_none() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "worker hazır değil");
    }
    while let Ok(Some(e)) = rd.next_entry().await {
        let fname = e.file_name().to_string_lossy().to_string();
        let Some(stem) = fname.strip_suffix(".npz") else {
            continue;
        };
        let name = format!("legacy-{}", slug(stem));
        let lib = &app.cfg.library;
        if lib.join(format!("{name}.json")).exists() {
            continue;
        }
        let _ = tokio::fs::copy(e.path(), lib.join(format!("{name}.npz"))).await;
        let bvh = app.cfg.legacy_output.join(format!("{stem}.bvh"));
        if bvh.exists() {
            let _ = tokio::fs::copy(&bvh, lib.join(format!("{name}.bvh"))).await;
        }
        let meta = json!({"id": name, "prompt": format!("(eski çıktı) {stem}"), "fps": 30.0, "created": "2026-11-14T01:00:00", "legacy": true});
        let _ = tokio::fs::write(lib.join(format!("{name}.json")), meta.to_string()).await;
        let cmd = json!({"cmd": "convert", "npz": lib.join(format!("{name}.npz")), "view": lib.join(format!("{name}.view.bin"))});
        send_line(&mut w, &cmd).await;
        imported.push(name);
    }
    drop(w);
    app.emit(json!({"type": "library"}));
    Json(json!({"imported": imported})).into_response()
}

/// Headless Blender: BVH -> skinli SOMA karakteri FBX + GLB (doğrulamalı, hata meta'ya yazılır).
async fn export_fbx(State(app): State<Shared>, Path(name): Path<String>) -> Response {
    if !valid_name(&name) {
        return err(StatusCode::BAD_REQUEST, "geçersiz isim");
    }
    let lib = app.cfg.library.clone();
    let (bvh, npz) = (
        lib.join(format!("{name}.bvh")),
        lib.join(format!("{name}.npz")),
    );
    if !bvh.exists() || !npz.exists() {
        return err(StatusCode::NOT_FOUND, "BVH/NPZ yok");
    }
    app.emit(json!({"type": "export", "name": name, "status": "running"}));
    let out = Command::new(&app.cfg.blender)
        .args(["-b", "--factory-startup", "-P"])
        .arg(&app.cfg.blender_script)
        .arg("--")
        .arg("--bvh")
        .arg(&bvh)
        .arg("--npz")
        .arg(&npz)
        .arg("--skin")
        .arg(&app.cfg.skin_npz)
        .arg("--out")
        .arg(lib.join(format!("{name}.fbx")))
        .arg("--glb")
        .arg(lib.join(format!("{name}.glb")))
        .arg("--verify")
        .kill_on_drop(true)
        .output()
        .await;
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Blender başlatılamadı: {e}"),
            )
        }
    };
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let report = text
        .lines()
        .find_map(|l| l.strip_prefix("[k6] REPORT "))
        .and_then(|s| serde_json::from_str::<Value>(s).ok());
    let Some(report) = report else {
        let tail: String = text
            .lines()
            .rev()
            .take(15)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        app.log(format!("[export] {name} BAŞARISIZ:\n{tail}")).await;
        app.emit(json!({"type": "export", "name": name, "status": "failed"}));
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Blender dönüşümü başarısız (log'a bak)",
        );
    };
    let max_err = report["vertex_err_max_m"]
        .as_object()
        .map(|m| m.values().filter_map(|v| v.as_f64()).fold(0.0, f64::max))
        .unwrap_or(f64::NAN);
    let meta_path = lib.join(format!("{name}.json"));
    if let Ok(txt) = tokio::fs::read_to_string(&meta_path).await {
        if let Ok(mut meta) = serde_json::from_str::<Value>(&txt) {
            meta["export"] = json!({"fbx": true, "glb": true, "vertex_err_max_m": max_err, "ok": max_err < 1e-3});
            let _ =
                tokio::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).await;
        }
    }
    app.log(format!(
        "[export] {name}: FBX+GLB hazır, en büyük köşe hatası {:.1} µm",
        max_err * 1e6
    ))
    .await;
    app.emit(json!({"type": "export", "name": name, "status": "done"}));
    app.emit(json!({"type": "library"}));
    Json(json!({"ok": true, "vertex_err_max_m": max_err})).into_response()
}

async fn events(State(app): State<Shared>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = app.events.subscribe();
    let stream =
        BroadcastStream::new(rx).filter_map(|m| m.ok().map(|s| Ok(Event::default().data(s))));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[tokio::main]
async fn main() {
    let cfg = Config::from_env();
    std::fs::create_dir_all(&cfg.library).expect("library klasörü oluşturulamadı");
    for (what, p) in [
        ("python", &cfg.python),
        ("worker", &cfg.worker_script),
        ("kimodo repo", &cfg.kimodo_repo),
    ] {
        if !p.exists() {
            eprintln!("UYARI: {what} bulunamadı: {}", p.display());
        }
    }
    let (tx, _) = broadcast::channel(1024);
    let app: Shared = Arc::new(App {
        cfg: cfg.clone(),
        jobs: Mutex::new(vec![]),
        worker: Mutex::new(Worker {
            status: "starting".into(),
            info: json!({}),
            stdin: None,
            busy: None,
            restarts: 0,
            last_rss: None,
        }),
        logs: Mutex::new(VecDeque::new()),
        events: tx,
        restart: Notify::new(),
        counter: Default::default(),
    });
    tokio::spawn(supervisor(app.clone()));

    let router = Router::new()
        .route("/api/status", get(status))
        .route("/api/events", get(events))
        .route("/api/jobs", post(create_job))
        .route("/api/jobs/{id}", delete(cancel_job))
        .route("/api/worker/restart", post(restart_worker))
        .route("/api/library", get(library))
        .route("/api/library/{name}", delete(delete_item).patch(patch_item))
        .route("/api/import-legacy", post(import_legacy))
        .route("/api/library/{name}/export", post(export_fbx))
        .nest_service("/files", ServeDir::new(&cfg.library))
        .fallback_service(ServeDir::new(&cfg.web))
        .with_state(app);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
    println!("Kimodo Studio: http://{addr}");
    println!("  kütüphane: {}", cfg.library.display());
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("port açılamadı");
    axum::serve(listener, router).await.unwrap();
}
