"""
Flask backend for DSDP annotation tool.

Transforms the old visualization webapp into an annotator for arcs, labels,
and threads with LLM-assisted suggestions.

Run:
  python -u webapp/app.py
  # then open http://localhost:5050
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory, session

ROOT = Path(__file__).resolve().parent.parent   # app/ -> project root (data/, .env live here)


def _load_env():
    """Best-effort environment hydration so explain/Claude work regardless of HOW
    the app is launched -- a bare shell won't inherit a PyCharm run-config's vars,
    and that silently disables the explanation engine after a restart. Sources are
    tried in precedence order and NONE overrides an already-set var:
      1) KEY=VALUE lines from .env, then .env.example (this project keeps its working
         CLAUDE_CODE_OAUTH_TOKEN directly in .env.example);
      2) on Windows, the User environment scope (where this project's keys live).
    """
    for fname in (".env", ".env.example"):
        envf = ROOT / fname
        if not envf.exists():
            continue
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "HF_TOKEN"):
                    if os.environ.get(name):
                        continue
                    try:
                        val, _ = winreg.QueryValueEx(key, name)
                        if val:
                            os.environ[name] = str(val)
                    except FileNotFoundError:
                        pass
        except Exception:
            pass


_load_env()

from relations import (
    CANONICAL_RELATIONS_EXT, normalize_relation, RELATION_DEFS,
    build_relation_defs, DATASET_RELATIONS, _REL_DEF, relation_def,
)
from corpora import (DATASETS, DATA_DIR, get_prompts, find_consecutive_groups,
                       merge_dialogue, load_records)
from engines import Engine, register, get_engine, REGISTRY

ANN_DIR = ROOT / "data" / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)
PRECOMPUTE_DIR = ANN_DIR / "_precompute"
PRECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"   # torch fallback id
# The tool's OWN dialogue LLM (Qwen3). The webapp PINS this; it must NOT reuse a
# model reserved for the user's other experiments (RESERVED_MODELS) — not even as
# a fallback — so it can never steal the user's loaded model or its GPU.
DEFAULT_LMS_MODEL = os.environ.get("DSDP_DIALOGUE_LLM", "qwen/qwen3-32b")  # 32B (loaded FULL-GPU via the lmstudio SDK → fast)
RESERVED_MODELS = set(filter(None,
    os.environ.get("DSDP_RESERVED_MODELS", "ft_2_qwen_merged").split(",")))
GPU_INDEX = os.environ.get("DSDP_GPU", "1")   # tool loads models on GPU 1 only
# DDPE explanation student (LLaMA3 + LoRA) served by src/suggestion_engine.py (GPU 1).
SUGGESTION_ENGINE_URL = os.environ.get("DSDP_SUGGESTION_ENGINE_URL", "http://127.0.0.1:8092")
DDPE_DATASETS = {"stac", "molweni"}   # DDPE adapters only exist for these
DEDISCO_URL = os.environ.get("DSDP_DEDISCO_URL", "http://127.0.0.1:8093")
# Which fine-tuned DDPE adapter the PARSER uses, INDEPENDENT of the dialogue dataset.
# Default: the Molweni-fine-tuned parser, used even on STAC dialogues. Set "" to make the
# adapter follow the dataset, or override via DSDP_DDPE_ADAPTER.
DDPE_ADAPTER = os.environ.get("DSDP_DDPE_ADAPTER", "molweni")
def _ddpe_ds(dataset):
    """Adapter to request from the DDPE server for a given dialogue dataset."""
    return DDPE_ADAPTER or dataset

# Per-engine / per-role preferred LM Studio model ids (env-overridable). When the
# preferred model is not loaded, _generate() falls back to the first loaded LLM
# that is NOT reserved — never the user's reserved model, never a hard fail.
ROLE_MODELS = {
    # discourse pre-annotator engines (parent/arcs/incremental/label/threads/full)
    "preannotator": os.environ.get("DSDP_MODEL_PREANNOTATOR", DEFAULT_LMS_MODEL),
    "zeroshot":     os.environ.get("DSDP_MODEL_ZEROSHOT", DEFAULT_LMS_MODEL),
    "dsdp":         os.environ.get("DSDP_MODEL_DSDP", DEFAULT_LMS_MODEL),
    # reasoning + explanation model (used by /api/explain/arc)
    "reasoning": os.environ.get("DSDP_MODEL_REASONING",
                                "deepseek-r1-distill-qwen-32b"),
}
DEFAULT_ROLE = "preannotator"
# The study uses the official 16 SDRT relations — drop the extras (Sequence is CDU-level;
# Confirmation_Q is not part of the set).
_EXCLUDED_RELS = {"Sequence", "Confirmation_Q"}
RELATIONS = sorted(r for r in CANONICAL_RELATIONS_EXT if r not in _EXCLUDED_RELS)
LMS_CMD = shutil.which("lms") or "lms"
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"

app = Flask(__name__, static_folder="static")

# ── Study auth + SQLite data collection ──────────────────────────────────
# Login = a participant code (e.g. USER_007) + ONE shared study password. On the
# first login a profile form (demographics + background/training) is collected.
# Everything (users, action logs, questionnaires, annotations) is stored in SQLite.
DB_PATH = ROOT / "db" / "mudipa.db"
STUDY_PASSWORD = os.environ.get("MUDIPA_STUDY_PASSWORD", "catan-study")
app.secret_key = os.environ.get("MUDIPA_SECRET_KEY", "dsdp-mudipa-dev-secret-change-me")
# Session cookie: persists 7 days. For the https/ngrok tunnel (or any embedded/cross-site
# serving) set MUDIPA_COOKIE_CROSS_SITE=1 → SameSite=None + Secure, so the browser keeps and
# sends the cookie; otherwise SameSite=Lax for plain local http.
_cross_site = os.environ.get("MUDIPA_COOKIE_CROSS_SITE", "") in ("1", "true", "True", "yes")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None" if _cross_site else "Lax",
    SESSION_COOKIE_SECURE=_cross_site,
    PERMANENT_SESSION_LIFETIME=30 * 60,   # 30 minutes, for security
)
_db_lock = threading.Lock()


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          participant TEXT PRIMARY KEY,
          created_at  INTEGER,
          last_login  INTEGER,
          profile     TEXT,           -- JSON: demographics + background/training
          onboarded   INTEGER DEFAULT 0   -- 1 once the intro + tutorial were shown (first login)
        );
        CREATE TABLE IF NOT EXISTS events (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          participant TEXT, ts INTEGER, step INTEGER,
          condition   TEXT, action TEXT, payload TEXT
        );
        CREATE TABLE IF NOT EXISTS questionnaires (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          participant TEXT, ts INTEGER, condition TEXT,
          dataset TEXT, idx INTEGER, data TEXT
        );
        CREATE TABLE IF NOT EXISTS annotations (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          participant TEXT, ts INTEGER, condition TEXT,
          dataset TEXT, split TEXT, idx INTEGER, annotation TEXT
        );
        """)
        # migration: add `onboarded` to pre-existing users tables
        cols = [r[1] for r in c.execute("PRAGMA table_info(users)")]
        if "onboarded" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN onboarded INTEGER DEFAULT 0")


_init_db()


def _require_auth():
    """True if the current session cleared the common-password gate."""
    return bool(session.get("mudipa_auth"))


def _session_pid(fallback=None):
    """The logged-in participant for this session (falls back to a given value)."""
    return _safe_pid(session.get("participant") or fallback)


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    payload = request.json or {}
    raw = payload.get("participant")
    pid = _safe_pid(raw)
    if not raw or pid == "anon":
        return jsonify({"error": "participant code required"}), 400
    if str(payload.get("password") or "") != STUDY_PASSWORD:
        return jsonify({"error": "wrong study password"}), 403
    now = int(time.time() * 1000)
    onboarded = False
    with _db_lock, _db() as c:
        row = c.execute("SELECT profile, onboarded FROM users WHERE participant=?", (pid,)).fetchone()
        if row is None:
            c.execute("INSERT INTO users(participant, created_at, last_login, profile, onboarded) "
                      "VALUES(?,?,?,NULL,0)", (pid, now, now))
            has_profile = False
        else:
            c.execute("UPDATE users SET last_login=? WHERE participant=?", (now, pid))
            has_profile = bool(row["profile"])
            onboarded = bool(row["onboarded"])
    session.permanent = True
    session["mudipa_auth"] = True
    session["participant"] = pid
    return jsonify({"status": "ok", "participant": pid,
                    "needs_profile": not has_profile, "onboarded": onboarded})


@app.route("/api/auth/guest", methods=["POST"])
def auth_guest():
    """Frictionless demo entry: log in as a pre-onboarded experimenter with no study
    password and no demographics, landing directly in free-exploration mode (dataset
    picker, upload, and both engines). Intended for the public live demo and self-hosted
    trials; the guided study still uses /api/auth/login. Disable with MUDIPA_DEMO_MODE=0."""
    if os.environ.get("MUDIPA_DEMO_MODE", "1").lower() in ("0", "false", "no", "off"):
        return jsonify({"error": "guest mode disabled"}), 403
    import uuid
    pid = "exp_guest_" + uuid.uuid4().hex[:8]   # 'exp' prefix -> free-exploration mode in the UI
    now = int(time.time() * 1000)
    with _db_lock, _db() as c:
        c.execute("INSERT INTO users(participant, created_at, last_login, profile, onboarded) "
                  "VALUES(?,?,?,?,1)", (pid, now, now, json.dumps({"guest": True})))
    session.permanent = True
    session["mudipa_auth"] = True
    session["participant"] = pid
    return jsonify({"status": "ok", "participant": pid,
                    "needs_profile": False, "onboarded": True, "guest": True})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    if not _require_auth():
        return jsonify({"authed": False})
    pid = _session_pid()
    with _db() as c:
        row = c.execute("SELECT profile, onboarded FROM users WHERE participant=?", (pid,)).fetchone()
    if row is None:
        # stale cookie pointing to a user that no longer exists (e.g. DB wiped) → drop it.
        session.clear()
        return jsonify({"authed": False})
    return jsonify({"authed": True, "participant": pid,
                    "needs_profile": not (row and row["profile"]),
                    "onboarded": bool(row and row["onboarded"])})


@app.route("/api/auth/onboarded", methods=["POST"])
def auth_onboarded():
    """Mark the first-login intro + tutorial as shown (so they don't repeat)."""
    if not _require_auth():
        return jsonify({"error": "not authenticated"}), 401
    pid = _session_pid()
    with _db_lock, _db() as c:
        c.execute("UPDATE users SET onboarded=1 WHERE participant=?", (pid,))
    return jsonify({"status": "ok"})


@app.route("/api/auth/profile", methods=["POST"])
def auth_profile():
    if not _require_auth():
        return jsonify({"error": "not authenticated"}), 401
    payload = request.json or {}
    pid = _session_pid(payload.get("participant"))
    with _db_lock, _db() as c:
        c.execute("UPDATE users SET profile=? WHERE participant=?",
                  (json.dumps(payload.get("profile") or {}, ensure_ascii=False), pid))
    return jsonify({"status": "ok"})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"status": "ok"})


_model = None
_tokenizer = None
_model_name = None
_model_lock = threading.Lock()


# ── Model (lazy) ─────────────────────────────────────────────────────────

def _ensure_model(name):
    global _model, _tokenizer, _model_name
    with _model_lock:
        if _model is not None and _model_name == name:
            return _model, _tokenizer
        import warnings, torch, transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        transformers.logging.set_verbosity_error()
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="auto")
        _model.eval()
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model_name = name
        transformers.logging.set_verbosity_warning()
        return _model, _tokenizer


def _lms_model_id(m):
    """Canonical id used by the OpenAI-compatible API for a loaded model."""
    return m.get("identifier") or m.get("modelKey") or m.get("path")


def _pick_loaded_llm(llms, role=None, model=None):
    """Choose which loaded LLM to use.

    Preference order:
      1. explicit `model` id (if loaded),
      2. the model mapped to `role` in ROLE_MODELS (if loaded),
      3. the first loaded LLM (legacy behaviour / graceful fallback).
    Returns the chosen loaded-model dict, or None if no LLM is loaded.
    Never raises just because a preferred model is not loaded.
    """
    if not llms:
        return None
    preferred = model or (ROLE_MODELS.get(role) if role else None)
    if preferred:
        for m in llms:
            if _lms_model_id(m) == preferred or m.get("modelKey") == preferred:
                return m
    # Fallback: first loaded LLM that is NOT reserved for the user's other tasks.
    # (Never grab a RESERVED model — return None so the tool degrades gracefully.)
    for m in llms:
        if _lms_model_id(m) not in RESERVED_MODELS and m.get("modelKey") not in RESERVED_MODELS:
            return m
    return None


def _generate(system, user, max_new_tokens=512, role=DEFAULT_ROLE, model=None):
    """Prefer LM Studio (if a model is loaded); fall back to transformers.

    `role` selects a preferred model via ROLE_MODELS; `model` forces a specific
    LM Studio model id. Either is honoured only if that model is currently
    loaded; otherwise we fall back to the first loaded LLM so a single-model
    setup behaves exactly as before.
    """
    # Reasoning/explanation calls keep chain-of-thought; structured suggestion
    # calls (parents, labels, threads) disable it for speed and reliable JSON.
    no_think = (role != "reasoning")
    try:
        loaded = _lms_ps()
        llms = [m for m in loaded if m.get("type") == "llm"]
        chosen = _pick_loaded_llm(llms, role=role, model=model)
        if chosen is not None:
            return _lms_chat(chosen, system, user, max_new_tokens,
                             no_think=no_think)
    except Exception as ex:
        print(f"[_generate] LM Studio path failed: {ex}", flush=True)
    # Fallback: local transformers — only if installed. If neither a usable LM
    # Studio model nor torch is available, DEGRADE GRACEFULLY (return "") so the
    # endpoint returns 'no suggestion' instead of a 500.
    try:
        from DSDP_llm_dp_torch import build_chat_messages, batch_generate
        model, tok = _ensure_model(DEFAULT_MODEL)
        msgs = build_chat_messages(system, user, no_think=no_think)
        resp = batch_generate(model, tok, [msgs], batch_size=1,
                              max_new_tokens=max_new_tokens)
        return resp[0] if resp else ""
    except Exception as ex:
        print(f"[_generate] no usable backend (LM Studio model not matched; "
              f"torch fallback unavailable): {ex}", flush=True)
        return ""


# ── LM Studio helpers ────────────────────────────────────────────────────

def _run_lms(args, timeout=30):
    try:
        r = subprocess.run([LMS_CMD, *args], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.SubprocessError) as e:
        return 1, "", str(e)


def _lms_ls():
    code, out, _ = _run_lms(["ls", "--json"], timeout=10)
    if code == 0:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return []
    return []


def _lms_ps():
    code, out, _ = _run_lms(["ps", "--json"], timeout=10)
    if code == 0:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return []
    return []


def _lms_server_up():
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _lms_chat_raw(model, system, user, max_tokens=512, no_think=False,
                  temperature=0.1, logprobs=False, top_logprobs=0, n=1):
    """Low-level LM Studio chat call. Returns the parsed OpenAI-style response
    dict (so callers can read choices / logprobs). `_lms_chat` wraps this."""
    model_id = _lms_model_id(model)
    # Qwen3 is a hybrid reasoning model: by default it emits a <think>...</think>
    # block before the answer. For structured JSON calls that block burns the
    # token budget and gets truncated before the JSON is produced (empty result).
    # The Qwen3 soft-switch "/no_think" in the prompt disables reasoning for the
    # turn; harmless on non-thinking models (treated as literal text, ignored).
    if no_think:
        system = (system or "") + "\n\n/no_think"
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if n and n > 1:
        payload["n"] = n
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs or 10
    r = requests.post(LM_STUDIO_CHAT_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def _lms_chat(model, system, user, max_tokens=512, no_think=False):
    data = _lms_chat_raw(model, system, user, max_tokens=max_tokens, no_think=no_think)
    return data["choices"][0]["message"]["content"]


def _extract_json_loose(text):
    if not text:
        return {}
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        for end in range(len(text), start, -1):
            if text[end - 1] != closer:
                continue
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    return {}


# ── Dataset helpers ──────────────────────────────────────────────────────

def _load_dataset(dataset, split):
    return load_records(dataset, split)


def _ann_path(dataset, split, participant=None):
    if participant:
        return ANN_DIR / f"{participant}_{dataset}_{split}.json"
    return ANN_DIR / f"{dataset}_{split}.json"


def _load_anns(dataset, split, participant=None):
    p = _ann_path(dataset, split, participant)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            pass
    return {}


def _save_anns(dataset, split, anns, participant=None):
    _ann_path(dataset, split, participant).write_text(
        json.dumps(anns, ensure_ascii=False, indent=1), encoding="utf-8")


def _default_ann(dialogue):
    """Initial annotation: ALWAYS empty — the canvas starts clean and the human
    annotates incrementally. Gold is never seeded (load it explicitly via the UI)."""
    return {"arcs": [], "threads": [], "notes": ""}


def _load_dialogue(dataset, split, idx):
    """Load a single dialogue dict by (dataset, split, idx).
    Mirrors how get_dialogue() resolves a dialogue from request params."""
    data = _load_dataset(dataset, split)
    return data[int(idx)]


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(app.send_static_file("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _dataset_available(cfg):
    """A dataset is offered in the UI only if its data is actually on disk (or it was
    just uploaded). Keeps the picker to the datasets that really load — STAC and the
    multimodal DraDDP/MODDP here — instead of the many registered-but-absent corpora."""
    if cfg.get("uploaded"):
        return True
    return any((DATA_DIR / f).exists() for f in cfg.get("files", {}).values())


@app.route("/api/config")
def get_config():
    from dataset_papers import paper_for
    lms_available = _run_lms(["--version"], timeout=3)[0] == 0
    return jsonify({
        "datasets": [{"name": k, "splits": list(v["files"].keys()),
                      "paper": paper_for(k, v)}          # source publication (§provenance)
                     for k, v in DATASETS.items() if _dataset_available(v)],
        "relations": RELATIONS,
        "model": DEFAULT_MODEL,
        "lms_available": lms_available,
        "lms_server_up": _lms_server_up() if lms_available else False,
    })


# ── LM Studio management endpoints ───────────────────────────────────────

@app.route("/api/lms/models")
def lms_models():
    """Returns all available and currently loaded models."""
    available = _lms_ls()
    loaded = _lms_ps()
    loaded_keys = {m.get("identifier") or m.get("modelKey") for m in loaded}
    llms = [m for m in available if m.get("type") == "llm"]
    return jsonify({
        "available": [{
            "modelKey": m.get("modelKey"),
            "displayName": m.get("displayName"),
            "path": m.get("path"),
            "sizeBytes": m.get("sizeBytes"),
            "paramsString": m.get("paramsString"),
            "architecture": m.get("architecture"),
            "quant": (m.get("quantization") or {}).get("name"),
            "ctx": m.get("maxContextLength"),
            "loaded": (m.get("modelKey") in loaded_keys
                       or m.get("path") in loaded_keys),
        } for m in llms],
        "loaded": [{
            "identifier": m.get("identifier"),
            "modelKey": m.get("modelKey"),
            "displayName": m.get("displayName"),
            "type": m.get("type"),
        } for m in loaded],
        "server_up": _lms_server_up(),
    })


@app.route("/api/models/roles")
def models_roles():
    """Role -> preferred model id map, plus whether each is currently loaded.

    Lets the UI surface which model serves each role (and warn when a
    preferred model is not loaded and the fallback will kick in).
    """
    loaded = _lms_ps()
    loaded_ids = set()
    for m in loaded:
        if m.get("type") != "llm":
            continue
        loaded_ids.add(_lms_model_id(m))
        if m.get("modelKey"):
            loaded_ids.add(m.get("modelKey"))
    first_loaded = next((_lms_model_id(m) for m in loaded
                         if m.get("type") == "llm"), None)
    roles = {}
    for role, model_id in ROLE_MODELS.items():
        is_loaded = model_id in loaded_ids
        roles[role] = {
            "model": model_id,
            "loaded": is_loaded,
            # which model a request for this role would actually hit right now
            "effective": model_id if is_loaded else first_loaded,
        }
    return jsonify({
        "roles": roles,
        "default_role": DEFAULT_ROLE,
        "server_up": _lms_server_up(),
    })


def _load_llm_full_gpu(model_id):
    """Load a dialogue LLM with FULL GPU offload via the official lmstudio SDK.
    The CLI `lms load --gpu max` under-offloads big models (qwen3-32b → only ~18 GB
    on GPU → ~89 s/call); the SDK's gpu.ratio=1.0 puts it fully on GPU (~0.3 s/gen).
    Unloads everything first; falls back to the CLI load if the SDK is unavailable."""
    # If the target model is ALREADY loaded — e.g. the user loaded it in the LM
    # Studio GUI with GPU Offload = MAX (the only reliable way to fully offload a
    # big model) — KEEP it as-is. A code reload would under-offload it (→ slow).
    try:
        if any(_lms_model_id(m) == model_id or m.get("modelKey") == model_id
               for m in _lms_ps()):
            return True
    except Exception:
        pass
    try:
        import lmstudio as lms
        c = lms.get_default_client()
        for m in c.llm.list_loaded():            # unload ALL existing instances first
            try: c.llm.unload(getattr(m, "identifier", str(m)))
            except Exception: pass
        # load_new_instance forces a FRESH load with the gpu.ratio=1.0 config;
        # plain llm() is get-or-load and would reuse a partially-offloaded copy.
        c.llm.load_new_instance(model_id, config={"gpu": {"ratio": 1.0}})
        return True
    except Exception as ex:
        print(f"[engine] SDK full-GPU load failed ({ex}); CLI fallback", flush=True)
        _run_lms(["unload", "--all"], timeout=60)
        code, _, _ = _run_lms(["load", model_id, "--yes", "--gpu", "1.0"], timeout=300)
        return code == 0


@app.route("/api/engine/activate", methods=["POST"])
def engine_activate():
    """DDPE is the default and only GPU engine now — the LM Studio 32b is NEVER
    loaded anymore (per user decision: "useremo sempre ddpe"). GPU 1 hosts DDPE
    persistently.
      - 'ddpe'   : ensure the DDPE student (LLaMA3+LoRA) is loaded on GPU 1.
      - 'claude' : explanations via the Anthropic API (no GPU); DDPE stays loaded.
      - others   : no-op on the GPU (the 32b is not loaded).
    """
    payload = request.json or {}
    engine = payload.get("engine", "ddpe").lower()
    eng = get_engine(engine)
    if eng is None:
        return jsonify({"engine": engine, "steps": [
            f"'{engine}': unknown engine -- use one of {REGISTRY.names()}"],
            "ready": False})
    out = eng.activate(payload.get("dataset"))
    out["engine"] = engine
    return jsonify(out)


@app.route("/api/lms/load", methods=["POST"])
def lms_load():
    payload = request.json or {}
    model_key = payload.get("model", "").strip()
    if not model_key:
        return jsonify({"error": "missing model"}), 400
    args = ["load", model_key, "--yes"]
    # Default to GPU 1 (the idle card): GPU 0 is reserved for the user's own work.
    # Pass gpu:0 explicitly (or "auto") to override.
    gpu = payload.get("gpu", 1)
    if gpu not in (None, "", "auto"):
        args += ["--gpu", f"{float(gpu):.1f}"]
    code, out, err = _run_lms(args, timeout=600)
    if code != 0:
        return jsonify({"error": err or out}), 500
    return jsonify({"status": "ok", "output": out})


@app.route("/api/lms/unload", methods=["POST"])
def lms_unload():
    payload = request.json or {}
    model_key = payload.get("model", "").strip()
    args = ["unload"]
    if model_key:
        args.append(model_key)
    else:
        args.append("--all")
    code, out, err = _run_lms(args, timeout=30)
    if code != 0:
        return jsonify({"error": err or out}), 500
    return jsonify({"status": "ok", "output": out})


@app.route("/api/dialogues")
def list_dialogues():
    dataset = request.args.get("dataset", "stac")
    split = request.args.get("split", "test")
    data = _load_dataset(dataset, split)
    anns = _load_anns(dataset, split)
    out = []
    for i, d in enumerate(data):
        out.append({
            "idx": i,
            "id": d.get("id", f"idx_{i}"),
            "n_edus": len(d["edus"]),
            "n_rels": len(d.get("relations", [])),
            "annotated": str(i) in anns,
        })
    return jsonify(out)


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Upload a dialogue file and register it as a runtime, UI-selectable dataset.

    Accepts JSON (a single dialogue object or a list of them) or JSONL (one dialogue
    per line). Each dialogue follows the MuDiPA schema:
        {"id": <str>,
         "edus": [{"speaker": <str>, "text": <str>}, ...],   # required, non-empty
         "relations": [{"type": <str>, "x": <int>, "y": <int>}, ...]}  # optional
    Relations may be omitted/empty to annotate a pre-segmented dialogue from scratch.
    A plain list of strings is also accepted for `edus` (speaker defaults to "").
    """
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file (send multipart form field 'file')"}), 400
    raw = f.read().decode("utf-8-sig", errors="replace").strip()
    if not raw:
        return jsonify({"error": "empty file"}), 400
    # JSON (object or list) first, then JSONL fallback.
    try:
        obj = json.loads(raw)
        records = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        try:
            records = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        except json.JSONDecodeError as ex:
            return jsonify({"error": f"not valid JSON or JSONL: {ex}"}), 400
    if not records:
        return jsonify({"error": "no dialogues found"}), 400

    clean = []
    for i, r in enumerate(records):
        if not isinstance(r, dict) or not isinstance(r.get("edus"), list) or not r["edus"]:
            return jsonify({"error": f"dialogue {i}: missing non-empty 'edus' list"}), 400
        edus = []
        for j, e in enumerate(r["edus"]):
            if isinstance(e, str):
                e = {"speaker": "", "text": e}
            if not isinstance(e, dict) or "text" not in e:
                return jsonify({"error": f"dialogue {i}, EDU {j}: each EDU needs a 'text' field"}), 400
            edus.append({"speaker": str(e.get("speaker", "")), "text": str(e["text"]),
                         "speechturn": e.get("speechturn", j)})
        rels = []
        for rel in (r.get("relations") or []):
            try:
                rels.append({"type": str(rel.get("type", "")),
                             "x": int(rel["x"]), "y": int(rel["y"])})
            except (KeyError, TypeError, ValueError):
                return jsonify({"error": f"dialogue {i}: each relation needs integer 'x','y'"}), 400
        clean.append({"id": str(r.get("id", f"dlg_{i}")), "edus": edus, "relations": rels})

    base = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(f.filename)[0])[:40] or "corpus"
    name = f"upload_{base}"
    rel_path = f"uploads/{name}.jsonl"
    with open(DATA_DIR / rel_path, "w", encoding="utf-8") as out:
        for r in clean:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Register at runtime so it shows up in /api/config and loads like any dataset.
    DATASETS[name] = {
        "loader": "native", "format": "jsonl",
        "context": f"user-uploaded dialogues ({f.filename})",
        "relations": DATASETS["stac"]["relations"],
        "files": {"test": rel_path},
        "uploaded": True,
    }
    return jsonify({"dataset": name, "split": "test", "n_dialogues": len(clean)})


@app.route("/api/dialogue")
def get_dialogue():
    dataset = request.args.get("dataset", "stac")
    split = request.args.get("split", "test")
    idx = int(request.args.get("idx", 0))
    data = _load_dataset(dataset, split)
    d = data[idx]
    gold = [{"x": int(r["x"]), "y": int(r["y"]),
             "type": normalize_relation(r.get("type", ""))}
            for r in d.get("relations", [])]
    pid = _session_pid()
    anns = _load_anns(dataset, split, pid)
    ann = anns.get(str(idx)) or _default_ann(d)
    speakers = [e["speaker"] for e in d["edus"]]
    # Precomputed LLM suggestions, if any
    precomp_path = PRECOMPUTE_DIR / f"{dataset}_{split}_{idx}.json"
    precomp = None
    if precomp_path.exists():
        try:
            precomp = json.loads(precomp_path.read_text(encoding="utf-8"))
        except Exception:
            precomp = None
    # Optional multimodal payload (DraDDP/MODDP/CoMuMDR-style). Degrades to
    # text-only when the dataset has no media fields.
    mdir_edu = DATA_DIR / dataset / "media"

    def _edu_out(e):
        o = {"speaker": e["speaker"], "text": e["text"]}
        clip = e.get("clip") or (
            [e["start"], e.get("end")] if "start" in e else None)
        if clip:
            o["clip"] = clip          # [start_sec, end_sec] for this EDU
        # Per-EDU media segment: `clip_src` is an extension-less base (e.g.
        # "clips/test/dia0_utt1"); serve the actual file, preferring .mp4 (video +
        # audio) over audio-only .flac/.wav/.m4a. Omitted if no file exists.
        base = e.get("clip_src")
        if base:
            for ext, kind in ((".mp4", "video"), (".m4a", "audio"),
                              (".wav", "audio"), (".flac", "audio")):
                if (mdir_edu / f"{base}{ext}").exists():
                    o["clip_src"] = f"/media/{dataset}/{base}{ext}"
                    o["clip_type"] = kind
                    break
        return o

    media = d.get("media")            # {"type":"video"|"audio","src":"<file>"}
    if media:
        raw_src = str(media.get("src", ""))
        is_url = raw_src.startswith(("http://", "https://", "/"))
        mdir = DATA_DIR / dataset / "media"
        # Only surface media whose file actually exists: corpora that ship the
        # annotations but not the audio/video (e.g. MODDP) degrade to text-only
        # instead of a broken <video>/<audio> element.
        if is_url or (mdir / raw_src).exists():
            if not is_url:
                media = {**media, "src": f"/media/{dataset}/{raw_src}"}
            # Attach precomputed frame artefacts (precompute_media.py) when present:
            #   peaks -> waveform envelope json; frames -> video keyframe index json.
            did = str(d.get("id", f"idx_{idx}"))
            if (mdir / "peaks" / f"{did}.json").exists():
                media = {**media, "peaks": f"/media/{dataset}/peaks/{did}.json"}
            if (mdir / "frames" / f"{did}.json").exists():
                media = {**media, "frames": f"/media/{dataset}/frames/{did}.json"}
        else:
            media = None

    return jsonify({
        "idx": idx,
        "id": d.get("id", f"idx_{idx}"),
        "edus": [_edu_out(e) for e in d["edus"]],
        "gold": gold,
        "annotation": ann,
        "precompute": precomp,
        "media": media,
    })


@app.route("/media/<dataset>/<path:fname>")
def serve_media(dataset, fname):
    """Serve dialogue media (video/audio) from data/<dataset>/media/.
    Supports HTTP range requests so <video>/<audio> seeking works."""
    media_dir = DATA_DIR / dataset / "media"
    return send_from_directory(media_dir, fname, conditional=True)


@app.route("/api/annotation", methods=["POST"])
def save_annotation():
    payload = request.json or {}
    dataset = payload["dataset"]
    split = payload["split"]
    idx = str(int(payload["idx"]))
    ann = payload["annotation"]
    pid = _session_pid(payload.get("participant"))
    anns = _load_anns(dataset, split, pid)
    anns[idx] = ann
    _save_anns(dataset, split, anns, pid)
    with _db_lock, _db() as c:        # also keep a per-participant, timestamped copy
        c.execute("INSERT INTO annotations(participant, ts, condition, dataset, split, idx, annotation) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (pid, int(time.time() * 1000),
                   payload.get("condition"), dataset, split, int(payload["idx"]),
                   json.dumps(ann, ensure_ascii=False)))
    return jsonify({"status": "ok"})


@app.route("/api/annotation", methods=["DELETE"])
def delete_annotation():
    dataset = request.args.get("dataset", "stac")
    split = request.args.get("split", "test")
    idx = str(int(request.args.get("idx", 0)))
    pid = _session_pid()
    anns = _load_anns(dataset, split, pid)
    if idx in anns:
        del anns[idx]
        _save_anns(dataset, split, anns, pid)
    return jsonify({"status": "ok"})


# ── LLM suggestion endpoints ─────────────────────────────────────────────

def _dialogue_text(edus, speakers):
    return "\n".join(
        f"  ({i}) {speakers[i]}: \"{edus[i]}\""
        for i in range(len(edus))
    )


SYS_ARCS = """\
You analyze multi-party dialogues. For a target utterance, identify which \
previous utterances it is directly responding to.

A direct link means the target utterance would not have been said the same \
way without the source utterance. Score each candidate on a 1-5 scale:

5 = clear direct response / answer / reaction
4 = likely linked
3 = plausible but uncertain
2 = weak connection
1 = unrelated (siblings, new topic)

Independent reactions to the same earlier statement are NOT linked to each \
other — they are siblings.

Respond ONLY with JSON: {"<candidate_id>": {"score": <1-5>, "reason": "<short>"}, ...}"""


@app.route("/api/suggest/arcs", methods=["POST"])
def suggest_arcs():
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    target = int(payload["target"])
    if target < 1 or target >= len(edus):
        return jsonify({"error": "invalid target"}), 400

    candidates = list(range(max(0, target - 10), target))
    dial = _dialogue_text(edus, speakers)
    user = (f"Dialogue:\n{dial}\n\n"
            f"Target utterance: ({target}) {speakers[target]}: "
            f"\"{edus[target]}\"\n\n"
            f"Candidate parents: {candidates}\n"
            f"Score each on 1-5 for likelihood of being a parent of the target.")

    raw = _generate(SYS_ARCS, user, max_new_tokens=512, role="preannotator")
    parsed = _extract_json_loose(raw)
    scored = []
    for cid in candidates:
        entry = parsed.get(str(cid), {}) if isinstance(parsed, dict) else {}
        if isinstance(entry, dict):
            try:
                sc = float(entry.get("score", 0))
            except (TypeError, ValueError):
                sc = 0
            scored.append({
                "source": cid,
                "score": sc,
                "reason": str(entry.get("reason", ""))[:200],
            })
        else:
            scored.append({"source": cid, "score": 0, "reason": ""})
    scored.sort(key=lambda x: -x["score"])
    return jsonify({"target": target, "candidates": scored, "raw": raw[:400]})


# ── Interpretability: faithful occlusion-based arc saliency ──────────────
# Model-free leave-one-out perturbation → faithful BY CONSTRUCTION, unlike
# free-text LLM rationales which are plausible but unfaithful
# (cf. search-based Saliency Map Verbalization, Feldhus et al., NLRSE@ACL 2023).
SYS_PAIR_SCORE = """\
You are reading a multi-party board game dialogue (Settlers of Catan).
Score, from 0 to 100, how strongly the TARGET utterance is a direct discourse \
response to the SOURCE utterance (answers, reacts to, continues, clarifies it).
0 = unrelated, 100 = unmistakably a direct response.
Respond ONLY with JSON: {"score": <0-100>}"""


def _score_pair(edus, speakers, source, target, target_text=None,
                role=DEFAULT_ROLE):
    """LLM score in [0,1] that `target` directly responds to `source`.
    `target_text` overrides edus[target] (used for occlusion).
    `role` selects which model to prefer (see ROLE_MODELS)."""
    tgt = target_text if target_text is not None else edus[target]
    ctx = "\n".join(f"  ({i}) {speakers[i]}: \"{edus[i]}\""
                    for i in range(max(0, source - 2), target))
    user = (f"Context:\n{ctx}\n\n"
            f"SOURCE: ({source}) {speakers[source]}: \"{edus[source]}\"\n"
            f"TARGET: ({target}) {speakers[target]}: \"{tgt}\"")
    try:
        raw = _generate(SYS_PAIR_SCORE, user, max_new_tokens=24, role=role)
    except Exception:
        return None  # no LLM backend available
    parsed = _extract_json_loose(raw)
    try:
        return max(0.0, min(1.0, float(parsed.get("score", 0)) / 100.0))
    except (TypeError, ValueError):
        return 0.0


# ── C2: G-Eval confidence module (Liu et al., 2023) ──────────────────────
# CoT + form-filling. Returns {score in [0,1], confidence in [0,1]}.
# This is the UNCERTAINTY signal shown in the UI — never the accuracy metric.

_GEVAL_VALMAP = {
    "yesno": {"No": 0.0, "Yes": 1.0},
    "1-5":   {"1": 0.0, "2": 0.25, "3": 0.5, "4": 0.75, "5": 1.0},
}


def _geval_rubric(scale):
    return ["No", "Yes"] if scale == "yesno" else ["1", "2", "3", "4", "5"]


def _parse_score_token(txt, rubric):
    """Pull the form-filled answer token out of a completion."""
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.DOTALL)
    m = re.search(r"SCORE:\s*([A-Za-z0-9]+)", txt, re.IGNORECASE)
    if m:
        cand = m.group(1)
        for t in rubric:
            if cand.lower() == t.lower():
                return t
    found = [t for t in rubric if re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])",
                                            txt, re.IGNORECASE)]
    return found[-1] if found else None


def _geval_logprob_dist(model, system, user, rubric):
    """Best-effort: one logprob call → probability distribution over rubric
    tokens (read the top_logprobs of the answer token). Returns {tok: prob} or
    None (caller then falls back to k-sampling)."""
    try:
        data = _lms_chat_raw(model, system, user, max_tokens=160, no_think=True,
                             temperature=0.0, logprobs=True, top_logprobs=12)
        content = (data["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            return None
        rl = {t.lower(): t for t in rubric}
        # find the answer token: a rubric token, preferably just after "SCORE"/":"
        cand_idx = None
        for i, tok in enumerate(content):
            if str(tok.get("token", "")).strip().lower() in rl:
                prev = " ".join(str(content[j].get("token", "")) for j in range(max(0, i - 3), i))
                if "SCORE" in prev.upper() or ":" in prev:
                    cand_idx = i
                    break
                if cand_idx is None:
                    cand_idx = i
        if cand_idx is None:
            return None
        import math
        tops = content[cand_idx].get("top_logprobs") or []
        mass = {t: 0.0 for t in rubric}
        for alt in tops:
            tk = str(alt.get("token", "")).strip()
            if tk.lower() in rl:
                mass[rl[tk.lower()]] += math.exp(alt.get("logprob", -50))
        tot = sum(mass.values())
        if tot <= 0:
            return None
        return {t: mass[t] / tot for t in rubric}
    except Exception:
        return None


def _geval(system, user, scale="yesno", k=5):
    """G-Eval confidence. score = probability-weighted mean over the rubric
    (normalized 0–1); confidence = 1 − normalized entropy of that distribution.
    Prefers a single logprob call; falls back to k-sample answer frequency.
    Uses the reasoning role (CoT allowed). Returns {score:None,confidence:0} with
    no LLM backend."""
    import math
    rubric = _geval_rubric(scale)
    valmap = _GEVAL_VALMAP[scale]
    instr = ("\n\nReason in ONE short sentence, then output a final line exactly:\n"
             f"SCORE: <X>\nwhere <X> is one of: {', '.join(rubric)}.")
    sys_p = (system or "") + instr
    try:
        llms = [m for m in _lms_ps() if m.get("type") == "llm"]
        model = _pick_loaded_llm(llms, role="reasoning")
    except Exception:
        model = None
    if model is None:
        return {"score": None, "confidence": 0.0}

    dist = _geval_logprob_dist(model, sys_p, user, rubric)
    if dist is None:
        counts = {t: 0 for t in rubric}
        got = 0
        for _ in range(max(1, k)):
            try:
                data = _lms_chat_raw(model, sys_p, user, max_tokens=140,
                                     temperature=0.4, no_think=True)
                tok = _parse_score_token(data["choices"][0]["message"]["content"], rubric)
            except Exception:
                tok = None
            if tok:
                counts[tok] += 1
                got += 1
        if got == 0:
            return {"score": None, "confidence": 0.0}
        dist = {t: counts[t] / got for t in rubric}

    score = sum(dist[t] * valmap[t] for t in rubric)
    ent = -sum(p * math.log(p) for p in dist.values() if p > 0)
    maxent = math.log(len(rubric)) or 1.0
    conf = 1.0 - ent / maxent
    return {"score": round(score, 3), "confidence": round(max(0.0, min(1.0, conf)), 3)}


@app.route("/api/explain/arc", methods=["POST"])
def explain_arc():
    """Faithful occlusion (leave-one-out) saliency for one arc source→target.

    Removes each word of the target utterance, re-scores the arc, and reports
    the confidence drop. Faithful by construction (model-free perturbation),
    in the spirit of search-based SMV (Feldhus et al., 2023). Degrades
    gracefully (all-zero saliency) when no LLM backend is connected.
    """
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    source = int(payload["source"])
    target = int(payload["target"])
    if not (0 <= source < target < len(edus)):
        return jsonify({"error": "need 0 <= source < target < n"}), 400

    words = edus[target].split()
    MAX_TOK = 30
    truncated = len(words) > MAX_TOK
    idxs = list(range(min(len(words), MAX_TOK)))

    dataset = payload.get("dataset", "stac")
    # Faithful occlusion re-scoring goes through the engine registry: prefer DDPE
    # (the deployed parser -> saliency faithful to what actually parses), and fall
    # back to the local LM Studio LLM only when DDPE can't score this dataset.
    scorer = get_engine("ddpe")
    if not (scorer and scorer.can_score_pair and scorer.supports_dataset(dataset)):
        scorer = get_engine("zeroshot")

    base = scorer.score_pair(edus, speakers, source, target, dataset)
    if base is None:
        return jsonify({
            "source": source, "target": target,
            "backend_available": False,
            "saliency": [], "base_score": None,
            "verbalization": f"No scoring backend available for '{scorer.name}' "
                             "(DDPE unreachable on :8092, or no LM Studio) — "
                             "cannot compute saliency.",
            "method": f"occlusion (leave-one-out) · {scorer.name}", "faithful": True,
        })
    sal = []
    for k in idxs:
        occ = " ".join(w for j, w in enumerate(words) if j != k)
        s = scorer.score_pair(edus, speakers, source, target, dataset, target_text=occ)
        sal.append({"i": k, "token": words[k],
                    "saliency": round(base - (s if s is not None else base), 4)})

    ranked = sorted(sal, key=lambda x: -x["saliency"])
    top = [t for t in ranked if t["saliency"] > 0][:5]
    verbal = ((f"The link ({source}→{target}) hinges on: "
               + ", ".join(f"“{t['token']}”" for t in top))
              if top else
              f"No single word strongly drives the link ({source}→{target}); "
              "evidence is diffuse (or no LLM backend connected).")

    # CONTRASTIVE relation justification (Pillar C). This is a *plausible* model
    # rationale, NOT a faithful attribution — so we (a) GROUND it in the faithful
    # salient tokens above and (b) flag it `faithful: False` to keep the
    # distinction explicit (Feldhus et al.: free-text rationales are plausible,
    # not faithful). The occlusion saliency stays the faithful signal.
    rels = payload.get("relations") or DATASET_RELATIONS.get(dataset, DATASET_RELATIONS["stac"])
    contrastive = None
    toks = ", ".join(t["token"] for t in top) or "(evidence is diffuse)"
    try:
        sys_c = ("You label SDRT discourse relations in multi-party dialogue. Given a SOURCE "
                 "and TARGET utterance and the words the link most depends on, choose the TWO "
                 "most likely relations and explain why the FIRST holds and the SECOND does not, "
                 "citing those words. Reply ONLY JSON: "
                 '{"relations":["R1","R2"],"justification":"R1 not R2 because ..."}')
        usr_c = (f"SOURCE ({source}, {speakers[source]}): {edus[source]}\n"
                 f"TARGET ({target}, {speakers[target]}): {edus[target]}\n"
                 f"The link most depends on these words: {toks}\n"
                 f"Allowed relations: {', '.join(rels)}")
        raw = _reason_generate(sys_c, usr_c, max_new_tokens=240)
        parsed = _extract_json_loose(raw) if raw else None
        if isinstance(parsed, dict) and parsed.get("relations"):
            contrastive = {
                "relations": parsed.get("relations", [])[:2],
                "justification": parsed.get("justification", ""),
                "grounded_in": [t["token"] for t in top],
                "faithful": False,   # plausible model rationale, grounded but not faithful
            }
    except Exception:
        contrastive = None

    return jsonify({
        "source": source, "target": target,
        "base_score": round(base, 4),
        "saliency": sal,
        "verbalization": verbal,
        "method": f"occlusion (leave-one-out) · {scorer.name}",
        "faithful": True,                  # the saliency signal is faithful
        "contrastive": contrastive,        # the relation rationale is plausible (faithful: False)
        "truncated": truncated,
        "calls": len(idxs) + 1,
    })


SYS_PARENT = """\
You are reading a multi-party board game dialogue (Settlers of Catan).

For the target utterance, name the utterance(s) it is directly responding to. \
A parent is a previous utterance whose content the target directly addresses, \
answers, reacts to, or continues.

If the target starts a new topic or is a purely independent statement, return an \
empty list.

Ground your judgement in the SDRT discourse relation set (the relation that would \
label each parent -> target arc):
__RELDEFS__

Respond ONLY with JSON: {"parents": [<ids>], "scores": {"<id>": <1-5>, ...}, \
"types": {"<id>": "<relation>", ...}, "reason": "<short>"}"""


@app.route("/api/suggest/parent", methods=["POST"])
def suggest_parent():
    """Raw labeling: 'who is the parent of X?' — no scoring, just named parents."""
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    target = int(payload["target"])
    if target < 1 or target >= len(edus):
        return jsonify({"error": "invalid target"}), 400

    window = "\n".join(
        f"  ({i}) {speakers[i]}: \"{edus[i]}\""
        for i in range(target + 1)
    )
    user = (f"Dialogue:\n{window}\n\n"
            f"Who is the parent of utterance ({target}) "
            f"{speakers[target]}: \"{edus[target]}\"?")

    raw = _generate(SYS_PARENT.replace("__RELDEFS__", RELATION_DEFS), user,
                    max_new_tokens=420, role="preannotator")
    parsed = _extract_json_loose(raw)
    parents = []
    reason = ""
    scores, types = {}, {}
    if isinstance(parsed, dict):
        for p in parsed.get("parents", []):
            try:
                pv = int(p)
            except (TypeError, ValueError):
                continue
            if 0 <= pv < target:
                parents.append(pv)
        reason = str(parsed.get("reason", ""))[:600]
        for k, v in (parsed.get("scores", {}) or {}).items():
            try:
                kid, sc = int(k), float(v)
            except (TypeError, ValueError):
                continue
            if 0 <= kid < target:
                scores[str(kid)] = max(1.0, min(5.0, sc))
        for k, v in (parsed.get("types", {}) or {}).items():
            try:
                kid = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= kid < target and str(v).strip():
                types[str(kid)] = str(v).strip()
    return jsonify({"target": target, "parents": parents, "reason": reason,
                    "scores": scores, "types": types, "raw": raw[:400]})


# ── C1: the LLM EXPLAINS candidate arcs (link + relation), it does NOT decide ──
# The human chooses every link and relation; these prompts only produce evidence.

SYS_LINK_EXPLAIN = """\
You are a discourse-annotation ASSISTANT for a multi-party dialogue. You do NOT \
decide attachments — the human annotator does. For a CANDIDATE attachment from an \
earlier utterance (SOURCE) to a later utterance (TARGET), explain in 1-2 sentences \
the evidence for and (if any) against TARGET attaching to SOURCE: what in TARGET \
responds to / answers / continues / reacts to SOURCE. Then state whether TARGET \
attaches to SOURCE and your confidence 1-5 (5 = certain it attaches).

Respond ONLY with JSON: {"explanation": "<1-2 sentences>", \
"attaches": "Yes"|"No", "confidence": <1-5>}"""

SYS_LINK_JUDGE = """\
You judge discourse attachment in a multi-party dialogue. Does the TARGET utterance \
attach directly to the SOURCE utterance (respond to / answer / continue / react to \
it)?"""

SYS_REL_EXPLAIN = """\
You are a discourse-annotation ASSISTANT. For the discourse arc SOURCE -> TARGET, \
pick the single most likely SDRT relation and explain in 1-2 sentences why it fits. \
You suggest, you do not decide.

SDRT discourse relations (choose exactly one as the main relation):
__RELDEFS__

Also rate how well R1 fits the arc on a 1-5 scale ("fit", 5 = perfect fit).

Respond ONLY with JSON: {"relation": "<R1>", "explanation": "<why R1>", \
"fit": <1-5>}"""

SYS_REL_JUDGE = """\
You judge SDRT discourse relations. On a 1-5 scale, how well does the relation \
"__REL__" describe the arc SOURCE -> TARGET? (5 = perfect fit, 1 = wrong relation.)"""


def _conf_from_1to5(v):
    """Map a 1-5 form-filled rating to a [0,1] confidence."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, (x - 1.0) / 4.0)), 3)


def _explain_candidate(edus, speakers, p, t, rigorous=False):
    """C1: for candidate arc p->t, return SEPARATE link and relation explanations,
    each with a confidence. Suggests evidence only — never commits an annotation.

    Two modes for the confidence signal:
      - default (interactive): form-filled G-Eval (CoT + 1-5 rating folded into the
        explanation call) → 2 LLM calls total, ~30s. Used live / on-demand.
      - rigorous=True (precompute, C6): distribution-based `_geval` (k-sampling,
        ~12 calls). Slow but entropy-based; latency hidden behind precompute.
    (LM Studio returns no token logprobs here, so the 1-call logprob path is N/A.)"""
    ctx = "\n".join(f"  ({i}) {speakers[i]}: \"{edus[i]}\""
                    for i in range(max(0, p - 2), t + 1))
    pair = (f"SOURCE ({p}) {speakers[p]}: \"{edus[p]}\"\n"
            f"TARGET ({t}) {speakers[t]}: \"{edus[t]}\"")
    base_user = f"Dialogue context:\n{ctx}\n\nArc under consideration:\n{pair}"

    # LINK explanation (+ folded Yes/No + 1-5 confidence). no_think → clean JSON.
    lraw = _generate(SYS_LINK_EXPLAIN, base_user, max_new_tokens=260, role="preannotator")
    lp = _extract_json_loose(lraw) or {}
    link_expl = str(lp.get("explanation", "")).strip()[:500] \
        or re.sub(r"<think>.*?</think>", "", lraw, flags=re.DOTALL).strip()[:500]
    attaches = str(lp.get("attaches", "")).strip().lower().startswith("y")

    # RELATION explanation (+ relation, 1-5 fit) in one call.
    rraw = _generate(SYS_REL_EXPLAIN.replace("__RELDEFS__", RELATION_DEFS),
                     base_user, max_new_tokens=340, role="preannotator")
    rp = _extract_json_loose(rraw) or {}
    rel = str(rp.get("relation", "")).strip()
    if rel and rel not in CANONICAL_RELATIONS_EXT:
        from relations import normalize_relation_ext
        rel = normalize_relation_ext(rel) or rel

    if rigorous:
        lc = _geval(SYS_LINK_JUDGE, base_user, scale="yesno", k=5)
        link_score, link_conf = lc.get("score"), lc.get("confidence", 0.0)
        rc = (_geval(SYS_REL_JUDGE.replace("__REL__", rel), base_user, scale="1-5", k=5)
              if rel else {"score": None, "confidence": 0.0})
        rel_fit, rel_conf = rc.get("score"), rc.get("confidence", 0.0)
    else:
        link_score = 1.0 if attaches else 0.0
        link_conf = _conf_from_1to5(lp.get("confidence"))
        rel_fit = round(float(rp.get("fit")) / 5.0, 3) if str(rp.get("fit", "")).strip() else None
        rel_conf = _conf_from_1to5(rp.get("fit"))

    return {
        "parent": p,
        "link": {"explanation": link_expl, "score": link_score, "confidence": link_conf},
        "relation": {"candidate": rel,
                     "explanation": str(rp.get("explanation", "")).strip()[:500],
                     "fit": rel_fit, "confidence": rel_conf},
    }


def _ddpe_explain(edus, speakers, p, t, dataset="stac"):
    """Call the DDPE serving process (src/suggestion_engine.py, GPU 1, :8092) to explain
    arc p->t with the DDPE student. Returns a C1-shaped candidate dict or None."""
    try:
        r = requests.post(f"{SUGGESTION_ENGINE_URL}/ddpe/explain", json={
            "edus": [{"speaker": speakers[i], "text": edus[i]} for i in range(len(edus))],
            "source": p, "target": t, "dataset": _ddpe_ds(dataset)}, timeout=300)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── C1b-Claude: use Claude as the DDPE *teacher* (the role GPT-4 Turbo plays in
# Liu et al. 2025) directly at inference — rich link+relation+contrastive
# explanations in the DDPE format, via the official Anthropic SDK. API, no GPU. ──
CLAUDE_MODEL = os.environ.get("DSDP_CLAUDE_MODEL", "claude-opus-4-8")
_claude_client = None

# Pre-generated explanation hints (built by build_explanation_hints.py).
# Keyed by "<dlg_idx>_<src>_<tgt>". Loaded once on first use.
_EXPL_HINTS = None
_EXPL_HINTS_PATH = Path(__file__).parent / "data" / "explanation_hints.json"

def _get_expl_hints():
    global _EXPL_HINTS
    if _EXPL_HINTS is None:
        if _EXPL_HINTS_PATH.exists():
            with open(_EXPL_HINTS_PATH, encoding="utf-8") as f:
                _EXPL_HINTS = json.load(f)
            app.logger.info("Loaded %d explanation hints from %s", len(_EXPL_HINTS), _EXPL_HINTS_PATH)
        else:
            _EXPL_HINTS = {}
    return _EXPL_HINTS

def _hint_for_arc(dlg_idx, src, tgt):
    """Return the pre-generated scaffold for arc (dlg_idx, src, tgt), or None."""
    if dlg_idx is None:
        return None
    key = f"{dlg_idx}_{min(src,tgt)}_{max(src,tgt)}"
    return _get_expl_hints().get(key)


def _get_claude():
    global _claude_client
    if _claude_client is None:
        import anthropic  # official SDK (skill: claude-api)
        # The Claude Code harness injects an EMPTY ANTHROPIC_AUTH_TOKEN into the
        # environment; the SDK then prefers bearer auth and sends an illegal
        # `Authorization: Bearer ` header (LocalProtocolError -> APIConnectionError).
        # Drop it so the client authenticates with x-api-key instead.
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        _claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _claude_client


def _claude_text_cli(system, user, max_tokens=240):
    """One-shot free-text generation via the Claude CLI using its stored OAuth login
    (no ANTHROPIC_API_KEY, no --bare). Returns the model's text, or None on failure."""
    exe = _find_claude_exe()
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "-p", user, "--system-prompt", system,
             "--model", CLAUDE_MODEL, "--output-format", "json"],
            capture_output=True, text=True, timeout=90,
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return None
        outer = json.loads(proc.stdout)
        return outer.get("result", "") if isinstance(outer, dict) else None
    except Exception:
        return None


def _reason_generate(system, user, max_new_tokens=240):
    """Free-text 'reasoning' generation (the contrastive justification in
    /api/explain/arc). Prefers the Claude CLI (OAuth), then the local LM Studio
    'reasoning' model. Returns text, or '' if no backend is available."""
    txt = _claude_text_cli(system, user, max_tokens=max_new_tokens)
    if txt:
        return txt
    try:
        return _generate(system, user, max_new_tokens=max_new_tokens, role="reasoning") or ""
    except Exception:
        return ""


# ── Parser-error awareness for the explanation engine ────────────────────────
# When the label being explained was proposed by the PARSER (not the human),
# the LLM is warned and given the P(gold | predicted) column of the DDPE
# confusion matrix (build_confusion_index.py) plus real mistaken examples.
_CONF_RT = {"rag": None, "loaded": False}

def _conf_rag():
    """Lazily load the confusion RAG once (used by the parser-caution note).
    None if the index isn't built."""
    if not _CONF_RT["loaded"]:
        _CONF_RT["loaded"] = True
        try:
            from confusion_rag import ConfusionRAG
            _CONF_RT["rag"] = ConfusionRAG()
        except Exception as ex:
            app.logger.info("confusion RAG unavailable: %s", ex)
            _CONF_RT["rag"] = None
    return _CONF_RT.get("rag")


def _parser_confusion_note(pred_rel, edus=None, speakers=None, p=None, t=None):
    """Caution block for a parser-PROPOSED label. Warns the LLM the label is
    machine-made, gives the P(true | predicted) column of the DDPE confusion
    matrix, and — for the top true-relation alternatives — the recorded parser
    error whose DISCOURSE configuration most resembles THIS arc (arc_sim.py +
    dense; cell-constrained)."""
    base = ("Caution: this label was proposed by an automatic parser that makes "
            "labelling errors — weigh the evidence yourself, do not defer to it.")
    rag = _conf_rag()
    if not rag:
        return base
    column = rag.pred_column(pred_rel)   # [(gold, share, cell), ...] desc
    if not column:
        return base
    # true-relation distribution given this prediction (renormalised over column)
    col_tot = sum(share for _, share, _ in column) or 1.0
    dist = ", ".join(f"{g} {share / col_tot:.0%}" for g, share, _ in column[:4])
    lines = [base,
             f"On held-out data, when the parser predicts '{pred_rel}' the true "
             f"relation is instead: {dist}. The most likely confusions here, each "
             f"illustrated by the recorded error most similar in dialogue structure "
             f"to this arc:"]
    qsig = qvec = None
    if edus is not None and p is not None and t is not None:
        from arc_sim import arc_signature
        qsig = arc_signature(speakers[p], edus[p], speakers[t], edus[t], t - p)
        full = [{"speaker": speakers[i], "text": edus[i]} for i in range(t + 1)]
        qvec = rag.embed_query(speakers[p], edus[p], speakers[t], edus[t],
                               full_edus=full, source=p, target=t)
    for g, share, _ in column[:2]:
        hits = rag.retrieve_in_cell(g, pred_rel, qsig, k=1, query_vec=qvec) if qsig else \
               [(0.0, e) for e in (rag.cell(g, pred_rel) or {}).get("examples", [])[:1]]
        if not hits:
            continue
        ex = hits[0][1]
        line = (f"  - {ex['src_speaker']}: \"{ex['src_text']}\" -> "
                f"{ex['tgt_speaker']}: \"{ex['tgt_text']}\" — parser said "
                f"{pred_rel}, gold is {g}")
        why = ex.get("why") or {}
        if why.get("cue"):
            line += f" (cue to tell {g} from {pred_rel}: {why['cue']})"
        lines.append(line)
    return "\n".join(lines)


# ── Shared EGM prompt/parse, used by BOTH the persistent SDK path and the CLI path,
#    so the explanation TEXT (the [i]'…' markers the UI renders into bubbles) is identical
#    whichever transport produced it. ──
def _egm_build(edus, speakers, p, t, chosen_rel=None, prior_arcs=None, hint=None,
               rel_source=None):
    """Build (system_prompt, user_text) for the DDPE-EGM explanation of arc p->t.
    `hint` is an optional pre-generated scaffold from explanation_hints.json.
    `rel_source` says who proposed `chosen_rel`: "user" (human annotator) or
    "parser" (accepted DDPE suggestion) — the LLM gives its own opinion either
    way, but a parser-proposed label additionally triggers the confusion-prior
    caution block so the LLM knows how that prediction tends to be wrong."""
    # Context window: the 15 preceding utterances (absolute indices kept so
    # [p]/[t] still match; the source p is always included even if long-distance).
    start = min(max(0, t - 14), p)
    dial = "\n".join(f"[{i}] {speakers[i]}: {edus[i]}" for i in range(start, t + 1))
    struct = ""
    if prior_arcs:
        lines = []
        for a in prior_arcs:
            ax, ay = int(a.get("x")), int(a.get("y"))
            c, par = max(ax, ay), min(ax, ay)
            if start <= c <= t:
                lines.append(f"[{c}]–>[{par}]" + (f":'{a.get('type')}'" if a.get("type") else ""))
        if lines:
            struct = "\nstructure so far (child–>parent):\n" + "  ".join(lines)
    user = (f"dialogue (last {t - start + 1} utterances):\n{dial}{struct}\n\n"
            f"Candidate arc [{t}]–>[{p}]  "
            f"(parent [{p}] {speakers[p]}: \"{edus[p]}\"  ->  "
            f"child [{t}] {speakers[t]}: \"{edus[t]}\").")
    if chosen_rel:
        rel_def = relation_def(chosen_rel) or f"{chosen_rel}: (no gloss available)"
        user += (
            f"\n\nProposed relation for this arc: '{chosen_rel}'.\n"
            f"Definition — {rel_def}\n\n"
            f"Decide ONLY whether '{chosen_rel}' holds for [{t}]->[{p}], judging it strictly "
            f"against this definition and the actual wording. Keep \"relation\"='{chosen_rel}'. "
            f"If '{chosen_rel}' holds, set \"suggestion_verdict\":\"agree\" and explain in "
            f"\"relation_explanation\" why it fits the definition. "
            f"If '{chosen_rel}' does NOT hold, set \"suggestion_verdict\":\"disagree\" and explain "
            f"in \"relation_explanation\" why it does not fit the definition. "
            f"You are given this one relation and its definition only — do NOT name, propose, "
            f"or compare against any other relation.")
    else:
        user += ("\n\nNo relation is proposed — analyse the LINK only: does [t] attach "
                 "to [p]? Set \"attaches\" and explain in \"link_explanation\" why the "
                 "link holds or does not. Set \"suggestion_verdict\":\"none\".")
    # Inject pre-generated scaffold to ground reasoning in this specific arc.
    # NOT for relation verdicts: the scaffold names likely/alternative relations, which
    # would leak the inventory — the engine must reason from the single given definition.
    if hint and not chosen_rel:
        scaffold = hint.get("scaffold", "")
        key_dist = hint.get("key_distinction", "")
        likely = hint.get("likely_relations", [])
        parts = []
        if key_dist:
            parts.append(f"Pre-analysis of this arc: {key_dist}")
        if likely:
            parts.append(f"Most likely relations for this arc: {', '.join(likely)}.")
        if scaffold:
            parts.append(f"Reasoning scaffold: {scaffold}")
        if parts:
            user += "\n\n[Context hint — use to ground your explanation but do not quote directly]\n" + "\n".join(parts)
    return SYS_CLAUDE_EGM, user


def _egm_contract(obj, p, engine, chosen_rel=None, rel_source=None):
    """Map the model's JSON object to the C1 per-arc contract the frontend expects."""
    rel = str(obj.get("relation", "")).strip()
    if rel and rel not in CANONICAL_RELATIONS_EXT:
        from relations import normalize_relation_ext
        rel = normalize_relation_ext(rel) or rel
    attaches = bool(obj.get("attaches"))
    opinion = None
    if chosen_rel:
        opinion = {"suggested": chosen_rel,
                   "source": rel_source or "user",
                   "verdict": obj.get("suggestion_verdict") or None}
    return {
        "opinion": opinion,
        "parent": p,
        "link": {"explanation": str(obj.get("link_explanation", "")).strip()[:600],
                 "score": 1.0 if attaches else 0.0,
                 "confidence": _conf_from_1to5(obj.get("link_confidence"))},
        "relation": {"candidate": rel,
                     "explanation": str(obj.get("relation_explanation", "")).strip()[:600],
                     "fit": round(float(obj.get("fit", 0)) / 5.0, 3) if obj.get("fit") else None,
                     "confidence": _conf_from_1to5(obj.get("fit"))},
        "engine": engine,
    }


def _claude_explain(edus, speakers, p, t, chosen_rel=None, prior_arcs=None, hint=None,
                    rel_source=None):
    """Explain arc p->t via the PERSISTENT Anthropic SDK client (kept alive across
    requests → no per-call cold start, httpx keep-alive connection pool). Uses the same
    DDPE-EGM prompt as the CLI so the output format is identical. Returns the C1 contract
    or None (no key / SDK / API error) so the caller can fall back to the CLI/LLM."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        client = _get_claude()
    except Exception:
        return None
    sys_prompt, user = _egm_build(edus, speakers, p, t, chosen_rel, prior_arcs, hint=hint,
                                  rel_source=rel_source)
    try:
        # Plain messages.create (same shape as the working virtual-user path);
        # the system prompt already asks for a single JSON object, which we parse
        # from the text (strip code fences) exactly like the CLI path. Avoids the
        # beta `output_config` structured-output route (fails with a connection
        # error on this SDK/endpoint).
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=900,
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s[s.find("{"):]
        s = s[s.find("{"): s.rfind("}") + 1]
        obj = json.loads(s)
    except Exception as ex:
        app.logger.warning("claude SDK explain failed: %s", ex)
        return None
    return _egm_contract(obj, p, "claude-sdk", chosen_rel, rel_source)


def _find_claude_exe():
    """Locate the local Claude CLI ('client') executable."""
    cand = os.environ.get("DSDP_CLAUDE_EXE")
    if cand and os.path.isfile(cand):
        return cand
    import shutil
    for name in ("claude", "claude.exe", "client.exe"):
        w = shutil.which(name)
        if w:
            return w
    for p in (os.path.expanduser(r"~\.local\bin\claude.exe"),
              os.path.expanduser(r"~\.local\bin\claude")):
        if os.path.isfile(p):
            return p
    return None


# DDPE-EGM teacher system prompt (deduced from Liu et al. 2025: the EGM module's
# explanation format — arc notation [t]–>[p], evidence localised with [i]/[j] markers).
# Machine-consumable: emit ONE JSON object only.
# General, DOMAIN-INDEPENDENT labelling pitfalls distilled from validation error
# analysis (build_error_set.py; the non-domain error types). Injected into the
# analyze system prompt so the LLM weighs them when judging a proposed label.
GENERAL_ERROR_CAUSES = """\
Common domain-independent labelling pitfalls to weigh (do not over-apply — check the actual function):
 - Surface interrogative form ("?", question syntax) does NOT by itself make a question relation (QAP/Clarification_Q/Q-Elab): a rhetorical, exclamatory, or backchannel turn that neither asks nor answers is not one.
 - A surface negation word ("no", "not") is not automatically Contrast/Correction: it can be a curt confirmation, a specification, or an acknowledgement.
 - Conditional syntax ("if…", "when X") is not automatically Conditional: the turn may be a Result, an Alternation, or an Acknowledgement.
 - Same-speaker continuity or lexical/topical repetition does not force Continuation: check for Elaboration, Clarification_Q, Parallel, QAP, or Narration.
 - Opinion/stance-sounding wording ("i know", exclamations) is not automatically Comment: it can be Elaboration, Acknowledgement, or Explanation.
 - Minimal acceptance tokens ("thanks", "ok", "np", "yeah") usually mark Acknowledgement, not Comment (a reaction) or QAP (an answer).
 - Mind causal direction: Explanation = Y gives the reason for X; Result = Y is the consequence of X; Background = Y sets context. Do not swap them.
 - Prefer the genuine LOCAL antecedent over a distant, topically unrelated earlier turn: shared function words or off-topic small talk can look like evidence but are not."""

SYS_CLAUDE_EGM = """\
You are the analysis assistant of a human-in-the-loop discourse annotation tool for
multi-party dialogue under SDRT. A HUMAN annotator decides the structure; you only
ANALYSE a candidate arc and explain — you never commit it.

You are given the target arc and the 15 preceding utterances as context.

Notation: an arc is written [t]->[p], meaning utterance t attaches to its parent p
(t is later than p). Ground every claim in the actual wording.

CRITICAL quoting rule — to keep the rendering unambiguous, WHENEVER you quote the words
of an utterance, wrap the verbatim span in ANGLE BRACKETS right after its index:
  [i]<verbatim words from utterance i>
NEVER use apostrophes or quotation marks to quote utterances — apostrophes occur in
possessives and contractions (e.g. "the speaker's", "don't") and would break parsing.
Apostrophes may appear ONLY inside the relation name in the header (e.g. :'Elaboration':).
Example: [3]<no sorry> is a direct negative answer to [0]<do anyone have clay or wheat>.

Your task depends on what is provided:
 - RELATION analysis (ONE proposed relation and its definition are given): decide only
   whether that single relation holds for [t]->[p], judging it strictly against the
   definition you are given and the actual wording. Write "relation_explanation" about
   the PROPOSED relation only, and keep "relation" exactly as proposed.
   If it DOES hold: set "suggestion_verdict":"agree" and explain why it fits the definition.
   If it does NOT hold: set "suggestion_verdict":"disagree" and explain why it does not fit
   the definition. You are shown a single relation and its definition and nothing else — do
   NOT name, propose, or compare against any other relation, and do NOT rely on any inventory.
 - LINK analysis (no relation proposed): judge whether [t] attaches to [p]. Set
   "attaches" and explain in "link_explanation" why the link holds or does not.

Output ONLY one JSON object, no prose around it:
{"attaches":bool,"link_confidence":1-5,"link_explanation":str,"relation":str,
"relation_explanation":str,"fit":1-5,"suggestion_verdict":"agree"|"disagree"|"none"}"""


def _claude_explain_cli(edus, speakers, p, t, chosen_rel=None, prior_arcs=None, hint=None,
                        rel_source=None):
    """Explain arc p->t by shelling out to the local Claude CLI (client.exe) with a
    custom DDPE-EGM system prompt and --output-format json. Uses the CLI's own stored
    OAuth login (do NOT pass --bare -- it forces API-key auth we don't have and makes
    explanations silently fall back to an empty contract). Needs no ANTHROPIC_API_KEY.
    Returns the C1 contract or None on failure.
    If `chosen_rel` is given (relation-explain), Claude EXPLAINS that exact relation and,
    on disagreement, names the relation it considers correct from the inventory.
    `prior_arcs` (the already-annotated structure up to t) is included as context."""
    exe = _find_claude_exe()
    if not exe:
        return None
    sys_prompt, user = _egm_build(edus, speakers, p, t, chosen_rel, prior_arcs, hint=hint,
                                  rel_source=rel_source)
    try:
        proc = subprocess.run(
            [exe, "-p", user, "--system-prompt", sys_prompt,
             "--model", CLAUDE_MODEL, "--output-format", "json"],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            app.logger.warning("claude CLI rc=%s: %s", proc.returncode, (proc.stderr or "")[:300])
            return None
        outer = json.loads(proc.stdout)
        # the CLI reports API/auth failures via is_error while still exiting 0
        # (e.g. "Not logged in · Please run /login"); treat those as a failure so
        # we fall back cleanly instead of parsing the error string as our JSON.
        if isinstance(outer, dict) and outer.get("is_error"):
            app.logger.warning("claude CLI is_error: %s", str(outer.get("result", ""))[:200])
            return None
        result = outer.get("result", "") if isinstance(outer, dict) else ""
        # the result should be our JSON object (strip code fences if any)
        s = result.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s[s.find("{"):]
        s = s[s.find("{"): s.rfind("}") + 1]
        obj = json.loads(s)
    except Exception as ex:
        app.logger.warning("claude CLI explain failed: %s", ex)
        return None
    return _egm_contract(obj, p, "claude-cli", chosen_rel, rel_source)


# ── Pluggable engine registry ────────────────────────────────────────────────
# Each engine wraps the model-specific helpers above behind the uniform Engine
# interface (engines.py). Endpoints resolve an engine by name via get_engine()
# and delegate, so a new engine (a new DISRPT/multimodal model) is added by
# registering a class here -- no /api/suggest/* edits required.

class DdpeEngine(Engine):
    name = "ddpe"
    label = "DDPE student (LLaMA3+LoRA, GPU)"
    can_score_links = True
    can_score_rel = True
    can_score_pair = True
    can_explain = True
    needs_activation = True
    datasets = DDPE_DATASETS

    def _edu_dicts(self, edus, speakers):
        return [{"speaker": speakers[i], "text": edus[i]} for i in range(len(edus))]

    def score_pair(self, edus, speakers, source, target, dataset, target_text=None):
        """Faithful attach score P(target -> source) from the DDPE parser. With
        `target_text` set (a word occluded), the drop in this score is the token's
        saliency. Reads the `source` link from /ddpe/score's ranked candidates."""
        eds = self._edu_dicts(edus, speakers)
        if target_text is not None and 0 <= target < len(eds):
            eds = [dict(e) for e in eds]
            eds[target] = {**eds[target], "text": target_text}
        try:
            r = requests.post(f"{SUGGESTION_ENGINE_URL}/ddpe/score", json={
                "edus": eds, "target": target, "dataset": _ddpe_ds(dataset)}, timeout=120)
            links = r.json().get("links", [])
        except Exception:
            return None
        for l in links:
            if int(l.get("source", -1)) == source:
                return max(0.0, min(1.0, float(l.get("score", 0.0))))
        return 0.0    # DDPE reachable but source not among candidates -> zero attach

    def activate(self, dataset=None):
        ds = (dataset or "stac").lower()
        try:
            r = requests.post(f"{SUGGESTION_ENGINE_URL}/ddpe/load",
                              json={"dataset": _ddpe_ds(ds)}, timeout=600)
            j = r.json()
            return {"steps": [f"ddpe load ({ds}): {j.get('msg')}"],
                    "ready": bool(j.get("ok")), "ddpe_loaded": bool(j.get("ok"))}
        except Exception as ex:
            return {"steps": [f"ddpe load FAILED: {ex} (is suggestion_engine running on "
                              ":8092? start it: python src/suggestion_engine.py)"],
                    "ready": False, "ddpe_loaded": False}

    def score_links(self, edus, speakers, target, dataset):
        r = requests.post(f"{SUGGESTION_ENGINE_URL}/ddpe/score", json={
            "edus": self._edu_dicts(edus, speakers),
            "target": target, "dataset": _ddpe_ds(dataset)}, timeout=180)
        return r.json().get("links", [])

    def score_rel(self, edus, speakers, source, target, dataset):
        r = requests.post(f"{SUGGESTION_ENGINE_URL}/ddpe/score_rel", json={
            "edus": self._edu_dicts(edus, speakers),
            "source": source, "target": target, "dataset": _ddpe_ds(dataset)}, timeout=180)
        return r.json().get("relations", [])

    def explain(self, edus, speakers, source, target, dataset, **ctx):
        return _ddpe_explain(edus, speakers, source, target, dataset)


class DediscoEngine(Engine):
    name = "dedisco"
    label = "DeDisCo parser (:8093)"
    can_score_rel = True

    def score_rel(self, edus, speakers, source, target, dataset):
        r = requests.post(f"{DEDISCO_URL}/dedisco/score_rel", json={
            "edus": [{"speaker": speakers[i], "text": edus[i]} for i in range(len(edus))],
            "source": source, "target": target}, timeout=60)
        return r.json().get("relations", [])


class ClaudeEngine(Engine):
    name = "claude"
    label = f"Claude teacher ({CLAUDE_MODEL})"
    can_explain = True
    needs_activation = True

    def activate(self, dataset=None):
        cli = _find_claude_exe()
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if cli:
            step = f"Claude ({CLAUDE_MODEL}) via local CLI: {os.path.basename(cli)} (OAuth)"
        else:
            step = (f"Claude ({CLAUDE_MODEL}): CLI not found; "
                    f"API key {'present' if has_key else 'MISSING'}")
        ready = bool(cli) or has_key
        return {"steps": [step], "ready": ready, "claude_ready": ready}

    def explain(self, edus, speakers, source, target, dataset, **ctx):
        # Prefer the SDK/API path when a key is present (fast, on the API rate
        # limit); fall back to the local CLI (OAuth subscription, ~15s) only if
        # there is no key or the SDK call fails.
        c = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            c = _claude_explain(edus, speakers, source, target,
                                ctx.get("chosen_rel"),
                                ctx.get("arcs_ctx"), hint=ctx.get("hint"),
                                rel_source=ctx.get("rel_source"))
        if c is None:
            c = _claude_explain_cli(edus, speakers, source, target,
                                    ctx.get("chosen_rel"),
                                    ctx.get("arcs_ctx"), hint=ctx.get("hint"),
                                    rel_source=ctx.get("rel_source"))
        return c


class ZeroshotEngine(Engine):
    name = "zeroshot"
    label = "Local LLM (LM Studio, zero-shot)"
    can_score_pair = True
    can_explain = True

    def score_pair(self, edus, speakers, source, target, dataset, target_text=None):
        return _score_pair(edus, speakers, source, target,
                           target_text=target_text, role="reasoning")

    def explain(self, edus, speakers, source, target, dataset, **ctx):
        return _explain_candidate(edus, speakers, source, target,
                                  rigorous=bool(ctx.get("rigorous")))


register(DdpeEngine(), default=True)
register(ClaudeEngine())
register(DediscoEngine())
register(ZeroshotEngine())


@app.route("/api/engines")
def list_engines():
    """Discovery: every registered engine + its capabilities, so the frontend can
    populate the engine picker dynamically instead of hard-coding the list."""
    return jsonify({"default": REGISTRY.default,
                    "engines": [e.info() for e in REGISTRY.all()]})


@app.route("/api/suggest/incremental", methods=["POST"])
def suggest_incremental():
    """C1: the engine EXPLAINS candidate arcs (link + relation, each with a
    confidence) — it does NOT predict or commit. The human sets
    every link and relation. Optional `parents` (list of candidate ids) restricts
    which arcs to explain; default = the nearest 2 candidates (bounded for cost).
    engine='ddpe' (STAC/Molweni) routes to the local DDPE student, else the LLM."""
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    target = int(payload["target"])
    if target < 1 or target >= len(edus):
        return jsonify({"error": "invalid target"}), 400
    engine = (payload.get("engine") or "zeroshot").lower()
    dataset = (payload.get("dataset") or "stac").lower()
    rigorous = bool(payload.get("rigorous"))   # C6: precompute uses the rigorous G-Eval
    chosen_rel = payload.get("relation")       # proposed relation (relation-explain) → Claude gives its OPINION on it
    # who proposed chosen_rel: "parser" (accepted DDPE suggestion) triggers the
    # confusion-prior caution in the prompt; default "user" (human annotator)
    rel_source = (payload.get("relation_source") or "user") if chosen_rel else None
    dlg_idx = payload.get("dlg_idx")           # dialogue index for hint lookup
    use_hint = bool(payload.get("use_hint"))    # only inject hint when parser scored this arc

    req = payload.get("parents")
    if isinstance(req, list) and req:
        cands = [int(p) for p in req
                 if isinstance(p, (int, float)) and 0 <= int(p) < target]
    else:
        cands = list(range(max(0, target - 2), target))   # default: 2 nearest
    cands = sorted(set(cands))[:6]                          # hard cap (cost)

    arcs_ctx = payload.get("arcs")

    def build(p):
        hint = _hint_for_arc(dlg_idx, p, target) if use_hint else None
        # Try the requested engine (Claude cold-starts the local CLI to avoid SDK
        # billing, then falls back to the SDK; DDPE hits the GPU student). Any engine
        # returning None means "unavailable" -> fall through to the local zero-shot LLM.
        eng = get_engine(engine)
        if eng is not None and eng.can_explain and eng.supports_dataset(dataset):
            c = eng.explain(edus, speakers, p, target, dataset,
                            chosen_rel=chosen_rel, arcs_ctx=arcs_ctx,
                            hint=hint, rel_source=rel_source, rigorous=rigorous)
            if c is not None:
                return c
        return get_engine("zeroshot").explain(edus, speakers, p, target, dataset,
                                              rigorous=rigorous)

    candidates = [build(p) for p in cands]
    return jsonify({"target": target, "candidates": candidates, "engine": engine})


@app.route("/api/suggest/relation", methods=["POST"])
def suggest_relation():
    """Relation prediction SCORES for an arc p→t. Proxies either DDPE (:8092) or
    DeDisCo (:8093) depending on the `engine` field (default: ddpe).
    Output: {relations:[{relation, score 0..1}]} sorted descending."""
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    source = int(payload["source"])
    target = int(payload["target"])
    dataset = (payload.get("dataset") or "stac").lower()
    engine = (payload.get("engine") or "ddpe").lower()

    try:
        eng = get_engine(engine, fallback=True)
        if not (eng is not None and eng.can_score_rel and eng.supports_dataset(dataset)):
            eng = get_engine("ddpe")
        relations = eng.score_rel(edus, speakers, source, target, dataset)
        # HARD structural rules: zero out structurally impossible relations
        ruled = []
        try:
            from relation_rules import filter_relation_scores
            lo, hi = min(source, target), max(source, target)
            relations, ruled = filter_relation_scores(
                relations, edus[lo], edus[hi], speakers[lo], speakers[hi])
        except Exception as ex:
            app.logger.warning("relation rules skipped: %s", ex)
        return jsonify({"source": source, "target": target,
                        "relations": relations, "ruled_out": ruled, "engine": engine})
    except Exception as ex:
        return jsonify({"relations": [], "error": f"{engine} unreachable: {ex}"}), 200


@app.route("/api/suggest/links", methods=["POST"])
def suggest_links():
    """Parser link-suggestion (DDPE, CONSTRAINED to valid parents p<t) — scored, no
    explanation. Proxies suggestion_engine /ddpe/score. Returns candidates [{source, score}]
    with score in 0..1, sorted desc. (The 32b LLM is never loaded; DDPE only.)"""
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    target = int(payload["target"])
    dataset = (payload.get("dataset") or "stac").lower()
    rfc = payload.get("rfc", True)             # apply the Right-Frontier Constraint (default on)
    prior_arcs = payload.get("arcs") or []
    try:
        cands = get_engine("ddpe").score_links(edus, speakers, target, dataset)
        off_rfc = []
        if rfc:
            # SDRT Right-Frontier Constraint: only parents on the right frontier are valid
            # attachment points. Renormalise scores over the on-frontier candidates.
            try:
                from relation_rules import right_frontier
                fr = right_frontier(prior_arcs, target)
                on = [c for c in cands if int(c.get("source")) in fr]
                off_rfc = [int(c.get("source")) for c in cands if int(c.get("source")) not in fr]
                if on:                          # never empty the list (safety)
                    tot = sum(max(0.0, c.get("score", 0.0)) for c in on) or 1.0
                    for c in on:
                        c["score"] = max(0.0, c.get("score", 0.0)) / tot
                    cands = on
            except Exception as ex:
                app.logger.warning("RFC skipped: %s", ex)
        return jsonify({"target": target, "candidates": cands, "engine": "ddpe",
                        "off_right_frontier": off_rfc})
    except Exception as ex:
        return jsonify({"target": target, "candidates": [],
                        "error": f"DDPE unreachable: {ex} (start src/suggestion_engine.py)"}), 200


SYS_LABEL_TMPL = """\
You identify the discourse relation between two utterances in a multi-party \
dialogue. Choose exactly one SDRT relation from this list:

{relations}

Respond ONLY with JSON: {{"label": "<relation>", "confidence": <1-5>, "reason": "<short>"}}"""


@app.route("/api/suggest/label", methods=["POST"])
def suggest_label():
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    i = int(payload["source"])
    j = int(payload["target"])
    if not (0 <= i < len(edus)) or not (0 <= j < len(edus)) or i == j:
        return jsonify({"error": "invalid arc"}), 400

    dial = _dialogue_text(edus, speakers)
    system = SYS_LABEL_TMPL.format(relations=", ".join(RELATIONS))
    user = (f"Dialogue:\n{dial}\n\n"
            f"Arc: ({i}) {speakers[i]}: \"{edus[i]}\"\n"
            f"  -> ({j}) {speakers[j]}: \"{edus[j]}\"\n\n"
            f"Which discourse relation best fits this arc?")

    raw = _generate(system, user, max_new_tokens=256, role="preannotator")
    parsed = _extract_json_loose(raw)
    label = ""
    conf = 0
    reason = ""
    if isinstance(parsed, dict):
        label = str(parsed.get("label", "")).strip()
        try:
            conf = float(parsed.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0
        reason = str(parsed.get("reason", ""))[:200]
        if label and label not in CANONICAL_RELATIONS_EXT:
            from relations import normalize_relation_ext
            label = normalize_relation_ext(label) or ""
    return jsonify({"source": i, "target": j, "label": label,
                    "confidence": conf, "reason": reason, "raw": raw[:400]})


SYS_THREAD = """\
You DISENTANGLE a multi-party dialogue into the parallel conversation threads \
running through it. A thread is a chain of utterances that reply to / follow up \
on one another (the same question being answered, the same sub-topic, the same \
pair or group exchanging turns).

Guidelines:
 - Multi-party chats almost always INTERLEAVE two or more threads — prefer \
   splitting into several threads over lumping everything into one. A single \
   thread covering nearly all utterances is almost always wrong.
 - Put each utterance in the thread of the earlier utterance it responds to.
 - A new question, a new sub-topic, or a turn addressed to a different participant \
   STARTS A NEW thread, even when it is adjacent in time to the previous turn.
 - Independent reactions to the same earlier utterance belong to the SAME thread \
   as that utterance.
 - Return EVERY utterance exactly once across the threads (a partition: no \
   overlap, no omissions). Give each thread a short descriptive label.

Respond ONLY with JSON: {"threads": [[<ids>], [<ids>], ...], "labels": ["<short>", ...]}"""


@app.route("/api/suggest/threads", methods=["POST"])
def suggest_threads():
    payload = request.json or {}
    edus = payload["edus"]
    speakers = payload["speakers"]
    n = len(edus)
    dial = _dialogue_text(edus, speakers)
    user = (f"Dialogue with {n} utterances:\n{dial}\n\n"
            f"Group these {n} utterances into parallel conversation threads. "
            f"Each utterance belongs to exactly one thread.")

    raw = _generate(SYS_THREAD, user, max_new_tokens=512, role="preannotator")
    parsed = _extract_json_loose(raw)
    threads_in = parsed.get("threads", []) if isinstance(parsed, dict) else []
    labels_in = parsed.get("labels", []) if isinstance(parsed, dict) else []
    threads = []
    seen = set()
    for t_idx, t in enumerate(threads_in):
        if not isinstance(t, list):
            continue
        members = []
        for x in t:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= v < n and v not in seen:
                members.append(v)
                seen.add(v)
        if members:
            label = ""
            if t_idx < len(labels_in):
                label = str(labels_in[t_idx])[:60]
            threads.append({"members": sorted(members), "label": label})
    # Orphans go into a catch-all thread
    orphans = [i for i in range(n) if i not in seen]
    if orphans:
        threads.append({"members": orphans, "label": "(unassigned)"})
    return jsonify({"threads": threads, "raw": raw[:400]})


@app.route("/api/discourse/threads", methods=["POST"])
def discourse_threads():
    """Extract maximal DAG paths (discourse threads) from an arc set via BFS."""
    from collections import defaultdict
    payload = request.get_json(force=True) or {}
    arcs    = payload.get("arcs", [])
    n_edus  = int(payload.get("n_edus", 0))
    if n_edus == 0 or not arcs:
        return jsonify({"threads": [], "SD": {}, "roots": []})

    succ = defaultdict(list)
    for a in arcs:
        x, y = int(a["x"]), int(a["y"])
        if 0 <= x < n_edus and 0 <= y < n_edus:
            succ[x].append(y)

    has_incoming = {int(a["y"]) for a in arcs if 0 <= int(a["y"]) < n_edus}
    roots = sorted(set(range(n_edus)) - has_incoming) or [0]

    def _bfs(start):
        threads, SD = [], defaultdict(list)
        frontier = [[start]]
        level = 0
        while frontier:
            nxt = []
            for path in frontier:
                tail = path[-1]
                children = succ.get(tail, [])
                if not children:
                    threads.append({"path": path, "level": level})
                    SD[tail].append(path)
                else:
                    for c in children:
                        nxt.append(path + [c])
            frontier = nxt
            level += 1
        return threads, SD

    all_threads, all_SD = [], defaultdict(list)
    for r in roots:
        t, sd = _bfs(r)
        all_threads.extend(t)
        for node, paths in sd.items():
            all_SD[node].extend(paths)

    return jsonify({
        "threads": all_threads,
        "SD":      {str(k): v for k, v in all_SD.items()},
        "roots":   roots
    })


@app.route("/api/discourse/path_label", methods=["POST"])
def discourse_path_label():
    """Generate a short natural-language label for a maximal discourse path via Claude."""
    payload = request.get_json(force=True) or {}
    path     = payload.get("path", [])
    edus_txt = payload.get("edus", [])
    speakers = payload.get("speakers", [])
    arc_rels = payload.get("arc_rels", {})   # {"x,y": rel_type}

    if not path:
        return jsonify({"label": "?", "summary": "", "source": "empty"})

    if len(path) < 2:
        i = path[0]
        spk = speakers[i] if i < len(speakers) else "?"
        return jsonify({"label": f"{spk}: {(edus_txt[i] if i < len(edus_txt) else '?')[:40]}",
                        "summary": "", "source": "trivial"})

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"label": f"Path {path[0]}→{path[-1]}", "summary": "", "source": "no_key"})

    lines = []
    for k, i in enumerate(path):
        spk = speakers[i] if i < len(speakers) else "?"
        txt = edus_txt[i] if i < len(edus_txt) else "?"
        rel = arc_rels.get(f"{path[k-1]},{i}", "") if k > 0 else ""
        rel_str = f"  [{rel}]" if rel else ""
        lines.append(f"  [{i}]{rel_str} {spk}: {txt}")

    path_block = "\n".join(lines)

    system = (
        "You are a discourse analyst. Given a chain of utterances connected by discourse "
        "relations, produce a SHORT label (3-7 words) and a one-sentence summary of what "
        "this discourse thread is about. Output ONLY valid JSON: "
        '{"label":"...", "summary":"..."}'
    )
    user = f"Discourse path ({len(path)} EDUs):\n{path_block}\n\nLabel this path."

    try:
        client = _get_claude()
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=120,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        obj = json.loads(text)
        return jsonify({"label": obj.get("label", "?"),
                        "summary": obj.get("summary", ""),
                        "source": "claude"})
    except Exception as ex:
        app.logger.warning("path_label failed: %s", ex)
        return jsonify({"label": f"Path {path[0]}→{path[-1]}", "summary": "", "source": "error"})


@app.route("/api/suggest/full", methods=["POST"])
def suggest_full():
    """Run DSDP_h_iterative on the dialogue (merge + H1 + H2 + G-Eval)."""
    payload = request.json or {}
    dataset = payload.get("dataset", "stac")
    edus = payload["edus"]
    speakers = payload["speakers"]

    fake_dialogue = {
        "edus": [{"text": t, "speaker": s} for t, s in zip(edus, speakers)],
        "relations": [],
    }
    system_merge, _ = get_prompts(dataset)
    model, tok = _ensure_model(DEFAULT_MODEL)
    from DSDP_h_iterative import (
        run_h_pass, parse_parents, geval_edges, anonymize,
        SYS_H1, SYS_H2, GEVAL_THRESHOLD,
    )
    from DSDP_llm_dp_torch import build_chat_messages, batch_generate

    n_raw = len(edus)
    if n_raw < 2:
        return jsonify({"arcs": [], "merge_rels": []})

    # Merge step (same as in DSDP_h_iterative)
    groups = find_consecutive_groups(speakers)
    do_merge = len(set(speakers)) > 1 and len(groups) > 0
    merge_rels = {}
    confirmed = []
    if do_merge:
        pairs = [(g[ki], g[ki + 1]) for g in groups for ki in range(len(g) - 1)]
        if pairs:
            lines = [f"  ({i}) {speakers[i]}: {edus[i]}" for i in range(n_raw)]
            pd = [f'  Pair {pi}: ({a}) "{edus[a]}" -> ({b}) "{edus[b]}"  [{speakers[a]}]'
                  for pi, (a, b) in enumerate(pairs)]
            user_m = ("Dialogue:\n" + "\n".join(lines) +
                      "\n\nConsecutive same-speaker pairs:\n" + "\n".join(pd) +
                      "\n\nFor each pair: connected or independent?")
            msgs = build_chat_messages(system_merge, user_m, no_think=False)
            try:
                resp = batch_generate(model, tok, [msgs], batch_size=1,
                                      max_new_tokens=1024)
                raw_m = _extract_json_loose(resp[0])
            except RuntimeError:
                raw_m = {}
            pc = {}
            for pi, (a, b) in enumerate(pairs):
                e = raw_m.get(str(pi), {}) if isinstance(raw_m, dict) else {}
                conn = e.get("connected", False) if isinstance(e, dict) else False
                pc[(a, b)] = conn
                if conn:
                    merge_rels[(a, b)] = "Continuation"
            for g in groups:
                cur = [g[0]]
                for ki in range(len(g) - 1):
                    if pc.get((g[ki], g[ki + 1]), False):
                        cur.append(g[ki + 1])
                    else:
                        if len(cur) >= 2:
                            confirmed.append(cur)
                        cur = [g[ki + 1]]
                if len(cur) >= 2:
                    confirmed.append(cur)
        else:
            do_merge = False

    if do_merge and confirmed:
        merged, o2m = merge_dialogue(edus, speakers, confirmed)
        if len(merged) <= 2:
            do_merge = False
    if not (do_merge and confirmed):
        merged = [(edus[i], speakers[i], [i]) for i in range(n_raw)]
        o2m = {i: i for i in range(n_raw)}

    m = len(merged)
    edus_m = [t[0] for t in merged]
    spks_m = [t[1] for t in merged]
    anon_edus, anon_spks = anonymize(edus_m, spks_m)

    speakers_set = sorted(set(anon_spks))
    speakers_str = (", ".join(speakers_set[:-1]) + " and " + speakers_set[-1]
                    if len(speakers_set) > 1 else speakers_set[0])
    dial_text = "\n".join(
        f"  ({i}) {anon_spks[i]}: \"{anon_edus[i]}\"" for i in range(m))

    user_h1 = (f"Conversation between {speakers_str}:\n{dial_text}\n\n"
               f"For each utterance, who is the speaker responding to?\n\n"
               f"Note: ({0}) and ({1}) are already connected.\n\n"
               f"Respond with parent IDs for each utterance (2 through {m - 1}).")
    resp1, raw1 = run_h_pass(model, tok, SYS_H1, user_h1)
    edges_h1 = parse_parents(raw1, m)
    edges_h1.add((0, 1))
    scores1 = geval_edges(model, tok, edges_h1, merged, user_h1, resp1)

    # Map merged edges back to raw EDU arcs (use first element of each merged unit)
    m2raw_first = [t[2][0] for t in merged]
    arcs = []
    for (i, j), sc in scores1.items():
        if sc >= GEVAL_THRESHOLD:
            arcs.append({
                "x": m2raw_first[i],
                "y": m2raw_first[j],
                "type": "",
                "score": round(sc, 2),
            })
    # Add merge continuations
    for (a, b) in merge_rels:
        arcs.append({"x": a, "y": b, "type": "Continuation", "score": 5.0})

    return jsonify({"arcs": arcs, "merged_count": m})


# ── Guidelines / review-queue / export endpoints ─────────────────────────

_REL_INV_CACHE = {}   # dataset -> (inventory:list, framework:str)


def _dataset_framework(dataset):
    """Discourse framework for a dataset: sdrt/rst/pdtb/dep/... (erst normalised to
    rst) or "" (native). STAC/Molweni/draddp are SDRT."""
    if dataset in ("stac", "stac_full", "molweni", "draddp"):
        return "sdrt"
    cfg = DATASETS.get(dataset, {})
    corpus = cfg.get("corpus", "")           # DISRPT corpora: lang.framework.name
    if corpus.count(".") >= 2:
        fw = corpus.split(".")[1].lower()
        return "rst" if fw == "erst" else fw   # uniform: enhanced-RST -> RST
    return ""


def _dataset_relation_inventory(dataset):
    """The dataset's OWN relation inventory (its original labels), cached.
      - curated SDRT sets for stac/molweni/msdc/draddp_demo and any sdrt.* corpus;
      - otherwise scanned from the corpus's real gold labels (orig_label), most
        frequent first — so RST/PDTB/DEP show THEIR native inventory, not SDRT."""
    if dataset in _REL_INV_CACHE:
        return _REL_INV_CACHE[dataset]
    fw = _dataset_framework(dataset)
    if dataset in DATASET_RELATIONS:
        inv = list(DATASET_RELATIONS[dataset])
    elif fw == "sdrt" or dataset in ("draddp",):
        inv = list(DATASET_RELATIONS["stac"])
    else:
        # scan gold labels from the smallest available split (inventory only)
        cfg = DATASETS.get(dataset, {})
        splits = list(cfg.get("files", {}).keys())
        order = [s for s in ("dev", "test", "train") if s in splits] or splits
        seen = {}
        for sp in order[:1]:
            try:
                for rec in load_records(dataset, sp):
                    for r in rec.get("relations", []):
                        t = (r.get("type") or "").strip()
                        if t:
                            seen[t] = seen.get(t, 0) + 1
            except Exception:
                pass
        inv = [t for t, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
    _REL_INV_CACHE[dataset] = (inv, fw or "native")
    return _REL_INV_CACHE[dataset]


@app.route("/api/relation_defs")
def relation_defs():
    """Per-dataset relation inventory + definitions for inline guidelines / the picker.

    Returns {dataset, framework, inventory:[labels in the dataset's OWN scheme],
    relations:{name: def}, text}. SDRT datasets get curated defs; RST/PDTB/DEP get
    their native labels scanned from gold (defs empty -> frontend shows the raw label)."""
    from relation_glossary import define
    dataset = request.args.get("dataset", "stac")
    inv, fw = _dataset_relation_inventory(dataset)
    # SDRT canonical defs from relations._REL_DEF; other frameworks via the
    # cross-framework glossary (PDTB 3.0 / RST / eRST / SciDTB / ISO), composed.
    relations = {}
    for r in inv:
        d = _REL_DEF.get(r) or define(r, fw)
        if d:
            relations[r] = d
    try:
        text = build_relation_defs(dataset) if fw in ("sdrt", "native") else ""
    except Exception:
        text = ""
    return jsonify({"dataset": dataset, "framework": fw,
                    "inventory": inv, "relations": relations, "text": text})


@app.route("/api/review_queue", methods=["POST"])
def review_queue():
    """Order annotation arcs so a human reviews the UNCERTAIN / RARE first.

    Pure-Python triage (no LLM). Priority combines: low confidence,
    long-distance (|x-y|>1), multi-parent targets, and sibling ambiguity
    (a source with multiple children). Adjacent (|x-y|==1) high-confidence
    edges are split off as `trivial` (auto-acceptable)."""
    payload = request.json or {}
    arcs = payload.get("arcs", []) or []

    # Structural tallies over the arc set.
    parents_of = {}   # y -> set of x  (multi-parent detection)
    children_of = {}  # x -> set of y  (sibling-ambiguity detection)
    norm = []
    for a in arcs:
        try:
            x = int(a["x"])
            y = int(a["y"])
        except (KeyError, TypeError, ValueError):
            continue
        score = a.get("score", None)
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        norm.append({"x": x, "y": y, "type": a.get("type", ""), "score": score})
        parents_of.setdefault(y, set()).add(x)
        children_of.setdefault(x, set()).add(y)

    def _is_trivial(arc):
        adjacent = abs(arc["x"] - arc["y"]) == 1
        sc = arc["score"]
        # High-confidence = score >= 4 on the 1-5 scale, or no score given.
        high_conf = (sc is None) or (sc >= 4)
        single_parent = len(parents_of.get(arc["y"], set())) <= 1
        return adjacent and high_conf and single_parent

    queue = []
    trivial = []
    for arc in norm:
        dist = abs(arc["x"] - arc["y"])
        sc = arc["score"]
        reasons = []
        priority = 0.0
        # (a) low confidence: lower score -> higher priority
        if sc is not None:
            priority += (5.0 - max(1.0, min(5.0, sc))) * 2.0
            if sc <= 3:
                reasons.append(f"low-confidence (score {sc:g})")
        elif dist == 1:
            reasons.append("trivial-adjacent (no score)")
        else:
            # no score and non-adjacent -> mild uncertainty
            priority += 1.0
            reasons.append("no-score")
        # (b) long-distance
        if dist > 1:
            priority += float(dist)
            reasons.append(f"long-distance (|x-y|={dist})")
        # (c) multi-parent target
        n_par = len(parents_of.get(arc["y"], set()))
        if n_par > 1:
            priority += 3.0 * (n_par - 1)
            reasons.append(f"multi-parent (y={arc['y']} has {n_par} parents)")
        # (d) sibling ambiguity: source with multiple children
        n_chi = len(children_of.get(arc["x"], set()))
        if n_chi > 1:
            priority += 1.5 * (n_chi - 1)
            reasons.append(f"sibling-ambiguity (x={arc['x']} has {n_chi} children)")

        if _is_trivial(arc):
            trivial.append({"x": arc["x"], "y": arc["y"]})
            continue
        queue.append({
            "x": arc["x"], "y": arc["y"], "type": arc["type"],
            "score": arc["score"], "priority": round(priority, 3),
            "reasons": reasons,
        })

    queue.sort(key=lambda a: (-a["priority"], abs(a["x"] - a["y"]) * -1))
    return jsonify({
        "queue": queue, "trivial": trivial,
        "n_trivial": len(trivial), "n_review": len(queue),
    })


def _clean_relations(arcs):
    """Normalize arcs to [{x, y, type}] with int endpoints, drop malformed."""
    out = []
    for a in arcs or []:
        try:
            x = int(a["x"])
            y = int(a["y"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"x": x, "y": y, "type": a.get("type", "")})
    return out


@app.route("/api/admin/stats")
def admin_stats():
    """Experimenter-only dashboard: per-participant study metrics."""
    with _db() as c:
        # ── participants ──────────────────────────────────────────────────
        users = {r[0]: {"onboarded": bool(r[1]), "created": r[2]}
                 for r in c.execute("SELECT participant, onboarded, created_at FROM users")}

        # ── latest annotation per (participant, idx) ──────────────────────
        ann_rows = c.execute("""
            SELECT participant, idx, condition, annotation,
                   MAX(ts) as ts
            FROM annotations
            GROUP BY participant, idx
        """).fetchall()

        # ── questionnaires (TLX + SUS) ────────────────────────────────────
        q_rows = c.execute("""
            SELECT participant, condition, data FROM questionnaires
        """).fetchall()

        # ── events: timing + LLM call counts ─────────────────────────────
        ev_rows = c.execute("""
            SELECT participant, condition, action, payload FROM events
        """).fetchall()

    # build per-participant aggregates
    from collections import defaultdict
    stats = defaultdict(lambda: {
        "onboarded": False, "created": None,
        "dialogues": {},          # idx -> {arcs, condition, ts}
        "tlx": [],                # list of weighted TLX scores
        "sus": None,
        "llm_explain": 0, "llm_suggest": 0,
        "dlg_time_ms": defaultdict(int),   # idx -> total ms
    })

    for pid, info in users.items():
        stats[pid]["onboarded"] = info["onboarded"]
        stats[pid]["created"] = info["created"]

    for pid, idx, cond, ann_json, ts in ann_rows:
        try:
            ann = json.loads(ann_json)
            n_arcs = len(ann.get("arcs", []))
        except Exception:
            n_arcs = 0
        stats[pid]["dialogues"][idx] = {"arcs": n_arcs, "condition": cond, "ts": ts}

    for pid, cond, data_json in q_rows:
        try:
            data = json.loads(data_json)
        except Exception:
            continue
        kind = data.get("kind")
        if kind == "tlx":
            # weighted NASA-TLX: each dimension weight * rating / sum(weights)
            dims = data.get("dims", {})
            weights = data.get("weights", {})
            w_sum = sum(weights.values()) or 1
            score = sum(weights.get(k, 0) * v for k, v in dims.items()) / w_sum
            stats[pid]["tlx"].append({"condition": cond, "score": round(score, 1)})
        elif kind == "sus":
            items = data.get("items", [])
            if len(items) == 10:
                raw = sum((v - 1) if i % 2 == 0 else (5 - v)
                          for i, v in enumerate(items))
                stats[pid]["sus"] = round(raw * 2.5, 1)

    for pid, cond, action, payload_json in ev_rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        if action == "claude_explain_done":
            stats[pid]["llm_explain"] += 1
        elif action in ("link_tip_request", "rel_tip_request"):
            stats[pid]["llm_suggest"] += 1
        elif action == "dialogue_done":
            ms = payload.get("time_ms") or payload.get("ms") or 0
            idx = payload.get("idx", -1)
            stats[pid]["dlg_time_ms"][idx] += ms

    # serialise
    out = []
    for pid, s in sorted(stats.items()):
        dlgs = s["dialogues"]
        n_dlgs = len(dlgs)
        n_arcs = sum(d["arcs"] for d in dlgs.values())
        by_cond = {}
        for d in dlgs.values():
            c2 = d.get("condition") or "?"
            by_cond.setdefault(c2, {"dlgs": 0, "arcs": 0})
            by_cond[c2]["dlgs"] += 1
            by_cond[c2]["arcs"] += d["arcs"]
        times = [v for v in s["dlg_time_ms"].values() if v > 0]
        avg_time_s = round(sum(times) / len(times) / 1000, 1) if times else None
        out.append({
            "participant": pid,
            "onboarded": s["onboarded"],
            "n_dialogues": n_dlgs,
            "n_arcs": n_arcs,
            "by_condition": by_cond,
            "tlx": s["tlx"],
            "sus": s["sus"],
            "llm_explain": s["llm_explain"],
            "llm_suggest": s["llm_suggest"],
            "avg_time_s": avg_time_s,
        })
    return jsonify(out)


@app.route("/api/admin/participants")
def admin_participants():
    """Admin: list all annotating participants with dialogue/arc counts."""
    participants = {}
    # Scan annotation files
    for path in ANN_DIR.glob("*_stac_test.json"):
        name = path.stem  # e.g. "USER_007_stac_test"
        participant = name[: name.rfind("_stac_test")]
        if not participant:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            data = {}
        n_dlgs = len(data)
        n_arcs = sum(len((v or {}).get("arcs", [])) for v in data.values())
        participants[participant] = {
            "participant": participant,
            "n_dialogues": n_dlgs,
            "n_arcs": n_arcs,
            "datasets": [{"dataset": "stac", "split": "test",
                          "n_dlgs": n_dlgs, "n_arcs": n_arcs}],
        }
    # Add participants found in the DB that have no annotation files
    try:
        with _db() as c:
            for row in c.execute("SELECT participant FROM users"):
                pid = row["participant"]
                if pid not in participants:
                    participants[pid] = {
                        "participant": pid,
                        "n_dialogues": 0,
                        "n_arcs": 0,
                        "datasets": [],
                    }
    except Exception:
        pass
    return jsonify(sorted(participants.values(), key=lambda p: p["participant"]))


@app.route("/api/admin/compare")
def admin_compare():
    """Admin: load user + gold arcs for a dialogue and compute P/R/F1."""
    participant = request.args.get("participant", "")
    dataset = request.args.get("dataset", "stac")
    split = request.args.get("split", "test")
    idx = int(request.args.get("idx", 0))

    # Load the dialogue (EDUs + gold arcs)
    data = _load_dataset(dataset, split)
    d = data[idx]
    edus = [e["text"] for e in d["edus"]]
    speakers = [e["speaker"] for e in d["edus"]]
    gold_arcs_raw = d.get("relations", [])
    gold_arcs = [{"x": int(r["x"]), "y": int(r["y"]),
                  "type": normalize_relation(r.get("type", ""))}
                 for r in gold_arcs_raw]

    # Load user annotation
    user_anns = _load_anns(dataset, split, participant)
    user_ann = user_anns.get(str(idx)) or {"arcs": [], "threads": [], "notes": ""}
    user_arcs = [{"x": int(a["x"]), "y": int(a["y"]),
                  "type": normalize_relation(a.get("type", ""))}
                 for a in (user_ann.get("arcs") or [])]

    # Compute P/R/F1 on arcs (existence only)
    pred_links = {(a["x"], a["y"]) for a in user_arcs}
    gold_links = {(a["x"], a["y"]) for a in gold_arcs}
    arc_stats = _prf(pred_links, gold_links)

    # Compute P/R/F1 on relations (x, y, normalized_type)
    pred_rels = {(a["x"], a["y"], normalize_relation(a["type"])) for a in user_arcs}
    gold_rels = {(a["x"], a["y"], normalize_relation(a["type"])) for a in gold_arcs}
    rel_stats = _prf(pred_rels, gold_rels)

    return jsonify({
        "edus": edus,
        "speakers": speakers,
        "user_arcs": user_arcs,
        "gold_arcs": gold_arcs,
        "stats": {
            "arc_p": arc_stats["precision"],
            "arc_r": arc_stats["recall"],
            "arc_f1": arc_stats["f1"],
            "rel_p": rel_stats["precision"],
            "rel_r": rel_stats["recall"],
            "rel_f1": rel_stats["f1"],
        },
    })


@app.route("/api/export", methods=["POST"])
def export_annotation():
    """Serialize the current annotation for download (no file written).

    format ∈ {json, stac, glozz}. Returns {format, filename, content} where
    content is a string the frontend can offer as a download."""
    payload = request.json or {}
    dataset = payload.get("dataset", "stac")
    split = payload.get("split", "test")
    idx = int(payload.get("idx", 0))
    fmt = (payload.get("format") or "json").lower()
    arcs = payload.get("arcs", [])

    d = _load_dialogue(dataset, split, idx)
    edus = [{"speaker": e["speaker"], "text": e["text"]} for e in d["edus"]]
    dia_id = d.get("id", f"{dataset}_{split}_{idx}")
    relations = _clean_relations(arcs)
    base = f"{dataset}_{split}_{idx}"

    if fmt == "json":
        content = json.dumps(
            {"id": dia_id, "edus": edus, "relations": relations},
            ensure_ascii=False, indent=1)
        return jsonify({"format": "json", "filename": f"{base}.json",
                        "content": content})

    if fmt == "stac":
        # STAC-DPA shape (matches data/stac/*_dpa.json): per-EDU speechturn,
        # relations as {type, x, y}.
        stac_edus = []
        for k, e in enumerate(d["edus"]):
            stac_edus.append({
                "speaker": e["speaker"],
                "text": e["text"],
                "speechturn": e.get("speechturn", k),
            })
        stac_rels = [{"type": r["type"], "x": r["x"], "y": r["y"]}
                     for r in relations]
        content = json.dumps(
            {"id": dia_id, "edus": stac_edus, "relations": stac_rels},
            ensure_ascii=False, indent=1)
        return jsonify({"format": "stac", "filename": f"{base}_dpa.json",
                        "content": content})

    if fmt == "glozz":
        # Minimal, documented Glozz-style .aa stub. The full Glozz schema
        # (units with feature-sets, anchored positions, schema relations) is
        # out of scope here; this emits a partial XML with one relation
        # element per arc so downstream tooling has a starting point.
        units = "".join(
            f'  <unit id="{dia_id}_{k}"><characterisation>'
            f'<type>EDU</type></characterisation>'
            f'<positioning/></unit>\n'
            for k in range(len(edus)))
        rels = "".join(
            f'  <relation id="{dia_id}_r{n}">'
            f'<characterisation><type>{r["type"] or "UNLABELLED"}</type>'
            f'</characterisation>'
            f'<positioning><term id="{dia_id}_{r["x"]}"/>'
            f'<term id="{dia_id}_{r["y"]}"/></positioning></relation>\n'
            for n, r in enumerate(relations))
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!-- PARTIAL Glozz .aa export: units + relations only. '
            'Not a complete Glozz annotation (no anchored text positions / '
            'feature-sets). -->\n'
            f'<annotations id="{dia_id}">\n{units}{rels}</annotations>\n')
        return jsonify({
            "format": "glozz", "filename": f"{base}.aa", "content": xml,
            "note": "Partial Glozz export (units + relations only); "
                    "no anchored positions or feature-sets.",
        })

    return jsonify({"error": f"unknown format: {fmt}"}), 400


# ── P1 study instrumentation (S5 logging, S6 metrics/IAA, S7 questionnaires) ──
LOG_DIR = ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _safe_pid(pid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(pid or "anon"))[:64] or "anon"


@app.route("/api/log", methods=["GET"])
def study_log_get():
    """S5 — return one participant's full action log (JSONL) for download."""
    pid = _safe_pid(request.args.get("participant"))
    path = LOG_DIR / f"{pid}.jsonl"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return app.response_class(text, mimetype="application/x-ndjson")


@app.route("/api/log", methods=["POST"])
def study_log():
    """S5 — append ONE action event (or a batch) as JSONL under data/logs/."""
    payload = request.json or {}
    events = payload.get("events")
    if events is None:
        events = [payload]
    pid = _session_pid(payload.get("participant") or (events[0] if events else {}).get("participant"))
    path = LOG_DIR / f"{pid}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    with _db_lock, _db() as c:        # mirror into SQLite
        c.executemany(
            "INSERT INTO events(participant, ts, step, condition, action, payload) "
            "VALUES(?,?,?,?,?,?)",
            [(pid, ev.get("ts"), ev.get("step"), ev.get("condition"),
              ev.get("action"), json.dumps(ev.get("payload") or {}, ensure_ascii=False))
             for ev in events])
    return jsonify({"status": "ok", "logged": len(events), "file": path.name})


@app.route("/api/questionnaire", methods=["POST"])
def study_questionnaire():
    """S7 — store a NASA-TLX / SUS submission with participant + condition."""
    payload = request.json or {}
    pid = _session_pid(payload.get("participant"))
    rec = {"ts": int(time.time() * 1000),
           "participant": pid, "kind": payload.get("kind"), "condition": payload.get("condition"),
           "dataset": payload.get("dataset"), "idx": payload.get("idx"),
           "nasa_tlx": payload.get("nasa_tlx"),
           "nasa_tlx_pairs": payload.get("nasa_tlx_pairs"),
           "nasa_tlx_weights": payload.get("nasa_tlx_weights"),
           "nasa_tlx_weighted": payload.get("nasa_tlx_weighted"),
           "sus": payload.get("sus"), "sus_score": payload.get("sus_score"),
           "feedback": payload.get("feedback"), "llm": payload.get("llm"),
           "duration_ms": payload.get("duration_ms")}
    with open(LOG_DIR / f"{pid}_questionnaire.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with _db_lock, _db() as c:        # mirror into SQLite
        c.execute("INSERT INTO questionnaires(participant, ts, condition, dataset, idx, data) "
                  "VALUES(?,?,?,?,?,?)",
                  (pid, rec["ts"], rec["condition"], rec["dataset"],
                   payload.get("idx"), json.dumps(rec, ensure_ascii=False)))
    return jsonify({"status": "ok"})


def _prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "n_pred": len(pred), "n_gold": len(gold)}


def _arc_metrics(pred_arcs, gold_arcs):
    """S6 — link & link+relation F1 vs gold, exact AND partial (Jaccard-style)."""
    pl = {(int(a["x"]), int(a["y"])) for a in pred_arcs}
    gl = {(int(a["x"]), int(a["y"])) for a in gold_arcs}
    pr = {(int(a["x"]), int(a["y"]), str(a.get("type", "")).strip()) for a in pred_arcs}
    gr = {(int(a["x"]), int(a["y"]), str(a.get("type", "")).strip()) for a in gold_arcs}
    link = _prf(pl, gl)
    labeled = _prf(pr, gr)
    link_jac = (len(pl & gl) / len(pl | gl)) if (pl or gl) else 1.0
    # partial labeled: an arc with the right (x,y) scores 1 if the relation also
    # matches, else 0.5 (link found, relation wrong) — the partial/Jaccard credit.
    gtype = {(x, y): t for (x, y, t) in gr}
    partial = sum(1.0 if gtype.get((x, y)) == t else 0.5
                  for (x, y, t) in pr if (x, y) in gl)
    denom = len(pl | gl)
    return {"link": link, "labeled_exact": labeled,
            "link_jaccard": round(link_jac, 4),
            "labeled_partial": round(partial / denom, 4) if denom else 1.0}


def _kappa(arcs_a, arcs_b):
    """Cohen's κ over the union of candidate (x,y) items; category = relation or
    NONE. Used for human–human and human–LLM agreement (S6)."""
    la = {(int(a["x"]), int(a["y"])): (str(a.get("type", "")).strip() or "UNLABELLED") for a in arcs_a}
    lb = {(int(a["x"]), int(a["y"])): (str(a.get("type", "")).strip() or "UNLABELLED") for a in arcs_b}
    items = set(la) | set(lb)
    if not items:
        return None
    A = [la.get(it, "NONE") for it in items]
    B = [lb.get(it, "NONE") for it in items]
    n = len(items)
    po = sum(1 for x, y in zip(A, B) if x == y) / n
    cats = set(A) | set(B)
    pe = sum((A.count(c) / n) * (B.count(c) / n) for c in cats)
    k = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    return {"kappa": round(k, 4), "po": round(po, 4), "pe": round(pe, 4), "n_items": n}


@app.route("/api/metrics", methods=["POST"])
def study_metrics():
    """S6 — accuracy-vs-gold + IAA bundle for the current session."""
    payload = request.json or {}
    dataset = payload.get("dataset", "stac")
    split = payload.get("split", "test")
    idx = int(payload.get("idx", 0))
    arcs = payload.get("arcs", [])
    d = _load_dialogue(dataset, split, idx)
    gold = [{"x": int(r["x"]), "y": int(r["y"]), "type": r.get("type", "")}
            for r in d.get("relations", [])]
    out = {"dataset": dataset, "split": split, "idx": idx,
           "accuracy_vs_gold": _arc_metrics(arcs, gold)}
    if payload.get("arcs_b"):           # human–human IAA
        out["kappa_human_human"] = _kappa(arcs, payload["arcs_b"])
    if payload.get("arcs_llm"):         # human–LLM IAA
        out["kappa_human_llm"] = _kappa(arcs, payload["arcs_llm"])
    out["kappa_human_gold"] = _kappa(arcs, gold)
    return jsonify(out)


def _auto_load_default_model():
    """Ensure the default LM Studio model is loaded when the server starts.
    Runs in a background thread so Flask startup isn't blocked."""
    try:
        if not _lms_server_up():
            print(f"  [auto-load] LM Studio server not up — skip.")
            return
        loaded = _lms_ps()
        if any(_lms_model_id(m) == DEFAULT_LMS_MODEL or m.get("modelKey") == DEFAULT_LMS_MODEL
               for m in loaded):
            print(f"  [auto-load] {DEFAULT_LMS_MODEL} already loaded.")
            return
        print(f"  [auto-load] loading {DEFAULT_LMS_MODEL} on GPU {GPU_INDEX}...")
        code, out, err = _run_lms(["load", DEFAULT_LMS_MODEL, "--yes",
                                   "--gpu", f"{float(GPU_INDEX):.1f}"], timeout=300)
        if code == 0:
            print(f"  [auto-load] ready.")
        else:
            print(f"  [auto-load] FAILED: {err.strip() or out.strip()}")
    except Exception as e:
        print(f"  [auto-load] error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-auto-load", action="store_true",
                        help="do not auto-load the default LM Studio model")
    args = parser.parse_args()
    print(f"DSDP annotation tool -> http://{args.host}:{args.port}")
    # Loud check: the explanation engine silently degrading (empty contract) is what
    # keeps biting after restarts. Warn if NO Claude backend is reachable at all.
    if not (_find_claude_exe() or os.environ.get("ANTHROPIC_API_KEY")):
        print("[WARN] No Claude backend: the CLI (client/claude.exe) was not found and "
              "ANTHROPIC_API_KEY is unset. Explanations will be unavailable. Log in with "
              "the Claude CLI, or set ANTHROPIC_API_KEY / add a .env.", flush=True)
    if not args.no_auto_load:
        threading.Thread(target=_auto_load_default_model, daemon=True).start()
    # Warm the persistent Anthropic client at startup (import + client construct) so the
    # FIRST explanation isn't slowed by the one-time cold import.
    if os.environ.get("ANTHROPIC_API_KEY"):
        def _warm():
            try: _get_claude()
            except Exception: pass
        threading.Thread(target=_warm, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
