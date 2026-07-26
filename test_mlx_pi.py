#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich>=13", "httpx>=0.27"]
# ///
"""Lightweight tests for mlx-pi.

Run:  ./test_mlx_pi.py        (or:  uv run test_mlx_pi.py)

No pytest — a tiny assert-based runner. It loads the `mlx-pi` script as a
module and exercises the fiddly HF-cache and pi-config logic against throwaway
temp directories (by repointing the module's HF_CACHE / PI_CONFIG_* globals).
These cover the bugs we've actually hit: stale *.incomplete false-positives,
size formatting, and multi-model pi registration.
"""
import argparse
import importlib.util
import io
import json
import os
import plistlib
import shutil
import stat
import sys
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_module():
    loader = SourceFileLoader("mlxpi", str(HERE / "mlx-pi"))
    spec = importlib.util.spec_from_loader("mlxpi", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


m = _load_module()
# Silence the module's rich output (its ok()/info() helpers) during tests.
m.console = m.Console(file=io.StringIO())

# --- tiny runner -----------------------------------------------------------
TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn

# --- fixtures (plain functions; reassign module globals to temp dirs) -------
def fresh_cache():
    tmp = Path(tempfile.mkdtemp())
    m.HF_CACHE = tmp
    return tmp

def fresh_pi():
    tmp = Path(tempfile.mkdtemp())
    m.PI_CONFIG_DIR = tmp
    m.PI_CONFIG_FILE = tmp / "models.json"
    m.PI_SETTINGS_FILE = tmp / "settings.json"  # else tests clobber real settings
    return tmp

def fresh_state():
    tmp = Path(tempfile.mkdtemp())
    m.STATE_DIR = tmp
    m.STATE_FILE = tmp / "server.json"
    m.LEGACY_PID_FILE = tmp / "server.pid"
    m.CONTROL_LOCK_FILE = tmp / "control.lock"
    m.LOG_FILE = tmp / "server.log"
    return tmp

def make_model(cache, repo, *, complete=True, stale_incomplete=False,
               active_incomplete=False, blob_size=1024):
    """Create a fake HF cache dir for `repo`. A *.incomplete temp is 'stale' if
    its final blob exists, 'active' if it doesn't."""
    d = cache / f"models--{repo.replace('/', '--')}"
    blobs = d / "blobs"
    blobs.mkdir(parents=True)
    if complete:
        snaps = d / "snapshots" / "rev0"
        snaps.mkdir(parents=True)
        blob = blobs / ("a" * 64)
        with open(blob, "wb") as f:
            f.truncate(blob_size)  # sparse: large fake models consume no real RAM/disk
        (snaps / "model.safetensors").symlink_to(blob)
    if stale_incomplete:  # final blob ('a'*64) present -> stale
        (blobs / ("a" * 64 + ".deadbeef.incomplete")).write_bytes(b"y" * 256)
    if active_incomplete:  # final blob ('b'*64) absent -> active
        (blobs / ("b" * 64 + ".cafef00d.incomplete")).write_bytes(b"y" * 256)
    return repo


# --- pure helpers ----------------------------------------------------------
@test
def size_formats():
    assert m._size(0) == "—"
    assert m._size(None) == "—"
    assert m._size(512) == "512 B"
    assert m._size(1536) == "1.5 KB"
    assert m._size(2 * 1024 ** 3) == "2.0 GB"

@test
def vision_model_detection():
    os.environ.pop("MLX_VISION_MODELS", None); os.environ.pop("MLX_TEXT_MODELS", None)
    assert m.is_vision_model(m.GEMMA_MODEL) is True
    assert m.is_vision_model("mlx-community/Qwen2-VL-7B-4bit") is True
    assert m.is_vision_model("mlx-community/llava-1.5-7b-4bit") is True
    assert m.is_vision_model(m.DEFAULT_MODEL) is False
    assert m.is_vision_model(m.QWEN_CODER_MODEL) is False

@test
def vision_detection_env_overrides():
    try:
        os.environ["MLX_VISION_MODELS"] = "org/secretly-vision, org/another"
        assert m.is_vision_model("org/secretly-vision") is True
        os.environ["MLX_TEXT_MODELS"] = m.GEMMA_MODEL   # force a known VLM to text
        assert m.is_vision_model(m.GEMMA_MODEL) is False
    finally:
        os.environ.pop("MLX_VISION_MODELS", None); os.environ.pop("MLX_TEXT_MODELS", None)

@test
def backend_selection():
    os.environ.pop("MLX_VISION_MODELS", None); os.environ.pop("MLX_TEXT_MODELS", None)
    assert m.backend_name(m.GEMMA_MODEL) == m.VLM_SERVER
    assert m.backend_name(m.DEFAULT_MODEL) == m.LM_SERVER
    # server_bin() resolves to the right binary basename per model
    assert m.server_bin(m.GEMMA_MODEL).endswith("mlx_vlm.server")
    assert m.server_bin(m.DEFAULT_MODEL).endswith("mlx_lm.server")
    assert m.server_bin().endswith("mlx_lm.server")   # no model → text default
    assert m.generate_bin(m.GEMMA_MODEL).endswith("mlx_vlm.generate")
    assert m.generate_bin(m.DEFAULT_MODEL).endswith("mlx_lm.generate")

@test
def backend_kind_matches_server_backend_vocab():
    # Regression: cmd_up's "is the running backend wrong?" check compares
    # server_backend()'s short code ('lm'/'vlm') against a model's expected
    # backend. backend_kind() is that short code; backend_name() is the full
    # binary ('mlx_lm.server') and must NOT be used for the comparison — they
    # never match, which falsely flagged a correct backend and force-restarted.
    assert m.backend_kind(m.DEFAULT_MODEL) == "lm"
    assert m.backend_kind(m.GEMMA_MODEL) == "vlm"
    assert m.backend_name(m.DEFAULT_MODEL) != m.backend_kind(m.DEFAULT_MODEL)
    # The exact predicate cmd_up uses must be False when the right backend runs.
    for model in (m.DEFAULT_MODEL, m.GEMMA_MODEL):
        running = m.backend_kind(model)   # what server_backend(pid) returns
        assert (running not in (None, m.backend_kind(model))) is False, model

@test
def resolve_model_precedence():
    os.environ.pop("MLX_MODEL", None)
    ns = argparse.Namespace(qwen_coder=False, gemma=False, model=None)
    assert m.resolve_model(ns) == m.DEFAULT_MODEL
    ns.qwen_coder = True
    assert m.resolve_model(ns) == m.QWEN_CODER_MODEL
    ns.qwen_coder = False
    ns.model = "org/custom"
    assert m.resolve_model(ns) == "org/custom"

@test
def positional_repo_wins():
    ns = argparse.Namespace(qwen_coder=True, gemma=False, model=None, repo="org/explicit")
    assert m.resolve_target_model(ns) == "org/explicit"

@test
def hf_cache_resolution():
    for k in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        os.environ.pop(k, None)
    assert m._hf_cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"
    os.environ["HF_HOME"] = "/custom/home"
    assert m._hf_cache_dir() == Path("/custom/home/hub")
    os.environ["HF_HUB_CACHE"] = "/custom/hub"
    assert m._hf_cache_dir() == Path("/custom/hub")
    os.environ.pop("HF_HUB_CACHE"); os.environ.pop("HF_HOME")


# --- model_status / cache ---------------------------------------------------
@test
def status_absent():
    fresh_cache()
    assert m.model_status("org/nope") == "absent"

@test
def status_complete():
    c = fresh_cache()
    make_model(c, "org/done")
    assert m.model_status("org/done") == "complete"

@test
def status_active_partial():
    c = fresh_cache()
    make_model(c, "org/dl", complete=False, active_incomplete=True)
    assert m.model_status("org/dl") == "partial"

@test
def stale_incomplete_is_not_partial():
    # the regression: stale .incomplete must NOT mark a complete model partial
    c = fresh_cache()
    make_model(c, "org/done", stale_incomplete=True)
    assert m.model_status("org/done") == "complete"

@test
def prune_removes_stale_keeps_active():
    c = fresh_cache()
    make_model(c, "org/x", stale_incomplete=True, active_incomplete=True)
    assert m.prune_stale_incomplete("org/x") == 256  # only the stale temp's bytes
    names = {p.name for p in (c / "models--org--x" / "blobs").iterdir()}
    assert ("a" * 64 + ".deadbeef.incomplete") not in names  # stale removed
    assert ("b" * 64 + ".cafef00d.incomplete") in names      # active kept

@test
def cache_size_excludes_incomplete():
    c = fresh_cache()
    make_model(c, "org/x", blob_size=1000, stale_incomplete=True)
    assert m.cache_size_bytes("org/x") == 1000  # 256-byte temp not counted

@test
def present_lists_models():
    c = fresh_cache()
    make_model(c, "org/a")
    make_model(c, "org/b")
    assert set(m._present_repo_ids()) == {"org/a", "org/b"}


# --- pi config --------------------------------------------------------------
@test
def configure_pi_registers_all_downloaded():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a")
    make_model(c, "org/b")
    m.configure_pi("org/a", "http://localhost:8080/v1")
    ids = m.configured_model_ids()
    assert ids[0] == "org/a"               # chosen model is the default
    assert set(ids) == {"org/a", "org/b"}  # every downloaded model registered

@test
def refresh_adds_new_keeps_default():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a")
    m.configure_pi("org/a", "http://localhost:8080/v1")
    make_model(c, "org/b")                 # newly downloaded later
    assert m.refresh_pi_models() == 2
    ids = m.configured_model_ids()
    assert ids[0] == "org/a"               # default preserved
    assert set(ids) == {"org/a", "org/b"}

@test
def refresh_none_when_unconfigured():
    fresh_cache(); fresh_pi()              # no config written
    assert m.refresh_pi_models() is None

@test
def configure_pi_sets_startup_default_in_settings():
    # The bug: pi reads its startup model from settings.json (defaultModel),
    # NOT the order of models.json — so configure_pi must write settings too.
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a"); make_model(c, "org/b")
    m.configure_pi("org/b", "http://localhost:8080/v1")
    s = m.pi_settings()
    assert s["defaultProvider"] == m.PI_PROVIDER_ID
    assert s["defaultModel"] == "org/b"
    assert m.configured_model_id() == "org/b"   # reflects settings, not list order

@test
def settings_default_wins_over_list_order():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a"); make_model(c, "org/b")
    m.configure_pi("org/a", "http://localhost:8080/v1")  # org/a first in list
    m.set_pi_default("org/b")                            # but pi opens on org/b
    assert m.configured_model_id() == "org/b"

@test
def force_primary_moves_default_plain_refresh_does_not():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a")
    m.configure_pi("org/a", "http://localhost:8080/v1")  # default = org/a
    make_model(c, "org/b")
    m.refresh_pi_models()                                 # plain sync: keep default
    assert m.configured_model_id() == "org/a"
    m.refresh_pi_models(primary="org/b", force_primary=True)  # explicit switch
    assert m.configured_model_id() == "org/b"

@test
def align_pi_default_configures_when_unset():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a"); make_model(c, "org/b")
    # pi not configured yet → align should create the provider + set the default
    n = m.align_pi_default("org/b", 8080)
    assert n == 2
    assert m.configured_model_id() == "org/b"

@test
def align_pi_default_switches_existing():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a"); make_model(c, "org/b")
    m.configure_pi("org/a", "http://localhost:8080/v1")   # default = org/a
    m.align_pi_default("org/b", 8080)
    assert m.configured_model_id() == "org/b"             # moved to org/b

@test
def set_pi_default_preserves_other_settings():
    fresh_pi()
    m.PI_SETTINGS_FILE.write_text('{"defaultThinkingLevel": "high", "packages": ["x"]}')
    m.set_pi_default("org/keep")
    s = m.pi_settings()
    assert s["defaultThinkingLevel"] == "high"   # untouched
    assert s["packages"] == ["x"]                # untouched
    assert s["defaultModel"] == "org/keep"

@test
def multimodal_input_in_entry():
    assert m._pi_model_entry(m.GEMMA_MODEL)["input"] == ["text", "image"]
    assert m._pi_model_entry(m.DEFAULT_MODEL)["input"] == ["text"]

@test
def pi_maxtokens_matches_server_default():
    # pi must not advertise a larger maxTokens than the server's --max-tokens,
    # or responses truncate silently. They share DEFAULT_MAX_TOKENS. The big
    # presets have a 256K context, so no clamp applies — they get the full value.
    assert m._pi_model_entry(m.DEFAULT_MODEL)["maxTokens"] == m.DEFAULT_MAX_TOKENS

@test
def maxtokens_clamped_under_small_context():
    # A small-context model must keep maxTokens well under its window so the
    # prompt still fits — never claim the whole context for output.
    saved, m.hf_context_window = m.hf_context_window, lambda repo: 8192
    try:
        e = m._pi_model_entry("org/small-ctx")
        assert e["contextWindow"] == 8192
        assert e["maxTokens"] == 4096          # min(DEFAULT_MAX_TOKENS, 8192 // 2)
    finally:
        m.hf_context_window = saved


# --- sync policies: pin mode + RAM guard ------------------------------------
@test
def pin_enabled_reads_env():
    for v in ("1", "true", "YES", "On"):
        os.environ["MLX_PIN_MODEL"] = v
        assert m._pin_enabled() is True
    os.environ["MLX_PIN_MODEL"] = "0"
    assert m._pin_enabled() is False
    os.environ.pop("MLX_PIN_MODEL", None)
    assert m._pin_enabled() is False

@test
def pinned_mode_registers_only_primary():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a"); make_model(c, "org/b")
    os.environ["MLX_PIN_MODEL"] = "1"
    try:
        m.configure_pi("org/a", "http://localhost:8080/v1")
        assert m.configured_model_ids() == ["org/a"]  # only the chosen model
    finally:
        os.environ.pop("MLX_PIN_MODEL", None)

@test
def ram_guard_hides_oversized_but_keeps_primary():
    c = fresh_cache(); fresh_pi()
    # 'small' fits, 'big' doesn't — pretend the box has 1 GB of RAM.
    make_model(c, "org/small", blob_size=100 * 1024 ** 2)        # ~0.1 GB
    make_model(c, "org/big", blob_size=4 * 1024 ** 3)            # ~4 GB -> ~5 GB to load
    saved_ram, m.ram_gb = m.ram_gb, lambda: 1
    os.environ.pop("MLX_NO_RAM_GUARD", None)
    try:
        # primary is the small one: big is hidden.
        assert m._pi_model_ids("org/small", force_primary=True) == ["org/small"]
        assert m._ram_dropped(["org/small"]) == ["org/big"]
        # an explicitly-chosen big primary is always kept (you asked for it).
        assert "org/big" in m._pi_model_ids("org/big", force_primary=True)
        # escape hatch: guard off registers everything.
        os.environ["MLX_NO_RAM_GUARD"] = "1"
        assert set(m._pi_model_ids("org/small", force_primary=True)) == {"org/small", "org/big"}
    finally:
        m.ram_gb = saved_ram
        os.environ.pop("MLX_NO_RAM_GUARD", None)

@test
def fits_ram_unknown_never_blocks():
    saved_ram, m.ram_gb = m.ram_gb, lambda: 0   # unknown RAM
    try:
        assert m._fits_ram("org/whatever") is True
    finally:
        m.ram_gb = saved_ram


# --- correctness hardening -------------------------------------------------
@test
def metadata_and_context_are_conservative():
    assert m._ctx_from_config({"max_position_embeddings": True}) is None
    assert m._ctx_from_config({"max_position_embeddings": 512}) is None
    assert m._ctx_from_config({"max_position_embeddings": 32768}) == 32768
    assert m._vision_from_config({"vision_config": {"hidden_size": 1}}) is True
    assert m._vision_from_config({"architectures": ["TextForCausalLM"]}) is False
    saved, m.hf_context_window = m.hf_context_window, lambda repo: None
    try:
        entry = m._pi_model_entry("org/unknown")
        assert entry["contextWindow"] == 32768
        assert entry["maxTokens"] == 16384
    finally:
        m.hf_context_window = saved

@test
def pi_models_are_filtered_to_primary_backend():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/text")
    make_model(c, "org/Qwen2-VL")
    m.configure_pi("org/text", "http://localhost:8080/v1")
    assert m.configured_model_ids() == ["org/text"]
    m.configure_pi("org/Qwen2-VL", "http://localhost:8080/v1")
    assert m.configured_model_ids() == ["org/Qwen2-VL"]

@test
def deleting_default_promotes_compatible_replacement():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a"); make_model(c, "org/b")
    m.configure_pi("org/a", "http://localhost:8080/v1")
    shutil.rmtree(c / "models--org--a")
    assert m.refresh_pi_models() == 1
    assert m.configured_model_ids() == ["org/b"]
    assert m.configured_model_id() == "org/b"

@test
def deleting_only_default_clears_owned_settings():
    c = fresh_cache(); fresh_pi()
    make_model(c, "org/a")
    m.configure_pi("org/a", "http://localhost:8080/v1")
    settings = m.pi_settings(); settings["packages"] = ["keep"]
    m.PI_SETTINGS_FILE.write_text(json.dumps(settings))
    shutil.rmtree(c / "models--org--a")
    assert m.refresh_pi_models() == 0
    settings = m.pi_settings()
    assert "defaultProvider" not in settings and "defaultModel" not in settings
    assert settings["packages"] == ["keep"]

@test
def atomic_write_is_private_and_complete():
    path = Path(tempfile.mkdtemp()) / "state.json"
    m._atomic_write_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

@test
def stale_process_identity_is_never_signaled():
    fresh_state()
    state = {"version": 1, "instance": "old", "pid": 123, "pgid": 123,
             "started": "then", "command": "mlx_lm.server --model org/a",
             "backend": "lm", "model": "org/a", "host": "127.0.0.1", "port": 8080}
    m._atomic_write_json(m.STATE_FILE, state)
    saved_identity, saved_killpg = m._process_identity, m.os.killpg
    calls = []
    m._process_identity = lambda pid: {"started": "now", "pgid": pid, "command": "sleep 99"}
    m.os.killpg = lambda *args: calls.append(args)
    try:
        assert m._stop_locked() is False
        assert calls == []
        assert not m.STATE_FILE.exists()
    finally:
        m._process_identity, m.os.killpg = saved_identity, saved_killpg

@test
def matching_process_state_is_recognized():
    fresh_state()
    identity = {"started": "now", "pgid": 456, "command": "mlx_lm.server --model org/a"}
    state = {"version": 1, "instance": "x", "pid": 456, **identity,
             "backend": "lm", "model": "org/a", "host": "127.0.0.1", "port": 8080}
    m._atomic_write_json(m.STATE_FILE, state)
    saved = m._process_identity; m._process_identity = lambda pid: identity
    try:
        assert m.server_pid() == 456
        assert m.served_model(456) == "org/a"
        assert m.server_backend(456) == "lm"
    finally:
        m._process_identity = saved

@test
def readiness_uses_health_and_checks_vlm_model():
    class Response:
        status_code = 200
        def __init__(self, data): self._data = data
        def json(self): return self._data
    payload = {"status": "healthy", "loaded_model": "org/right"}
    class Client:
        def __init__(self, **kwargs): assert kwargs["trust_env"] is False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url):
            assert url.endswith("/health")
            return Response(payload)
        def post(self, url, json):
            assert url.endswith("/v1/chat/completions") and json["max_tokens"] == 1
            return Response({})
    saved = m.httpx.Client; m.httpx.Client = Client
    try:
        assert m.server_ready("0.0.0.0", 8080, "vlm", "org/right") is True
        assert m.server_ready("0.0.0.0", 8080, "vlm", "org/wrong") is False
        assert m.server_ready("127.0.0.1", 8080, "lm", "org/any") is True
    finally:
        m.httpx.Client = saved

@test
def lm_wait_issues_exactly_one_generation_probe():
    calls = {"get": 0, "post": 0}
    class Response:
        status_code = 200
        def json(self): return {"status": "ok"}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url): calls["get"] += 1; return Response()
        def post(self, url, json): calls["post"] += 1; return Response()
    saved = m.httpx.Client; m.httpx.Client = Client
    try:
        assert m._wait_until_ready("127.0.0.1", 8080, "lm", "org/a",
                                   m.time.monotonic() + 5, lambda: True)
        assert calls["get"] == 1 and calls["post"] == 1
    finally:
        m.httpx.Client = saved

@test
def generated_plist_round_trips_and_is_private():
    tmp = fresh_state(); fresh_cache()
    out = tmp / "server.plist"
    args = argparse.Namespace(model="org/a&b", qwen_coder=False, gemma=False,
                              host="127.0.0.1", port=8080, output=str(out),
                              allow_network=False)
    old = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = "token<&>"
    try:
        m.cmd_plist(args)
        data = plistlib.loads(out.read_bytes())
        assert "org/a&b" in data["ProgramArguments"]
        assert data["EnvironmentVariables"]["HF_TOKEN"] == "token<&>"
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
    finally:
        if old is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = old

@test
def uninstall_preserves_foreign_symlink():
    tmp = Path(tempfile.mkdtemp())
    foreign = tmp / "foreign"; foreign.write_text("x")
    link = tmp / "mlx-pi"; link.symlink_to(foreign)
    m.cmd_uninstall(argparse.Namespace(dir=str(tmp)))
    assert link.is_symlink()

@test
def validation_rejects_bad_ports_and_network_bind():
    assert m._port("1") == 1 and m._port("65535") == 65535
    for value in ("0", "65536", "nope"):
        try:
            m._port(value)
            assert False, value
        except argparse.ArgumentTypeError:
            pass
    assert m._network_bind_allowed(argparse.Namespace(host="localhost", allow_network=False))
    assert not m._network_bind_allowed(argparse.Namespace(host="0.0.0.0", allow_network=False))

@test
def empty_provider_never_mixes_backends():
    c = fresh_cache(); fresh_pi(); fresh_state()
    make_model(c, "org/text")
    make_model(c, "org/Qwen2-VL")
    m.PI_CONFIG_FILE.write_text(json.dumps({"providers": {m.PI_PROVIDER_ID: {
        "baseUrl": "http://localhost:8080/v1", "api": "openai-completions",
        "apiKey": "x", "models": [],
    }}}))
    assert m.refresh_pi_models() == 1
    ids = m.configured_model_ids()
    assert len(ids) == 1

@test
def malformed_settings_are_never_overwritten():
    fresh_pi()
    m.PI_SETTINGS_FILE.write_text("{broken")
    try:
        m.set_pi_default("org/a")
        assert False
    except SystemExit:
        pass
    assert m.PI_SETTINGS_FILE.read_text() == "{broken"

@test
def invalid_timeout_cannot_spawn_server():
    fresh_state()
    args = argparse.Namespace(host="127.0.0.1", port=8080, model=None,
                              qwen_coder=False, gemma=False)
    saved_popen = m.subprocess.Popen
    calls = []
    m.subprocess.Popen = lambda *a, **k: calls.append((a, k))
    old = os.environ.get("MLX_STARTUP_TIMEOUT")
    os.environ["MLX_STARTUP_TIMEOUT"] = "bad"
    try:
        try:
            m._start_locked(args, m.DEFAULT_MODEL)
            assert False
        except SystemExit:
            pass
        assert calls == [] and not m.STATE_FILE.exists()
    finally:
        m.subprocess.Popen = saved_popen
        if old is None: os.environ.pop("MLX_STARTUP_TIMEOUT", None)
        else: os.environ["MLX_STARTUP_TIMEOUT"] = old

@test
def invalid_log_limit_cannot_stop_owned_server():
    fresh_state()
    identity = {"started": "now", "pgid": 778, "command": "mlx_lm.server --model org/a"}
    state = {"version": 1, "instance": "x", "phase": "ready", "pid": 778, **identity,
             "backend": "lm", "model": "org/a", "host": "127.0.0.1", "port": 8080}
    m._atomic_write_json(m.STATE_FILE, state)
    args = argparse.Namespace(host="127.0.0.1", port=8080, model="org/a",
                              qwen_coder=False, gemma=False)
    saved_identity, saved_killpg, saved_popen = m._process_identity, m.os.killpg, m.subprocess.Popen
    calls = []
    m._process_identity = lambda pid: identity
    m.os.killpg = lambda *a: calls.append(("kill", a))
    m.subprocess.Popen = lambda *a, **k: calls.append(("spawn", a, k))
    old = os.environ.get("MLX_LOG_MAX_BYTES")
    os.environ["MLX_LOG_MAX_BYTES"] = "bad"
    try:
        try:
            m._start_locked(args, "org/a", force_restart=True)
            assert False
        except SystemExit:
            pass
        assert calls == [] and m.STATE_FILE.exists()
    finally:
        m._process_identity, m.os.killpg, m.subprocess.Popen = saved_identity, saved_killpg, saved_popen
        if old is None: os.environ.pop("MLX_LOG_MAX_BYTES", None)
        else: os.environ["MLX_LOG_MAX_BYTES"] = old

@test
def recovered_starting_state_requires_real_readiness():
    fresh_state()
    identity = {"started": "now", "pgid": 779, "command": "mlx_lm.server --model org/a"}
    state = {"version": 1, "instance": "x", "phase": "starting", "pid": 779, **identity,
             "backend": "lm", "model": "org/a", "host": "127.0.0.1", "port": 8080}
    m._atomic_write_json(m.STATE_FILE, state)
    args = argparse.Namespace(host="127.0.0.1", port=8080, model=None,
                              qwen_coder=False, gemma=False)
    saved = (m._process_identity, m.is_healthy, m._wait_until_ready, m._stop_locked)
    stopped = []
    m._process_identity = lambda pid: identity
    m.is_healthy = lambda host, port: True
    m._wait_until_ready = lambda *a, **k: False
    m._stop_locked = lambda: stopped.append(True) or True
    try:
        assert m._start_locked(args, "org/a") is False
        assert stopped == [True]
        assert json.loads(m.STATE_FILE.read_text())["phase"] == "starting"
    finally:
        (m._process_identity, m.is_healthy, m._wait_until_ready, m._stop_locked) = saved

@test
def bare_up_preserves_owned_nondefault_without_default_preflight():
    fresh_state()
    identity = {"started": "now", "pgid": 777, "command": "mlx_vlm.server --model org/VL"}
    state = {"version": 1, "instance": "x", "pid": 777, **identity,
             "backend": "vlm", "model": "org/VL", "host": "127.0.0.1", "port": 8080}
    m._atomic_write_json(m.STATE_FILE, state)
    args = argparse.Namespace(host="127.0.0.1", port=8080, model=None,
                              qwen_coder=False, gemma=False, yes=False,
                              allow_network=False)
    saved = (m._process_identity, m.is_healthy, m._start_locked,
             m.require_backend, m.confirm_download)
    called = []
    m._process_identity = lambda pid: identity
    m.is_healthy = lambda host, port: True
    m._start_locked = lambda args, model: called.append(model) or True
    m.require_backend = lambda model: (_ for _ in ()).throw(AssertionError("preflight"))
    m.confirm_download = lambda model, yes: (_ for _ in ()).throw(AssertionError("prompt"))
    try:
        assert m.cmd_up(args) is True
        assert called == ["org/VL"]
    finally:
        (m._process_identity, m.is_healthy, m._start_locked,
         m.require_backend, m.confirm_download) = saved


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  \033[32mok\033[0m   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  \033[31mFAIL\033[0m {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 — report any unexpected error
            failed += 1
            print(f"  \033[31mERR\033[0m  {t.__name__}: {e!r}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
