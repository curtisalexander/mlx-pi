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
import os
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
        blob.write_bytes(b"x" * blob_size)
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
def multimodal_detection():
    assert m._pi_multimodal(m.GEMMA_MODEL) is True
    assert m._pi_multimodal("mlx-community/Qwen2-VL-7B-4bit") is True
    assert m._pi_multimodal(m.DEFAULT_MODEL) is False

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
