# mlx-pi

Run LLMs **locally** on Apple Silicon (M-series) with [MLX](https://github.com/ml-explore/mlx) as the inference engine and [pi](https://github.com/earendil-works/pi) as the coding agent. No cloud, no API keys, no token costs — once a model is downloaded you can run fully offline.

> 📖 For the full illustrated walkthrough (architecture diagrams, how to read a Hugging Face model card, quantization, launchd, troubleshooting), read the **[online guide](https://curtisalexander.github.io/mlx-pi/)** — or open [`mlx-pi-guide.html`](./mlx-pi-guide.html) in a browser. The HTML uses Mermaid from a CDN, so diagrams require network access and the file is not fully offline/self-contained.

---

## Quickstart

```bash
./bootstrap.sh        # 1. once: installs uv (the only prerequisite)
./mlx-pi setup        # 2. installs mlx-lm + mlx-vlm + pi, configures everything
./mlx-pi up           # 3. starts the local model server (background)
./mlx-pi pi           # 4. launches the coding agent against it
```

When you're done:

```bash
./mlx-pi down         # stop the server, free your RAM
```

> 💡 `setup` symlinks `mlx-pi` into `~/.local/bin` by default, so after step 2 you can drop the `./` and just run `mlx-pi` from any directory. The symlink points back at this repo, so `git pull` updates the global command automatically. Pass `setup --no-link` to skip it, run `mlx-pi install` to add it later, or `mlx-pi uninstall` to remove it. (It's a single-file `uv` script, so this symlink — not `uv tool install`, which needs a packaged project — is the right way to install it globally.)

The default model is a small **Qwen3 4B** (~2 GB) so you can confirm the whole pipeline works in a couple of minutes before committing to anything bigger.

---

## What's in this folder

| File | Purpose |
|------|---------|
| `bootstrap.sh` | ~5-line shell script. Installs `uv` and nothing else. Must be shell because a fresh Mac ships neither `uv` nor a reliable `python3`. |
| `mlx-pi` | The single tool you actually use. A Python CLI run via `uv run` (deps `rich` + `httpx` are declared inline and built automatically on first run). |
| `mlx-pi-guide.html` | Visual guide with diagrams (Mermaid is loaded from a CDN). |
| `index.html` | Redirect to the guide, so GitHub Pages serves it at the repo's Pages URL. |
| `README.md` | This file. |
| `LICENSE` | MIT license. |
| `test_mlx_pi.py` | Lightweight tests (no pytest). Run `./test_mlx_pi.py`. |

---

## How it fits together

```
You ──run──▶ ./mlx-pi pi ──HTTP /v1/chat/completions──▶ MLX server ──▶ model in
                  │                                     (localhost:8080)   unified
                  └── reads ~/.pi/agent/models.json                        memory (GPU)
```

- **`./mlx-pi setup`** installs two MLX server backends globally (via `uv tool install`, so their CLIs work from any folder) and **pi** (via npm), then writes a `local-mlx` provider into `~/.pi/agent/models.json` pointing at `http://localhost:8080/v1`:
  - **mlx-lm** (Apple's [ml-explore](https://github.com/ml-explore/mlx-lm)) — runs **text-only** LLMs via `mlx_lm.server`.
  - **mlx-vlm** (community [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)) — runs **vision** models (text **+ images**) via `mlx_vlm.server`. It's the only backend that accepts image input.
- **`./mlx-pi up`** runs the **right backend for your model** as a background daemon — `mlx_vlm.server` for a vision model like Gemma, `mlx_lm.server` for a text model like Qwen — loading it into memory behind an OpenAI-compatible chat API. Their readiness behavior differs: `mlx-pi` polls `/health`, then verifies mlx-vlm's loaded model or makes one tiny mlx-lm completion before reporting startup complete. It's the only long-running process.
- **pi** sends prompts to that local endpoint instead of the cloud, and runs its Read/Write/Edit/Bash tools locally on your machine.

---

## Commands

```text
./mlx-pi setup          Install mlx-lm + mlx-vlm + pi, point pi at the local server, and symlink mlx-pi onto your PATH (--no-link to skip).
./mlx-pi up             Start the MLX server in the background; wait until healthy.
./mlx-pi down           Stop the background server.
./mlx-pi restart        down + up.
./mlx-pi use            Switch the served model AND pi's default together (download if needed; --pin to lock pi to it).
./mlx-pi status         Show running state + health (flags drift between the served model and pi's default).
./mlx-pi logs           Tail the server log.
./mlx-pi models         List preset + downloaded models, and what pi is set to use.
./mlx-pi models pull    Download a model to the cache (no server; resumes partials).
./mlx-pi models rm      Delete a model from the cache to free disk.
./mlx-pi models clean   Remove stale *.incomplete temp files (keeps active downloads).
./mlx-pi models sync    Register all downloaded models with pi (so pi's /model lists them).
./mlx-pi models doctor  Clean + sync in one step: tidy temp files and re-align pi.
./mlx-pi run            Run the server in the FOREGROUND (Ctrl-C to stop).
./mlx-pi pi             Ensure the server is up, then launch the pi agent.
./mlx-pi plist          Generate a launchd plist for auto-start (does NOT install it).
./mlx-pi install        Symlink mlx-pi into ~/.local/bin so `mlx-pi` works anywhere.
./mlx-pi uninstall      Remove that symlink.
```

Run `./mlx-pi <command> --help` for the flags on any subcommand.

---

## Choosing a model

`setup`, `up`, `run`, `pi`, and `plist` all accept the same model flags:

| Flag | Model | Disk / RAM |
|------|-------|-----------|
| *(default)* | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 2.1 GB / ~3 GB |
| `--qwen-coder` | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | 16.0 GB / ~20 GB |
| `--gemma` | `mlx-community/gemma-4-26b-a4b-it-8bit` (text+image) | 26.1 GB / ~33 GB |
| `--model <id>` | any MLX repo from Hugging Face | varies |

Disk figures are exact (summed from Hugging Face); RAM is an estimate (~disk × 1.25). Run `mlx-pi models` for live sizes — it shows the on-disk size of downloaded models and a live HF size for ones you haven't pulled yet.

`--gemma` is a **vision** model — `mlx-pi` serves it on the `mlx_vlm.server` backend so you can paste images into pi; text models use Apple's `mlx_lm.server`. See [Text vs vision backends](#text-vs-vision-backends-images).

Examples:

```bash
./mlx-pi setup --qwen-coder      # configure pi for the coding model (no download yet)
./mlx-pi up --qwen-coder         # serve it — downloads on first run
./mlx-pi up --model mlx-community/Qwen3-4B-Instruct-2507-4bit
```

> ⚠️ **`setup` configures; downloads are separate.** `setup --qwen-coder` only points pi at the model — it does **not** download the (multi-GB) weights. You have three ways to get them:
> - **Explicitly:** `./mlx-pi models pull --qwen-coder` downloads to the cache without starting a server (and resumes interrupted downloads).
> - **During setup:** add `-p`/`--prefetch` — `./mlx-pi setup --qwen-coder -p`.
> - **Lazily:** the first `./mlx-pi up --qwen-coder` fetches it as the server boots.
>
> Either way, **pass the same model flag to `up`/`pi`** — they do *not* inherit `setup`'s choice, so use `./mlx-pi up --qwen-coder` (or persist it with `export MLX_MODEL=<id>`), else the default 4B is served and mismatches what pi calls. If the server is **already running on a different model**, an explicit flag now makes `up`/`pi` **restart it onto the model you asked for** (a bare `up` with no flag leaves the running server alone). `pi --<flag>` also points pi's default at that model, so server and pi open in lockstep.

**Before downloading anything**, the tool queries Hugging Face for the exact size, prints the download size + a rough RAM estimate, warns if it exceeds your memory, and asks you to confirm (default **No**). Add `-y`/`--yes` to skip the prompt. If the model is already cached, it proceeds silently.

Run **`./mlx-pi models`** any time to see the presets, which are downloaded (with on-disk size — partial/interrupted downloads are flagged as `⏳ partial`), and which one pi is currently configured to call. Use **`./mlx-pi models pull <id|flags>`** to download one ahead of time (resumes partials), **`./mlx-pi models rm <id|flags>`** to delete one and reclaim disk, and **`./mlx-pi models clean`** to sweep stale `*.incomplete` temp files left by interrupted downloads (it keeps any download still in progress). `pull`/`rm` accept either a preset flag (`--qwen-coder`) or a model id directly, e.g. `./mlx-pi models rm mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`.

### Multiple models in pi

pi lists downloaded models that are compatible with the **active/default backend**, not every cached model. This prevents a text server from being offered vision-only models (and vice versa). Switch backend and model together with `mlx-pi use <model>`. The list stays in sync automatically: `setup` and `models pull` update `~/.pi/agent/models.json`, `models rm` removes entries, and pi's `/model` reloads that file without requiring a restart. If you download a model another way (or the list looks stale), run:

```bash
./mlx-pi models sync     # rebuild pi's model list from what's in the cache
./mlx-pi models doctor   # one-shot: clean stale temp files AND re-sync pi
```

`models doctor` is the "just make everything tidy and aligned" button — it runs `clean` then `sync` in one go. The model you passed to `setup` (or the existing default) stays pi's **default**; compatible downloaded models are selectable via `/model`. In `./mlx-pi models`, the `pi` column shows `▸` for pi's default and `✓` for the other models pi can use.

### Text vs vision backends (images)

Some models are **vision** models — they accept images, not just text (e.g. **Gemma 4**). These need a different server: `mlx_lm.server` is text-only and rejects images with *"Only 'text' content type is supported"*. So `mlx-pi` runs each model on the **right backend automatically**:

| Model kind | Backend | Package | Images? |
|---|---|---|---|
| Text LLMs (Qwen, Qwen-Coder, …) | `mlx_lm.server` | mlx-lm (Apple) | ❌ |
| Vision models (Gemma 4, *-VL, LLaVA, …) | `mlx_vlm.server` | mlx-vlm (community) | ✅ |

You don't choose the backend — `mlx-pi up --gemma` launches `mlx_vlm.server`, `mlx-pi up --qwen-coder` launches `mlx_lm.server`. The same model's image capability is advertised to pi to match, so pi only offers image paste when the backend can actually handle it. `./mlx-pi status` shows which backend is live (`backend  vision · mlx_vlm.server (images ✅)`), and `./mlx-pi models` lists each model's modality.

Detection is **metadata-first**: cached Hugging Face `config.json` metadata is used when decisive, then a name heuristic handles unknown/custom configs. Explicit environment overrides remain available:

```bash
export MLX_VISION_MODELS=org/my-vlm-with-an-odd-name   # force vision backend
export MLX_TEXT_MODELS=org/not-actually-vision         # force text backend
```

> **Note:** `mlx-vlm` is a well-regarded *community* package (not Apple-maintained, unlike `mlx-lm`/`mlx`). Text generation runs on the same MLX kernels, so speed is comparable; sending an image adds a one-time vision-encode cost on that request. Your text models are untouched — they keep running on Apple's `mlx_lm.server`.

### Keeping the server and pi in sync

The MLX server is launched with **one** model (its `--model`), but that's only a *preload/default*: the server **hot-swaps to whatever model a request names** (within the same backend). So when you `/model` to a different model inside pi, the server reloads that model on the fly. You won't get the wrong weights — but the swap costs a reload (seconds for small models, much longer for big ones) and the new model's RAM. The drift to watch for is *operational*, not correctness.

There are two separate things to keep aligned: which model the **server preloaded** (its `--model`) and which model **pi opens on**. pi's startup model lives in `~/.pi/agent/settings.json` (`defaultProvider` + `defaultModel`) — **not** in the order of the `models.json` list — so changing it means writing that file, which `setup`, `use`, and `pi --<model>` do for you.

- **`./mlx-pi up --<model>` / `pi --<model>`** — when an explicit model differs from the one a running server has, they **restart the server onto it** instead of just warning (a bare `up` leaves the running server alone). `pi --<model>` additionally sets pi's `defaultModel`, so the server and pi open on the same model.
- **`./mlx-pi use <model>`** moves both sides together *without launching pi*: downloads if needed, restarts the server on it, and sets it as pi's `defaultModel`. The deliberate "switch everything to this model" command.
- **`./mlx-pi status`** shows the **preloaded** model, the **backend** serving it, and **pi's default** side by side, and warns when the model and pi's default differ. mlx-vlm's `/health` can report its resident model; mlx-lm readiness does not expose equivalent resident-model information.
- **Pinned mode** (`--pin` on `setup`/`use`, or `export MLX_PIN_MODEL=1`) registers **only** the served model with pi, so pi's picker can't drift to something that triggers a surprise reload. Good default on low-RAM machines.
- **RAM guard** (on by default) hides models from pi's picker that clearly won't fit in your installed RAM, so selecting one can't OOM the box. The model you explicitly chose is always kept, and hidden models are reported (not silently dropped). Disable with `export MLX_NO_RAM_GUARD=1`.

### Hugging Face authentication (`HF_TOKEN`)

Set a [Hugging Face token](https://huggingface.co/settings/tokens) if you hit **gated models** (you must accept the model's license first) or want **higher download rate limits** (anonymous downloads can be throttled — a common cause of slow pulls):

```bash
export HF_TOKEN=hf_xxxxx        # bash/zsh — add to your shell rc to persist
set -Ux HF_TOKEN hf_xxxxx       # fish (universal, persists)
```

`models pull`, `up`, and `pi` all run their downloads as child processes that inherit your environment, so `HF_TOKEN` is picked up automatically — no flag needed. (A token saved via `hf auth login` works too.)

> ⚠️ The **launchd** auto-start server runs with a minimal environment and doesn't inherit your shell's `HF_TOKEN`. To handle this, `./mlx-pi plist` **bakes `HF_TOKEN` into the generated plist** if it's set in your environment when you run it — so export the token *before* generating. The token is stored **in plaintext** in `com.mlx-pi.server.plist`, so keep that file private (don't commit or share it). If no token is set, the plist is generated without one and `plist` tells you so.

> 💡 Want the MLX format. Look for repos under `mlx-community` (or `lmstudio-community` …-MLX-…). Avoid `.gguf` files — those are for llama.cpp/Ollama, not `mlx_lm`. See guide §2 for how to read model names, quantization (`4bit`/`8bit`/`bf16`), and `A3B`/`A4B` MoE "active params".

---

## Run at startup (optional)

`./mlx-pi plist` writes a **personalized** `com.mlx-pi.server.plist` (your real home dir, model, and port already filled in). It does **not** install it — you opt in:

```bash
./mlx-pi plist --qwen-coder                                   # generate
cp com.mlx-pi.server.plist ~/Library/LaunchAgents/            # install
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mlx-pi.server.plist # enable
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.mlx-pi.server.plist   # disable later
```

The generated plist is written with mode `0600`, particularly important when it contains a plaintext `HF_TOKEN`.

> ⚠️ A launchd server is **always on** — it holds the model in RAM from login. Great on a dedicated machine; on a laptop you may prefer manual `up`/`down`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `uv: command not found` running `./mlx-pi` | Run `./bootstrap.sh` first, then open a new terminal. |
| First `./mlx-pi` run pauses a few seconds | Normal — uv is building the script's dep env. Cached afterward. |
| `mlx_lm.server: command not found` | Open a new terminal so PATH refreshes. fish: `fish_add_path ~/.local/bin`. |
| pi hangs on first message | Server is still downloading the model — watch `./mlx-pi logs`. |
| Out of memory / sluggish | Model too big for your RAM — use a smaller one and close heavy apps. |
| Port already in use | `./mlx-pi up --port 8090` and re-run `./mlx-pi setup --port 8090`. |
| Download very slow, or `401`/gated error | Set `HF_TOKEN` (see [Hugging Face authentication](#hugging-face-authentication-hf_token)) — auth raises rate limits and unlocks gated models. |

---

## Environment variables

All optional — sensible defaults otherwise.

| Variable | Effect |
|----------|--------|
| `MLX_MODEL` | Default model id when no `--model`/preset flag is given (persists your choice across `up`/`pi`/`setup`). |
| `MLX_PIN_MODEL` | `1` to register **only** the served model with pi (no hot-swap drift); same as `--pin`. |
| `MLX_NO_RAM_GUARD` | `1` to register models even if they likely exceed installed RAM (the guard is on by default). |
| `MLX_VISION_MODELS` | Comma-separated repo ids to force onto the **vision** backend (`mlx_vlm.server`) when the name heuristic misses. |
| `MLX_TEXT_MODELS` | Comma-separated repo ids to force onto the **text** backend (`mlx_lm.server`), overriding vision detection. |
| `MLX_STATE_DIR` | Where versioned ownership state (`server.json`) and the server log live (default `~/.mlx-pi`). The CLI validates process identity; it does not trust a bare PID. |
| `MLX_LOG_MAX_BYTES` | Cap for `server.log` before it's trimmed (default `10485760` = 10 MB). |
| `MLX_STARTUP_TIMEOUT` | Seconds `up` waits for the server to become healthy before giving up (default `600`). |
| `HF_TOKEN` | Hugging Face token for gated models / higher download rate limits (see above). |
| `HF_HOME` / `HF_HUB_CACHE` | Relocate the Hugging Face cache; `mlx-pi` honors these when finding downloaded models. |

---

## Requirements

- Apple Silicon Mac (M1 or newer; tuned for M5's Metal-4 tensor accelerators)
- macOS with `curl` (preinstalled)
- Internet access for the one-time installs and model downloads

The scripts install `uv`, both MLX backends, pi, and Python dependencies. If Node is missing, `setup` installs it through Homebrew when available; otherwise install Node LTS first.

By default the API binds only to loopback. A non-loopback `--host` is refused unless you explicitly pass `--allow-network`; doing so exposes an **unauthenticated API** to any network that can reach that address.
