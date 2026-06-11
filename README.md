# mlx-pi

Run LLMs **locally** on Apple Silicon (M-series) with [MLX](https://github.com/ml-explore/mlx) as the inference engine and [pi](https://github.com/earendil-works/pi) as the coding agent. No cloud, no API keys, no token costs — once a model is downloaded you can run fully offline.

> 📖 For the full illustrated walkthrough (architecture diagrams, how to read a Hugging Face model card, quantization, launchd, troubleshooting), read the **[online guide](https://curtisalexander.github.io/mlx-pi/)** — or open the source [`mlx-pi-guide.html`](./mlx-pi-guide.html) in a browser.

---

## Quickstart

```bash
./bootstrap.sh        # 1. once: installs uv (the only prerequisite)
./mlx-pi setup        # 2. installs mlx-lm + pi, configures everything
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
| `mlx-pi-guide.html` | Self-contained visual guide with diagrams. |
| `README.md` | This file. |

---

## How it fits together

```
You ──run──▶ ./mlx-pi pi ──HTTP /v1/chat/completions──▶ MLX server ──▶ model in
                  │                                     (localhost:8080)   unified
                  └── reads ~/.pi/agent/models.json                        memory (GPU)
```

- **`./mlx-pi setup`** installs **mlx-lm** globally (via `uv tool install`, so the `mlx_lm.*` commands work from any folder) and **pi** (via npm), then writes a `local-mlx` provider into `~/.pi/agent/models.json` pointing at `http://localhost:8080/v1`.
- **`./mlx-pi up`** runs `mlx_lm.server` as a **background daemon** that loads your model into memory and exposes an OpenAI-compatible API. It's the only long-running process.
- **pi** sends prompts to that local endpoint instead of the cloud, and runs its Read/Write/Edit/Bash tools locally on your machine.

---

## Commands

```text
./mlx-pi setup      Install mlx-lm + pi, point pi at the local server, and symlink mlx-pi onto your PATH (--no-link to skip).
./mlx-pi up         Start the MLX server in the background; wait until healthy.
./mlx-pi down       Stop the background server.
./mlx-pi restart    down + up.
./mlx-pi status     Show running state + health.
./mlx-pi logs       Tail the server log.
./mlx-pi models       List preset + downloaded models, and what pi is set to use.
./mlx-pi models pull  Download a model to the cache (no server; resumes partials).
./mlx-pi models rm    Delete a model from the cache to free disk.
./mlx-pi run        Run the server in the FOREGROUND (Ctrl-C to stop).
./mlx-pi pi         Ensure the server is up, then launch the pi agent.
./mlx-pi plist      Generate a launchd plist for auto-start (does NOT install it).
./mlx-pi install    Symlink mlx-pi into ~/.local/bin so `mlx-pi` works anywhere.
./mlx-pi uninstall  Remove that symlink.
```

Run `./mlx-pi <command> --help` for the flags on any subcommand.

---

## Choosing a model

`setup`, `up`, `run`, `pi`, and `plist` all accept the same model flags:

| Flag | Model | Disk / RAM |
|------|-------|-----------|
| *(default)* | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2 GB / ~16 GB |
| `--qwen-coder` | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | ~16 GB / ~32 GB |
| `--gemma` | `mlx-community/gemma-4-26b-a4b-it-4bit` (text+image) | ~15.6 GB / ~32 GB |
| `--model <id>` | any MLX repo from Hugging Face | varies |

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
> Either way, **pass the same model flag to `up`** — it does *not* inherit `setup`'s choice, so `./mlx-pi up --qwen-coder` (or persist it with `export MLX_MODEL=<id>`), else it serves the default 4B and mismatches what pi calls.

**Before downloading anything**, the tool queries Hugging Face for the exact size, prints the download size + a rough RAM estimate, warns if it exceeds your memory, and asks you to confirm (default **No**). Add `-y`/`--yes` to skip the prompt. If the model is already cached, it proceeds silently.

Run **`./mlx-pi models`** any time to see the presets, which are downloaded (with on-disk size — partial/interrupted downloads are flagged as `⏳ partial`), and which one pi is currently configured to call. Use **`./mlx-pi models pull [flags]`** to download one ahead of time (resumes partials) and **`./mlx-pi models rm [flags]`** to delete one and reclaim disk.

### Hugging Face authentication (`HF_TOKEN`)

Set a [Hugging Face token](https://huggingface.co/settings/tokens) if you hit **gated models** (you must accept the model's license first) or want **higher download rate limits** (anonymous downloads can be throttled — a common cause of slow pulls):

```bash
export HF_TOKEN=hf_xxxxx        # bash/zsh — add to your shell rc to persist
set -Ux HF_TOKEN hf_xxxxx       # fish (universal, persists)
```

`models pull`, `up`, and `pi` all run their downloads as child processes that inherit your environment, so `HF_TOKEN` is picked up automatically — no flag needed. (A token saved via `hf auth login` works too.)

> ⚠️ The **launchd** auto-start server (`./mlx-pi plist`) runs with a minimal environment and does **not** see `HF_TOKEN`. If you auto-start *and* use a gated model, set the token in the plist (or pre-download with `./mlx-pi models pull` while authenticated).

> 💡 Want the MLX format. Look for repos under `mlx-community` (or `lmstudio-community` …-MLX-…). Avoid `.gguf` files — those are for llama.cpp/Ollama, not `mlx_lm`. See guide §2 for how to read model names, quantization (`4bit`/`8bit`/`bf16`), and `A3B`/`A4B` MoE "active params".

---

## Run at startup (optional)

`./mlx-pi plist` writes a **personalized** `com.mlx-pi.server.plist` (your real home dir, model, and port already filled in). It does **not** install it — you opt in:

```bash
./mlx-pi plist --qwen-coder                                   # generate
cp com.mlx-pi.server.plist ~/Library/LaunchAgents/            # install
launchctl load ~/Library/LaunchAgents/com.mlx-pi.server.plist # enable
launchctl unload ~/Library/LaunchAgents/com.mlx-pi.server.plist  # disable later
```

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

## Requirements

- Apple Silicon Mac (M1 or newer; tuned for M5's Metal-4 tensor accelerators)
- macOS with `curl` (preinstalled)
- Internet access for the one-time installs and model downloads

Everything else (`uv`, `mlx-lm`, Node, `pi`, Python deps) is installed by the two scripts.
