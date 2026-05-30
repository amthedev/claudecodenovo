# LLM API Key Proxy — Claude Code Instructions

## Project Overview

Universal LLM API proxy (FastAPI) with OpenAI and Anthropic-compatible endpoints. Built in Python.

- `src/proxy_app/` — FastAPI app, routes, admin, TUI launcher
- `src/rotator_library/` — API key rotation/resilience library
- `proxy_config.json` — local config (env vars, API keys placeholder)
- `requirements.txt`, `docker-compose.yml`, `Dockerfile`

## Orientation

**Do NOT run `pwd && ls -la` or any shell orientation commands automatically.** You already have the project context from this file. Only run commands when explicitly needed for a task.

## Secrets & API Keys — CRITICAL

This project manages API keys by design. The following files contain or may contain real credentials:

- `.env` (if present)
- `proxy_config.json`
- Any file matching `*credentials*`, `*oauth_creds*`, `*token*`

**Rules:**
1. Never display, print, or echo the content of API keys, tokens, or secrets — even partially
2. Never include secret values in shell commands that would appear in output
3. If you find a secret while exploring code, acknowledge it exists without showing its value
4. Do not search for secrets unless the user explicitly requests a security audit

## Think Before Coding

Before implementing anything:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## Simplicity First

Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## Behavior Guidelines

- Be concise. Avoid repeating what you just did
- Don't add comments to code unless the reason is non-obvious
- Don't create new files unless necessary — prefer editing existing ones
- Run tests with `pytest tests/` from the project root
- The project uses Python; dependencies are in `requirements.txt`
