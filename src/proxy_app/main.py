# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

import time
import uuid

# Phase 1: Minimal imports for arg parsing and TUI
import asyncio
import os
from pathlib import Path
import sys
import argparse
import logging
import hashlib
import hmac
import secrets
from urllib.parse import parse_qs

# --- Argument Parsing (BEFORE heavy imports) ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument(
    "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
)
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
parser.add_argument(
    "--enable-request-logging",
    action="store_true",
    help="Enable transaction logging in the library (logs request/response with provider correlation).",
)
parser.add_argument(
    "--enable-raw-logging",
    action="store_true",
    help="Enable raw I/O logging at proxy boundary (captures unmodified HTTP data, disabled by default).",
)
parser.add_argument(
    "--add-credential",
    action="store_true",
    help="Launch the interactive tool to add a new OAuth credential.",
)
args, _ = parser.parse_known_args()

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Check if we should launch TUI (no arguments = TUI mode)
if len(sys.argv) == 1:
    # TUI MODE - Load ONLY what's needed for the launcher (fast path!)
    from proxy_app.launcher_tui import run_launcher_tui

    run_launcher_tui()
    # Launcher modifies sys.argv and returns, or exits if user chose Exit
    # If we get here, user chose "Run Proxy" and sys.argv is modified
    # Re-parse arguments with modified sys.argv
    args = parser.parse_args()

# Check if credential tool mode (also doesn't need heavy proxy imports)
if args.add_credential:
    from rotator_library.credential_tool import run_credential_tool

    run_credential_tool()
    sys.exit(0)

# If we get here, we're ACTUALLY running the proxy - NOW show startup messages and start timer
_start_time = time.time()

# Load all .env files from root folder (main .env first, then any additional *.env files)
from dotenv import load_dotenv
from glob import glob

# Get the application root directory (EXE dir if frozen, else CWD)
# Inlined here to avoid triggering heavy rotator_library imports before loading screen
if getattr(sys, "frozen", False):
    _root_dir = Path(sys.executable).parent
else:
    _root_dir = Path.cwd()

# Load main .env first
load_dotenv(_root_dir / ".env")

# Load any additional .env files (e.g., gemini_cli_all_combined.env)
_env_files_found = list(_root_dir.glob("*.env"))
for _env_file in sorted(_root_dir.glob("*.env")):
    if _env_file.name != ".env":  # Skip main .env (already loaded)
        load_dotenv(_env_file, override=False)  # Don't override existing values

# Log discovered .env files for deployment verification
if _env_files_found:
    _env_names = [_ef.name for _ef in _env_files_found]
    print(f"📁 Loaded {len(_env_files_found)} .env file(s): {', '.join(_env_names)}")

# Get proxy API key for display
proxy_api_key = os.getenv("PROXY_API_KEY")
if proxy_api_key:
    key_display = f"✓ {proxy_api_key}"
else:
    key_display = "✗ Not Set (INSECURE - anyone can access!)"

print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print(f"GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
print("━" * 70)
print("Loading server components...")


# Phase 2: Load Rich for loading spinner (lightweight)
from rich.console import Console

_console = Console()

# Phase 3: Heavy dependencies with granular loading messages
print("  → Loading FastAPI framework...")
with _console.status("[dim]Loading FastAPI framework...", spinner="dots"):
    from contextlib import asynccontextmanager
    from html import escape
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, RedirectResponse
    from fastapi.security import APIKeyHeader

print("  → Loading core dependencies...")
with _console.status("[dim]Loading core dependencies...", spinner="dots"):
    from dotenv import load_dotenv
    import colorlog
    import json
    from typing import AsyncGenerator, Any, Dict, List, Optional, Union
    from pydantic import BaseModel, ConfigDict, Field

    # --- Early Log Level Configuration ---
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

print("  → Loading LiteLLM library...")
with _console.status("[dim]Loading LiteLLM library...", spinner="dots"):
    import litellm

litellm.suppress_debug_info = True

# Phase 4: Application imports with granular loading messages
print("  → Initializing proxy core...")
with _console.status("[dim]Initializing proxy core...", spinner="dots"):
    from rotator_library import RotatingClient
    from rotator_library.credential_manager import CredentialManager
    from rotator_library.background_refresher import BackgroundRefresher
    from rotator_library.model_info_service import init_model_info_service
    from proxy_app.request_logger import log_request_to_console
    from proxy_app.batch_manager import EmbeddingBatcher
    from proxy_app.detailed_logger import RawIOLogger

print("  → Discovering provider plugins...")
# Provider lazy loading happens during import, so time it here
_provider_start = time.time()
with _console.status("[dim]Discovering provider plugins...", spinner="dots"):
    from rotator_library import (
        PROVIDER_PLUGINS,
    )  # This triggers lazy load via __getattr__
_provider_time = time.time() - _provider_start

# Get count after import (without timing to avoid double-counting)
_plugin_count = len(PROVIDER_PLUGINS)


# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    input_type: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ModelCard(BaseModel):
    """Basic model card for minimal response."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Mirro-Proxy"


class ModelCapabilities(BaseModel):
    """Model capability flags."""

    tool_choice: bool = False
    function_calling: bool = False
    reasoning: bool = False
    vision: bool = False
    system_messages: bool = True
    prompt_caching: bool = False
    assistant_prefill: bool = False


class EnrichedModelCard(BaseModel):
    """Extended model card with pricing and capabilities."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "unknown"
    # Pricing (optional - may not be available for all models)
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    cache_read_input_token_cost: Optional[float] = None
    cache_creation_input_token_cost: Optional[float] = None
    # Limits (optional)
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    # Capabilities
    mode: str = "chat"
    supported_modalities: List[str] = Field(default_factory=lambda: ["text"])
    supported_output_modalities: List[str] = Field(default_factory=lambda: ["text"])
    capabilities: Optional[ModelCapabilities] = None
    # Debug info (optional)
    _sources: Optional[List[str]] = None
    _match_type: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields from the service


class ModelList(BaseModel):
    """List of models response."""

    object: str = "list"
    data: List[ModelCard]


class EnrichedModelList(BaseModel):
    """List of enriched models with pricing and capabilities."""

    object: str = "list"
    data: List[EnrichedModelCard]


# --- Anthropic API Models (imported from library) ---
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    AnthropicThinkingConfig,
)


# Calculate total loading time
_elapsed = time.time() - _start_time
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)

# Clear screen and reprint header for clean startup view
# This pushes loading messages up (still in scroll history) but shows a clean final screen
import os as _os_module

_os_module.system("cls" if _os_module.name == "nt" else "clear")

# Reprint header
print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print(f"GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
print("━" * 70)
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)


# Note: Debug logging will be added after logging configuration below

# --- Logging Configuration ---
# Import path utilities here (after loading screen) to avoid triggering heavy imports early
from rotator_library.utils.paths import get_logs_dir, get_data_file

LOG_DIR = get_logs_dir(_root_dir)

# Configure a console handler with color (INFO and above only, no DEBUG)
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)

# Configure a file handler for INFO-level logs and higher
info_file_handler = logging.FileHandler(LOG_DIR / "proxy.log", encoding="utf-8")
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure a dedicated file handler for all DEBUG-level logs
debug_file_handler = logging.FileHandler(LOG_DIR / "proxy_debug.log", encoding="utf-8")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)


# Create a filter to ensure the debug handler ONLY gets DEBUG messages from the rotator_library
class RotatorDebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG and record.name.startswith(
            "rotator_library"
        )


debug_file_handler.addFilter(RotatorDebugFilter())

# Configure a console handler with color
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)


# Add a filter to prevent any LiteLLM logs from cluttering the console
class NoLiteLLMLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("LiteLLM")


console_handler.addFilter(NoLiteLLMLogFilter())

# Get the root logger and set it to DEBUG to capture all messages
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Add all handlers to the root logger
root_logger.addHandler(info_file_handler)
root_logger.addHandler(console_handler)
root_logger.addHandler(debug_file_handler)

# Silence other noisy loggers by setting their level higher than root
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Isolate LiteLLM's logger to prevent it from reaching the console.
# We will capture its logs via the logger_fn callback in the client instead.
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.handlers = []
litellm_logger.propagate = False

# Now that logging is configured, log the module load time to debug file only
logging.debug(f"Modules loaded in {_elapsed:.2f}s")

# Load environment variables from .env file
load_dotenv(_root_dir / ".env")

# --- Configuration ---
USE_EMBEDDING_BATCHER = False
ENABLE_REQUEST_LOGGING = args.enable_request_logging
ENABLE_RAW_LOGGING = args.enable_raw_logging
if ENABLE_REQUEST_LOGGING:
    logging.info(
        "Transaction logging is enabled (library-level with provider correlation)."
    )
if ENABLE_RAW_LOGGING:
    logging.info("Raw I/O logging is enabled (proxy boundary, unmodified HTTP data).")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
# Note: PROXY_API_KEY validation moved to server startup to allow credential tool to run first

# Discover API keys from environment variables
api_keys = {}
for key, value in os.environ.items():
    if "_API_KEY" in key and key != "PROXY_API_KEY":
        provider = key.split("_API_KEY")[0].lower()
        if provider not in api_keys:
            api_keys[provider] = []
        api_keys[provider].append(value)

# Load model ignore lists from environment variables
ignore_models = {}
for key, value in os.environ.items():
    if key.startswith("IGNORE_MODELS_"):
        provider = key.replace("IGNORE_MODELS_", "").lower()
        models_to_ignore = [
            model.strip() for model in value.split(",") if model.strip()
        ]
        ignore_models[provider] = models_to_ignore
        logging.debug(
            f"Loaded ignore list for provider '{provider}': {models_to_ignore}"
        )

# Load model whitelist from environment variables
whitelist_models = {}
for key, value in os.environ.items():
    if key.startswith("WHITELIST_MODELS_"):
        provider = key.replace("WHITELIST_MODELS_", "").lower()
        models_to_whitelist = [
            model.strip() for model in value.split(",") if model.strip()
        ]
        whitelist_models[provider] = models_to_whitelist
        logging.debug(
            f"Loaded whitelist for provider '{provider}': {models_to_whitelist}"
        )


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the RotatingClient's lifecycle with the app's lifespan."""
    # [MODIFIED] Perform skippable OAuth initialization at startup
    skip_oauth_init = os.getenv("SKIP_OAUTH_INIT_CHECK", "false").lower() == "true"

    # The CredentialManager now handles all discovery, including .env overrides.
    # We pass all environment variables to it for this purpose.
    cred_manager = CredentialManager(os.environ)
    oauth_credentials = cred_manager.discover_and_prepare()

    if not skip_oauth_init and oauth_credentials:
        logging.info("Starting OAuth credential validation and deduplication...")
        processed_emails = {}  # email -> {provider: path}
        credentials_to_initialize = {}  # provider -> [paths]
        final_oauth_credentials = {}

        # --- Pass 1: Pre-initialization Scan & Deduplication ---
        # logging.info("Pass 1: Scanning for existing metadata to find duplicates...")
        for provider, paths in oauth_credentials.items():
            if provider not in credentials_to_initialize:
                credentials_to_initialize[provider] = []
            for path in paths:
                # Skip env-based credentials (virtual paths) - they don't have metadata files
                if path.startswith("env://"):
                    credentials_to_initialize[provider].append(path)
                    continue

                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    metadata = data.get("_proxy_metadata", {})
                    email = metadata.get("email")

                    if email:
                        if email not in processed_emails:
                            processed_emails[email] = {}

                        if provider in processed_emails[email]:
                            original_path = processed_emails[email][provider]
                            logging.warning(
                                f"Duplicate for '{email}' on '{provider}' found in pre-scan: '{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                            )
                            continue
                        else:
                            processed_emails[email][provider] = path

                    credentials_to_initialize[provider].append(path)

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logging.warning(
                        f"Could not pre-read metadata from '{path}': {e}. Will process during initialization."
                    )
                    credentials_to_initialize[provider].append(path)

        # --- Pass 2: Parallel Initialization of Filtered Credentials ---
        # logging.info("Pass 2: Initializing unique credentials and performing final check...")
        async def process_credential(provider: str, path: str, provider_instance):
            """Process a single credential: initialize and fetch user info."""
            try:
                await provider_instance.initialize_token(path)

                if not hasattr(provider_instance, "get_user_info"):
                    return (provider, path, None, None)

                user_info = await provider_instance.get_user_info(path)
                email = user_info.get("email")
                return (provider, path, email, None)

            except Exception as e:
                logging.error(
                    f"Failed to process OAuth token for {provider} at '{path}': {e}"
                )
                return (provider, path, None, e)

        # Collect all tasks for parallel execution
        tasks = []
        for provider, paths in credentials_to_initialize.items():
            if not paths:
                continue

            provider_plugin_class = PROVIDER_PLUGINS.get(provider)
            if not provider_plugin_class:
                continue

            provider_instance = provider_plugin_class()

            for path in paths:
                tasks.append(process_credential(provider, path, provider_instance))

        # Execute all credential processing tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- Pass 3: Sequential Deduplication and Final Assembly ---
        for result in results:
            # Handle exceptions from gather
            if isinstance(result, Exception):
                logging.error(f"Credential processing raised exception: {result}")
                continue

            provider, path, email, error = result

            # Skip if there was an error
            if error:
                continue

            # If provider doesn't support get_user_info, add directly
            if email is None:
                if provider not in final_oauth_credentials:
                    final_oauth_credentials[provider] = []
                final_oauth_credentials[provider].append(path)
                continue

            # Handle empty email
            if not email:
                logging.warning(
                    f"Could not retrieve email for '{path}'. Treating as unique."
                )
                if provider not in final_oauth_credentials:
                    final_oauth_credentials[provider] = []
                final_oauth_credentials[provider].append(path)
                continue

            # Deduplication check
            if email not in processed_emails:
                processed_emails[email] = {}

            if (
                provider in processed_emails[email]
                and processed_emails[email][provider] != path
            ):
                original_path = processed_emails[email][provider]
                logging.warning(
                    f"Duplicate for '{email}' on '{provider}' found post-init: '{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                )
                continue
            else:
                processed_emails[email][provider] = path
                if provider not in final_oauth_credentials:
                    final_oauth_credentials[provider] = []
                final_oauth_credentials[provider].append(path)

                # Update metadata (skip for env-based credentials - they don't have files)
                if not path.startswith("env://"):
                    try:
                        with open(path, "r+") as f:
                            data = json.load(f)
                            metadata = data.get("_proxy_metadata", {})
                            metadata["email"] = email
                            metadata["last_check_timestamp"] = time.time()
                            data["_proxy_metadata"] = metadata
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()
                    except Exception as e:
                        logging.error(f"Failed to update metadata for '{path}': {e}")

        logging.info("OAuth credential processing complete.")
        oauth_credentials = final_oauth_credentials

    # [NEW] Load provider-specific params
    litellm_provider_params = {
        "gemini_cli": {"project_id": os.getenv("GEMINI_CLI_PROJECT_ID")}
    }

    # Load global timeout from environment (default 30 seconds)
    global_timeout = int(os.getenv("GLOBAL_TIMEOUT", "30"))

    # The client now uses the root logger configuration
    client = RotatingClient(
        api_keys=api_keys,
        oauth_credentials=oauth_credentials,  # Pass OAuth config
        configure_logging=True,
        global_timeout=global_timeout,
        litellm_provider_params=litellm_provider_params,
        ignore_models=ignore_models,
        whitelist_models=whitelist_models,
        enable_request_logging=ENABLE_REQUEST_LOGGING,
    )

    await client.initialize_usage_managers()

    # Log loaded credentials summary (compact, always visible for deployment verification)
    # _api_summary = ', '.join([f"{p}:{len(c)}" for p, c in api_keys.items()]) if api_keys else "none"
    # _oauth_summary = ', '.join([f"{p}:{len(c)}" for p, c in oauth_credentials.items()]) if oauth_credentials else "none"
    # _total_summary = ', '.join([f"{p}:{len(c)}" for p, c in client.all_credentials.items()])
    # print(f"🔑 Credentials loaded: {_total_summary} (API: {_api_summary} | OAuth: {_oauth_summary})")
    client.background_refresher.start()  # Start the background task
    app.state.rotating_client = client

    # Warn if no provider credentials are configured
    if not client.all_credentials:
        logging.warning("=" * 70)
        logging.warning("⚠️  NO PROVIDER CREDENTIALS CONFIGURED")
        logging.warning("The proxy is running but cannot serve any LLM requests.")
        logging.warning(
            "Launch the credential tool to add API keys or OAuth credentials."
        )
        logging.warning("  • Executable: Run with --add-credential flag")
        logging.warning("  • Source: python src/proxy_app/main.py --add-credential")
        logging.warning("=" * 70)

    os.environ["LITELLM_LOG"] = "ERROR"
    litellm.set_verbose = False
    litellm.drop_params = True
    if USE_EMBEDDING_BATCHER:
        batcher = EmbeddingBatcher(client=client)
        app.state.embedding_batcher = batcher
        logging.info("RotatingClient and EmbeddingBatcher initialized.")
    else:
        app.state.embedding_batcher = None
        logging.info("RotatingClient initialized (EmbeddingBatcher disabled).")

    # Start model info service in background (fetches pricing/capabilities data)
    # This runs asynchronously and doesn't block proxy startup
    model_info_service = await init_model_info_service()
    app.state.model_info_service = model_info_service
    logging.info("Model info service started (fetching pricing data in background).")

    # Inicializa SQLite admin e registra rotas (carregamento seguro)
    try:
        from proxy_app.admin_db import init_db as _adb_init
        from proxy_app.admin_routes import register_admin_routes as _areg
        from proxy_app.web_routes import register_web_routes as _wreg
        _adb_init()
        _areg(app, proxy_api_key=proxy_api_key)
        _wreg(app)
        logging.info("[admin_db] SQLite admin carregado com sucesso.")
    except ImportError:
        logging.warning("[admin_db] Modulos admin_db/admin_routes nao encontrados — usando admin legado.")
    except Exception as _e:
        logging.warning(f"[admin_db] Falha ao inicializar admin SQLite: {_e}")

    yield

    await client.background_refresher.stop()  # Stop the background task on shutdown
    if app.state.embedding_batcher:
        await app.state.embedding_batcher.stop()
    await client.close()

    # Stop model info service
    if hasattr(app.state, "model_info_service") and app.state.model_info_service:
        await app.state.model_info_service.stop()

    if app.state.embedding_batcher:
        logging.info("RotatingClient and EmbeddingBatcher closed.")
    else:
        logging.info("RotatingClient closed.")


# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
ADMIN_DATA_FILE = _root_dir / os.getenv("ADMIN_DATA_FILE", "admin_data.json")
ADMIN_SESSION_COOKIE = "proxy_admin_session"
ADMIN_SESSION_SECONDS = int(os.getenv("ADMIN_SESSION_SECONDS", "43200"))
PASSWORD_ITERATIONS = 260_000


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _default_admin_data() -> Dict[str, Any]:
    return {"admin": None, "sessions": {}, "apps": []}


def _load_admin_data() -> Dict[str, Any]:
    if not ADMIN_DATA_FILE.exists():
        return _default_admin_data()
    try:
        with ADMIN_DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logging.exception("Failed to load admin data; using empty admin store")
        return _default_admin_data()

    data.setdefault("admin", None)
    data.setdefault("sessions", {})
    data.setdefault("apps", [])
    return data


def _save_admin_data(data: Dict[str, Any]) -> None:
    ADMIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ADMIN_DATA_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(ADMIN_DATA_FILE)


def _hash_secret(secret_value: str, salt: Optional[str] = None) -> Dict[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        secret_value.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return {"salt": salt, "hash": digest}


def _verify_secret(secret_value: str, stored: Dict[str, str]) -> bool:
    if not secret_value or not stored:
        return False
    candidate = _hash_secret(secret_value, stored.get("salt"))
    return hmac.compare_digest(candidate["hash"], stored.get("hash", ""))


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _generate_proxy_key() -> str:
    return "sk-" + secrets.token_urlsafe(32).replace("_", "").replace("-", "")[:42]


def _parse_daily_limit(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _parse_validity_days(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _expiry_from_days(validity_days: int) -> Optional[float]:
    if validity_days <= 0:
        return None
    return time.time() + (validity_days * 86400)


def _timestamp_value(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_expired(value: Any) -> bool:
    ts = _timestamp_value(value)
    return bool(ts and ts <= time.time())


async def _read_form_data(request: Request) -> Dict[str, str]:
    body = (await request.body()).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _extract_bearer_token(auth: Optional[str]) -> Optional[str]:
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return auth.strip()


def _candidate_api_keys_from_request(request: Request) -> List[str]:
    """Collect API key candidates from common OpenAI/Anthropic header styles."""
    candidates: List[str] = []

    for header_name in (
        "x-api-key",
        "api-key",
        "anthropic-api-key",
        "authorization",
    ):
        value = request.headers.get(header_name)
        if not value:
            continue

        stripped = value.strip()
        bearer = _extract_bearer_token(stripped)
        for candidate in (bearer, stripped):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    return candidates


def _static_env_models() -> list:
    """Retorna lista de modelos configurados via variáveis de ambiente (fallback estático)."""
    models = []
    for env_name in (
        "PROXY_MODELS",
        "STATIC_MODELS",
        "HOSTED_VLLM_MODELS",
        "OPENAI_MODELS",
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                models.extend(str(m) for m in parsed if m)
            elif isinstance(parsed, dict):
                models.extend(str(k) for k in parsed.keys() if k)
        except json.JSONDecodeError:
            # Trata como lista separada por vírgula
            models.extend(m.strip() for m in raw.split(",") if m.strip())
    return list(dict.fromkeys(models))


def _default_proxy_model() -> Optional[str]:
    for env_name in (
        "PROXY_DEFAULT_MODEL",
        "DEFAULT_PROXY_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        value = os.getenv(env_name)
        if value:
            return value.strip()

    static_models = _static_env_models()
    if static_models:
        return static_models[0]
    return None


def _resolve_model_alias(model: Optional[str]) -> Optional[str]:
    if not model:
        return model
    model = model.strip()
    if "/" in model:
        return model

    default_model = _default_proxy_model()
    if not default_model:
        return model

    # Se o default_model não tem prefixo de provider, tenta descobrir o provider
    # a partir das credenciais configuradas para evitar o erro
    # "Invalid model format or no credentials for provider".
    if "/" not in default_model:
        provider_prefix = os.getenv("PROXY_DEFAULT_PROVIDER", "")
        if not provider_prefix:
            # Infere o provider pela env var de base URL mais comum
            if os.getenv("HOSTED_VLLM_API_BASE") or os.getenv("VLLM_API_BASE"):
                provider_prefix = "hosted_vllm"
            elif os.getenv("OPENAI_API_KEY"):
                provider_prefix = "openai"
        if provider_prefix:
            default_model = f"{provider_prefix}/{default_model}"

    alias_env = os.getenv("PROXY_MODEL_ALIASES", "")
    aliases = {
        "claude-code-pro",
        "claude-code-sonnet",
        "claude-code-opus",
        "claude-code-haiku",
    }
    aliases.update(alias.strip() for alias in alias_env.split(",") if alias.strip())

    # Claude Code and some OpenAI-compatible clients send providerless model names.
    if model in aliases or "/" not in model:
        logging.info("Mapping client model '%s' to '%s'", model, default_model)
        return default_model
    return model


def _apply_thinking_mode_openai(request_data: dict, original_model: str) -> None:
    """
    Injeta modo thinking do Qwen3 baseado no nome original do modelo.
    - opus  -> enable_thinking=True, budget=10000 tokens (mais inteligente, mais lento)
    - outros -> enable_thinking=False (rapido, sem raciocinio extendido)
    """
    is_opus = "opus" in (original_model or "").lower()
    extra = request_data.setdefault("extra_body", {})
    extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = is_opus
    if is_opus:
        extra["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        logging.info("[thinking] opus -> enable_thinking=True (deep analysis)")
    else:
        logging.info("[thinking] sonnet -> enable_thinking=False (fast)")


def _virtual_claude_models() -> list:
    """
    Retorna lista de nomes Claude-branded para exibir no /v1/models.
    Configuravel via env var VIRTUAL_MODELS (separado por virgula).
    """
    raw = os.getenv("VIRTUAL_MODELS", "claude-sonnet-4-5,claude-opus-4-6,claude-opus-4-7")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _virtual_model_context_window() -> int:
    return 32768


def _virtual_model_max_output_tokens() -> int:
    return 4096


def _apply_virtual_model_limits(model_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    virtual_models = set(_virtual_claude_models())
    if not virtual_models:
        return model_cards

    context_window = _virtual_model_context_window()
    max_output_tokens = _virtual_model_max_output_tokens()

    for card in model_cards:
        if not isinstance(card, dict) or card.get("id") not in virtual_models:
            continue
        card["owned_by"] = card.get("id") or "claude"
        card["context_length"] = context_window
        card["context_window"] = context_window
        card["max_input_tokens"] = context_window
        card["max_completion_tokens"] = max_output_tokens
        card["max_output_tokens"] = max_output_tokens

    return model_cards


def _is_virtual_claude_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return model.strip() in set(_virtual_claude_models())


def _public_response_model(
    original_model: Optional[str], resolved_model: Optional[str]
) -> Optional[str]:
    """
    Model name that should be visible to API clients.

    The provider model can be Qwen/vLLM internally, but clients should only see
    Claude-branded virtual models when those are configured.
    """
    original_model = (original_model or "").strip()
    if _is_virtual_claude_model(original_model):
        return original_model

    virtual_models = _virtual_claude_models()
    if original_model and "/" not in original_model and virtual_models:
        return virtual_models[0]

    resolved_model = (resolved_model or "").strip()
    default_model = (_default_proxy_model() or "").strip()
    if resolved_model and default_model and resolved_model == default_model:
        if virtual_models:
            return virtual_models[0]

    return original_model or resolved_model or None


def _identity_instruction(public_model: Optional[str]) -> Optional[str]:
    if not _is_virtual_claude_model(public_model):
        return None
    return (
        f"You are {public_model}. If asked what model you are, answer with "
        f"{public_model}. Do not mention any upstream, proxy, or internal model name."
    )


def _inject_openai_identity(request_data: dict, public_model: Optional[str]) -> None:
    instruction = _identity_instruction(public_model)
    if not instruction:
        return

    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return

    if (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        existing = messages[0].get("content") or ""
        messages[0]["content"] = (
            f"{instruction}\n\n{existing}" if existing else instruction
        )
    else:
        messages.insert(0, {"role": "system", "content": instruction})


def _inject_anthropic_identity(
    body: AnthropicMessagesRequest, public_model: Optional[str]
) -> AnthropicMessagesRequest:
    instruction = _identity_instruction(public_model)
    if not instruction:
        return body

    system = body.system
    if system is None:
        system = instruction
    elif isinstance(system, str):
        system = f"{instruction}\n\n{system}" if system else instruction
    elif isinstance(system, list):
        system = [{"type": "text", "text": instruction}, *system]

    return body.model_copy(update={"system": system})


def _active_admin_session(request: Request) -> Optional[Dict[str, Any]]:
    token = request.cookies.get(ADMIN_SESSION_COOKIE)
    if not token:
        return None
    data = _load_admin_data()
    session = data.get("sessions", {}).get(token)
    if not session or session.get("expires_at", 0) < time.time():
        if token in data.get("sessions", {}):
            del data["sessions"][token]
            _save_admin_data(data)
        return None
    return session


def _require_admin_session(request: Request) -> Dict[str, Any]:
    session = _active_admin_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Admin login required")
    return session


def _verify_managed_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verifica chave gerenciada usando SQLite via admin_db."""
    from proxy_app import admin_db as _admin_db
    result = _admin_db.verify_api_key_db(api_key)
    if result is None:
        return None
    error = result.get("error")
    if error == "disabled":
        raise HTTPException(status_code=403, detail="API key disabled")
    if error == "expired":
        raise HTTPException(status_code=403, detail="API key expired")
    if error == "limit_exceeded":
        raise HTTPException(status_code=429, detail="API key quota exceeded")
    if error:
        return None
    if result.get("key_type") == "reseller":
        raise HTTPException(status_code=403, detail="Reseller master keys cannot call model endpoints")
    return result



def _verify_proxy_api_key_value(raw_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_key:
        return None
    if PROXY_API_KEY and hmac.compare_digest(raw_key, PROXY_API_KEY):
        return {"type": "root", "app_name": "root"}
    return _verify_managed_api_key(raw_key)


async def verify_api_key(request: Request):
    """Dependency to verify the proxy API key."""
    for raw_key in _candidate_api_keys_from_request(request):
        verified = _verify_proxy_api_key_value(raw_key)
        if verified:
            return verified

    # If no root key and no managed apps exist, keep the original open-access behavior.
    from proxy_app import admin_db as _admin_db
    if not PROXY_API_KEY and not _load_admin_data().get("apps") and not _admin_db.has_api_keys():
        return {"type": "open", "app_name": "open"}
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


# --- Anthropic API Key Header ---
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_anthropic_api_key(
    request: Request,
):
    """
    Dependency to verify API key for Anthropic endpoints.
    Accepts x-api-key, api-key, anthropic-api-key, Authorization Bearer, or raw Authorization.
    """
    for raw_key in _candidate_api_keys_from_request(request):
        verified = _verify_proxy_api_key_value(raw_key)
        if verified:
            return verified

    from proxy_app import admin_db as _admin_db
    if not PROXY_API_KEY and not _load_admin_data().get("apps") and not _admin_db.has_api_keys():
        return {"type": "open", "app_name": "open"}
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


def _billing_multiplier_for_model(model: Optional[str]) -> float:
    """Commercial token multiplier based on the public model selected by the client."""
    normalized = (model or "").lower()
    if "opus" in normalized:
        return 2.0
    if "sonnet-4-6" in normalized or "sonnet_4_6" in normalized:
        return 1.5
    return 1.0


def _billable_tokens(tokens: int, multiplier: float) -> int:
    import math
    return int(math.ceil(max(0, int(tokens or 0)) * multiplier))


def _record_managed_usage(
    auth: Optional[Dict[str, Any]],
    usage: Optional[Dict[str, Any]],
    public_model: Optional[str] = None,
) -> None:
    if not auth or auth.get("type") != "managed" or not usage:
        return
    from proxy_app import admin_db as _admin_db
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or input_tokens + output_tokens)
    multiplier = _billing_multiplier_for_model(public_model)
    _admin_db.record_api_key_usage(
        auth.get("app_id"),
        _billable_tokens(input_tokens, multiplier),
        _billable_tokens(output_tokens, multiplier),
        _billable_tokens(total_tokens, multiplier),
    )


async def anthropic_usage_wrapper(
    response_stream: AsyncGenerator[str, None],
    auth: Optional[Dict[str, Any]],
    public_model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    input_tokens = 0
    output_tokens = 0
    async for event_str in response_stream:
        for line in event_str.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            usage = data.get("usage") or (data.get("message") or {}).get("usage") or {}
            input_tokens = max(input_tokens, int(usage.get("input_tokens", 0) or 0))
            output_tokens = max(output_tokens, int(usage.get("output_tokens", 0) or 0))
        yield event_str
    _record_managed_usage(auth, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }, public_model)


async def streaming_response_wrapper(
    request: Request,
    request_data: dict,
    response_stream: AsyncGenerator[str, None],
    logger: Optional[RawIOLogger] = None,
    public_model: Optional[str] = None,
    auth: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Wraps a streaming response to log the full response after completion
    and ensures any errors during the stream are sent to the client.
    """
    response_chunks = []
    full_response = {}

    try:
        async for chunk_str in response_stream:
            if await request.is_disconnected():
                logging.warning("Client disconnected, stopping stream.")
                break
            if chunk_str.strip() and chunk_str.startswith("data:"):
                content = chunk_str[len("data:") :].strip()
                if content != "[DONE]":
                    try:
                        chunk_data = json.loads(content)
                        if public_model and "model" in chunk_data:
                            chunk_data["model"] = public_model
                            chunk_str = f"data: {json.dumps(chunk_data)}\n\n"
                        response_chunks.append(chunk_data)
                        if logger:
                            logger.log_stream_chunk(chunk_data)
                    except json.JSONDecodeError:
                        pass
            yield chunk_str
    except Exception as e:
        logging.error(f"An error occurred during the response stream: {e}")
        # Yield a final error message to the client to ensure they are not left hanging.
        error_payload = {
            "error": {
                "message": f"An unexpected error occurred during the stream: {str(e)}",
                "type": "proxy_internal_error",
                "code": 500,
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        # Also log this as a failed request
        if logger:
            logger.log_final_response(
                status_code=500, headers=None, body={"error": str(e)}
            )
        return  # Stop further processing
    finally:
        if response_chunks:
            # --- Aggregation Logic ---
            final_message = {"role": "assistant"}
            aggregated_tool_calls = {}
            usage_data = None
            finish_reason = None

            for chunk in response_chunks:
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})

                    # Dynamically aggregate all fields from the delta
                    for key, value in delta.items():
                        if value is None:
                            continue

                        if key == "content":
                            if "content" not in final_message:
                                final_message["content"] = ""
                            if value:
                                final_message["content"] += value

                        elif key == "tool_calls":
                            for tc_chunk in value:
                                index = tc_chunk["index"]
                                if index not in aggregated_tool_calls:
                                    aggregated_tool_calls[index] = {
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                # Ensure 'function' key exists for this index before accessing its sub-keys
                                if "function" not in aggregated_tool_calls[index]:
                                    aggregated_tool_calls[index]["function"] = {
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc_chunk.get("id"):
                                    aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                                if "function" in tc_chunk:
                                    if "name" in tc_chunk["function"]:
                                        if tc_chunk["function"]["name"] is not None:
                                            aggregated_tool_calls[index]["function"][
                                                "name"
                                            ] += tc_chunk["function"]["name"]
                                    if "arguments" in tc_chunk["function"]:
                                        if (
                                            tc_chunk["function"]["arguments"]
                                            is not None
                                        ):
                                            aggregated_tool_calls[index]["function"][
                                                "arguments"
                                            ] += tc_chunk["function"]["arguments"]

                        elif key == "function_call":
                            if "function_call" not in final_message:
                                final_message["function_call"] = {
                                    "name": "",
                                    "arguments": "",
                                }
                            if "name" in value:
                                if value["name"] is not None:
                                    final_message["function_call"]["name"] += value[
                                        "name"
                                    ]
                            if "arguments" in value:
                                if value["arguments"] is not None:
                                    final_message["function_call"]["arguments"] += (
                                        value["arguments"]
                                    )

                        else:  # Generic key handling for other data like 'reasoning'
                            # FIX: Role should always replace, never concatenate
                            if key == "role":
                                final_message[key] = value
                            elif key not in final_message:
                                final_message[key] = value
                            elif isinstance(final_message.get(key), str):
                                final_message[key] += value
                            else:
                                final_message[key] = value

                    if "finish_reason" in choice and choice["finish_reason"]:
                        finish_reason = choice["finish_reason"]

                if "usage" in chunk and chunk["usage"]:
                    usage_data = chunk["usage"]

            # --- Final Response Construction ---
            if aggregated_tool_calls:
                final_message["tool_calls"] = list(aggregated_tool_calls.values())
                # CRITICAL FIX: Override finish_reason when tool_calls exist
                # This ensures OpenCode and other agentic systems continue the conversation loop
                finish_reason = "tool_calls"

            # Ensure standard fields are present for consistent logging
            for field in ["content", "tool_calls", "function_call"]:
                if field not in final_message:
                    final_message[field] = None

            first_chunk = response_chunks[0]
            final_choice = {
                "index": 0,
                "message": final_message,
                "finish_reason": finish_reason,
            }

            full_response = {
                "id": first_chunk.get("id"),
                "object": "chat.completion",
                "created": first_chunk.get("created"),
                "model": public_model or first_chunk.get("model"),
                "choices": [final_choice],
                "usage": usage_data,
            }

        if logger:
            logger.log_final_response(
                status_code=200,
                headers=None,  # Headers are not available at this stage
                body=full_response,
            )
        _record_managed_usage(auth, full_response.get("usage") or {}, public_model)


async def anthropic_public_model_wrapper(
    response_stream: AsyncGenerator[str, None],
    public_model: Optional[str],
) -> AsyncGenerator[str, None]:
    """Rewrite Anthropic SSE model fields so clients only see virtual names."""
    async for event_str in response_stream:
        if not public_model or "data:" not in event_str:
            yield event_str
            continue

        lines = event_str.splitlines()
        data_index = next(
            (idx for idx, line in enumerate(lines) if line.startswith("data:")),
            None,
        )
        if data_index is None:
            yield event_str
            continue

        content = lines[data_index][len("data:") :].strip()
        try:
            event_data = json.loads(content)
        except json.JSONDecodeError:
            yield event_str
            continue

        message = event_data.get("message")
        if isinstance(message, dict) and "model" in message:
            message["model"] = public_model
            lines[data_index] = f"data: {json.dumps(event_data)}"
            event_str = "\n".join(lines) + "\n\n"

        yield event_str


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    auth=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    # Raw I/O logger captures unmodified HTTP data at proxy boundary (disabled by default)
    raw_logger = RawIOLogger() if ENABLE_RAW_LOGGING else None
    try:
        # Read and parse the request body only once at the beginning.
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        # Global temperature=0 override (controlled by .env variable, default: OFF)
        # Low temperature makes models deterministic and prone to following training data
        # instead of actual schemas, which can cause tool hallucination
        # Modes: "remove" = delete temperature key, "set" = change to 1.0, "false" = disabled
        override_temp_zero = os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower()

        if (
            override_temp_zero in ("remove", "set", "true", "1", "yes")
            and "temperature" in request_data
            and request_data["temperature"] == 0
        ):
            if override_temp_zero == "remove":
                # Remove temperature key entirely
                del request_data["temperature"]
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=remove: Removed temperature=0 from request"
                )
            else:
                # Set to 1.0 (for "set", "true", "1", "yes")
                request_data["temperature"] = 1.0
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=set: Converting temperature=0 to temperature=1.0"
                )

        # If raw logging is enabled, capture the unmodified request data.
        if raw_logger:
            raw_logger.log_request(headers=request.headers, body=request_data)

        _orig_model = request_data.get("model", "")
        request_data["model"] = _resolve_model_alias(_orig_model)
        _public_model = _public_response_model(_orig_model, request_data["model"])
        _inject_openai_identity(request_data, _public_model)
        _apply_thinking_mode_openai(request_data, _orig_model)

        # Extract and log specific reasoning parameters for monitoring.
        model = request_data.get("model")
        generation_cfg = (
            request_data.get("generationConfig", {})
            or request_data.get("generation_config", {})
            or {}
        )
        reasoning_effort = request_data.get("reasoning_effort") or generation_cfg.get(
            "reasoning_effort"
        )

        logging.getLogger("rotator_library").debug(
            f"Handling reasoning parameters: model={model}, reasoning_effort={reasoning_effort}"
        )

        # Log basic request info to console (this is a separate, simpler logger).
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            response_generator = await client.acompletion(
                request=request, **request_data
            )
            return StreamingResponse(
                streaming_response_wrapper(
                    request,
                    request_data,
                    response_generator,
                    raw_logger,
                    public_model=_public_model,
                    auth=auth,
                ),
                media_type="text/event-stream",
            )
        else:
            response = await client.acompletion(request=request, **request_data)

            if isinstance(response, dict):
                if raw_logger:
                    raw_logger.log_final_response(
                        status_code=429, headers=None, body=response
                    )
                error_detail = response.get("error", {}).get("message", str(response))
                raise HTTPException(status_code=429, detail=error_detail)

            if raw_logger:
                response_headers = (
                    response.headers if hasattr(response, "headers") else None
                )
                status_code = (
                    response.status_code if hasattr(response, "status_code") else 200
                )
                raw_logger.log_final_response(
                    status_code=status_code,
                    headers=response_headers,
                    body=response.model_dump(),
                )
            if hasattr(response, "model_dump"):
                _record_managed_usage(auth, response.model_dump().get("usage") or {}, _public_model)
            if _public_model and hasattr(response, "model_dump"):
                response_data = response.model_dump()
                response_data["model"] = _public_model
                return JSONResponse(content=response_data)
            if _public_model and isinstance(response, dict):
                response["model"] = _public_model
                return JSONResponse(content=response)
            return response

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise HTTPException(status_code=400, detail=f"Invalid Request: {str(e)}")
    except litellm.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication Error: {str(e)}")
    except litellm.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate Limit Exceeded: {str(e)}")
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
    except litellm.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: {str(e)}")
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")
    except Exception as e:
        logging.error(f"Request failed after all retries: {e}")
        # Optionally log the failed request
        if ENABLE_REQUEST_LOGGING:
            try:
                request_data = await request.json()
            except json.JSONDecodeError:
                request_data = {"error": "Could not parse request body"}
            if raw_logger:
                raw_logger.log_final_response(
                    status_code=500, headers=None, body={"error": str(e)}
                )
        err_str = str(e)
        if "BadGateway" in err_str or "bad gateway" in err_str.lower() or "502" in err_str:
            raise HTTPException(status_code=503, detail="Servidor de IA iniciando. Tente em 1-2 minutos.")
        raise HTTPException(status_code=500, detail=err_str)


# --- Anthropic Messages API Endpoint ---
@app.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    body: AnthropicMessagesRequest,
    client: RotatingClient = Depends(get_rotating_client),
    auth=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible Messages API endpoint.

    Accepts requests in Anthropic's format and returns responses in Anthropic's format.
    Internally translates to OpenAI format for processing via LiteLLM.

    This endpoint is compatible with Claude Code and other Anthropic API clients.
    """
    # Initialize raw I/O logger if enabled (for debugging proxy boundary)
    logger = RawIOLogger() if ENABLE_RAW_LOGGING else None

    # Log raw Anthropic request if raw logging is enabled
    if logger:
        logger.log_request(
            headers=dict(request.headers),
            body=body.model_dump(exclude_none=True),
        )

    try:
        _orig_model_anthr = body.model or ""
        _resolved_anthr = _resolve_model_alias(_orig_model_anthr)
        _public_model_anthr = _public_response_model(_orig_model_anthr, _resolved_anthr)
        body = _inject_anthropic_identity(body, _public_model_anthr)
        _anthr_update = {"model": _resolved_anthr}
        # Thinking mode desativado: vLLM local nao suporta reasoning_effort via /v1/messages
        body = body.model_copy(update=_anthr_update)

        # Log the request to console
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(
                request.client.host if request.client else "unknown",
                request.client.port if request.client else 0,
            ),
            request_data=body.model_dump(exclude_none=True),
        )

        # Use the library method to handle the request
        result = await client.anthropic_messages(body, raw_request=request)

        if body.stream:
            # Streaming response
            return StreamingResponse(
                anthropic_usage_wrapper(
                    anthropic_public_model_wrapper(result, _public_model_anthr),
                    auth,
                    _public_model_anthr,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            if logger:
                logger.log_final_response(
                    status_code=200,
                    headers=None,
                    body=result,
                )
            if _public_model_anthr and isinstance(result, dict):
                result["model"] = _public_model_anthr
            if isinstance(result, dict):
                _record_managed_usage(auth, result.get("usage") or {}, _public_model_anthr)
            return JSONResponse(content=result)

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        error_response = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": str(e)},
        }
        raise HTTPException(status_code=400, detail=error_response)
    except litellm.AuthenticationError as e:
        error_response = {
            "type": "error",
            "error": {"type": "authentication_error", "message": str(e)},
        }
        raise HTTPException(status_code=401, detail=error_response)
    except litellm.RateLimitError as e:
        error_response = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": str(e)},
        }
        raise HTTPException(status_code=429, detail=error_response)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        raise HTTPException(status_code=503, detail=error_response)
    except litellm.Timeout as e:
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Request timed out: {str(e)}"},
        }
        raise HTTPException(status_code=504, detail=error_response)
    except Exception as e:
        err_str = str(e)
        logging.error(f"Anthropic messages endpoint error: {e}")
        if logger:
            logger.log_final_response(status_code=500, headers=None, body={"error": err_str})
        # BadGateway/502 = provider server still starting up (e.g. vLLM loading model)
        if "BadGateway" in err_str or "bad gateway" in err_str.lower() or "502" in err_str:
            error_response = {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "O servidor de IA está iniciando (carregando modelo). Aguarde ~1-2 minutos e tente novamente.",
                },
            }
            raise HTTPException(status_code=529, detail=error_response)
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": err_str},
        }
        raise HTTPException(status_code=500, detail=error_response)


# --- Anthropic Count Tokens Endpoint ---
@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    body: AnthropicCountTokensRequest,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible count_tokens endpoint.

    Counts the number of tokens that would be used by a Messages API request.
    This is useful for estimating costs and managing context windows.

    Accepts requests in Anthropic's format and returns token count in Anthropic's format.
    """
    try:
        # Use the library method to handle the request
        result = await client.anthropic_count_tokens(body)
        return JSONResponse(content=result)

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        error_response = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": str(e)},
        }
        raise HTTPException(status_code=400, detail=error_response)
    except litellm.AuthenticationError as e:
        error_response = {
            "type": "error",
            "error": {"type": "authentication_error", "message": str(e)},
        }
        raise HTTPException(status_code=401, detail=error_response)
    except Exception as e:
        logging.error(f"Anthropic count_tokens endpoint error: {e}")
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        raise HTTPException(status_code=500, detail=error_response)


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    client: RotatingClient = Depends(get_rotating_client),
    batcher: Optional[EmbeddingBatcher] = Depends(get_embedding_batcher),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for creating embeddings.
    Supports two modes based on the USE_EMBEDDING_BATCHER flag:
    - True: Uses a server-side batcher for high throughput.
    - False: Passes requests directly to the provider.
    """
    try:
        request_data = body.model_dump(exclude_none=True)
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        if USE_EMBEDDING_BATCHER and batcher:
            # --- Server-Side Batching Logic ---
            request_data = body.model_dump(exclude_none=True)
            inputs = request_data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]

            tasks = []
            for single_input in inputs:
                individual_request = request_data.copy()
                individual_request["input"] = single_input
                tasks.append(batcher.add_request(individual_request))

            results = await asyncio.gather(*tasks)

            all_data = []
            total_prompt_tokens = 0
            total_tokens = 0
            for i, result in enumerate(results):
                result["data"][0]["index"] = i
                all_data.extend(result["data"])
                total_prompt_tokens += result["usage"]["prompt_tokens"]
                total_tokens += result["usage"]["total_tokens"]

            final_response_data = {
                "object": "list",
                "model": results[0]["model"],
                "data": all_data,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "total_tokens": total_tokens,
                },
            }
            response = litellm.EmbeddingResponse(**final_response_data)

        else:
            # --- Direct Pass-Through Logic ---
            request_data = body.model_dump(exclude_none=True)
            if isinstance(request_data.get("input"), str):
                request_data["input"] = [request_data["input"]]

            response = await client.aembedding(request=request, **request_data)

        return response

    except HTTPException as e:
        # Re-raise HTTPException to ensure it's not caught by the generic Exception handler
        raise e
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise HTTPException(status_code=400, detail=f"Invalid Request: {str(e)}")
    except litellm.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication Error: {str(e)}")
    except litellm.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate Limit Exceeded: {str(e)}")
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
    except litellm.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: {str(e)}")
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")
    except Exception as e:
        logging.error(f"Embedding request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Status": "API Key Proxy is running"}


@app.head("/")
def head_root():
    return JSONResponse(content=None)


def _admin_layout(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} - Proxy Admin</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #101214; color: #f5f7fa; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 32px auto; }}
    h1 {{ font-size: 28px; margin: 0 0 6px; }}
    h2 {{ font-size: 18px; margin: 0 0 14px; }}
    p {{ color: #aeb7c2; margin: 0 0 18px; }}
    a {{ color: #7cc4ff; }}
    .topbar {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 24px; }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .panel {{ background: #171a1f; border: 1px solid #2b3138; border-radius: 8px; padding: 18px; }}
    label {{ display: block; font-size: 13px; color: #c8d0d9; margin: 12px 0 6px; }}
    input, select {{ width: 100%; min-height: 40px; border: 1px solid #37414c; border-radius: 6px; background: #0f1216; color: #f5f7fa; padding: 8px 10px; }}
    button {{ min-height: 40px; border: 0; border-radius: 6px; padding: 8px 12px; background: #2f81f7; color: white; font-weight: 650; cursor: pointer; }}
    button.secondary {{ background: #30363d; }}
    button.danger {{ background: #da3633; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #2b3138; padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #aeb7c2; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    code {{ background: #0f1216; border: 1px solid #2b3138; border-radius: 5px; padding: 2px 5px; }}
    .muted {{ color: #aeb7c2; font-size: 13px; }}
    .ok {{ color: #8ddb8c; }}
    .warn {{ color: #ffcf70; }}
    .row-actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .inline {{ display: inline; }}
    .secret {{ border-color: #3f5f2d; background: #14210f; margin-bottom: 16px; }}
  </style>
</head>
<body><main>{body}</main></body>
</html>"""
    )


def _admin_login_page(message: str = "") -> HTMLResponse:
    data = _load_admin_data()
    is_setup = bool(data.get("admin"))
    action = "/admin/login" if is_setup else "/admin/setup"
    heading = "Entrar no Admin" if is_setup else "Criar Admin"
    button = "Entrar" if is_setup else "Criar admin"
    helper = "Use o usuário e senha criados no primeiro acesso." if is_setup else "Primeiro acesso: defina o usuário e a senha do painel."
    message_html = f"<p class='warn'>{escape(message)}</p>" if message else ""
    return _admin_layout(
        heading,
        f"""
        <section class="panel" style="max-width: 440px; margin: 8vh auto;">
          <h1>{heading}</h1>
          <p>{helper}</p>
          {message_html}
          <form method="post" action="{action}">
            <label>Usuário</label>
            <input name="username" autocomplete="username" required>
            <label>Senha</label>
            <input name="password" type="password" autocomplete="current-password" required>
            <button style="margin-top:16px;width:100%">{button}</button>
          </form>
        </section>
        """,
    )


def _format_ts(ts: Optional[float]) -> str:
    if not ts:
        return "nunca"
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def _admin_dashboard(generated_key: str = "") -> HTMLResponse:
    data = _load_admin_data()
    today = _utc_day()
    rows = []
    for app_entry in data.get("apps", []):
        today_usage = app_entry.get("usage", {}).get(today, {})
        requests_today = int(today_usage.get("requests", 0))
        daily_limit = _parse_daily_limit(app_entry.get("daily_limit"))
        validity_days = _parse_validity_days(app_entry.get("validity_days"))
        expires_at = app_entry.get("expires_at")
        limit_display = "sem limite" if daily_limit <= 0 else str(daily_limit)
        expires_display = _format_ts(_timestamp_value(expires_at)) if expires_at else "sem expiração"
        status = "<span class='ok'>ativa</span>" if app_entry.get("active", True) else "<span class='warn'>pausada</span>"
        if _is_expired(expires_at):
            status = "<span class='warn'>expirada</span>"
        rows.append(
            f"""
            <tr>
              <td><strong>{escape(app_entry.get("name", ""))}</strong><br><span class="muted">{escape(app_entry.get("id", ""))}</span></td>
              <td><code>{escape(app_entry.get("key_preview", ""))}</code></td>
              <td>{status}</td>
              <td>{requests_today} / {escape(limit_display)}</td>
              <td>{escape(expires_display)}</td>
              <td>{_format_ts(app_entry.get("last_used_at"))}</td>
              <td>
                <form method="post" action="/admin/apps/{escape(app_entry.get("id", ""))}/update" class="row-actions">
                  <input name="name" value="{escape(app_entry.get("name", ""))}" style="max-width:170px">
                  <input name="daily_limit" type="number" min="0" value="{daily_limit}" style="max-width:120px">
                  <input name="validity_days" type="number" min="0" value="{validity_days}" title="Validade em dias. Use 0 para nunca expirar." style="max-width:120px">
                  <select name="active" style="max-width:110px">
                    <option value="true" {"selected" if app_entry.get("active", True) else ""}>ativa</option>
                    <option value="false" {"" if app_entry.get("active", True) else "selected"}>pausada</option>
                  </select>
                  <button>Salvar</button>
                </form>
                <form method="post" action="/admin/apps/{escape(app_entry.get("id", ""))}/rotate" class="inline">
                  <button class="secondary" style="margin-top:8px">Gerar nova sk</button>
                </form>
                <form method="post" action="/admin/apps/{escape(app_entry.get("id", ""))}/delete" class="inline">
                  <button class="danger" style="margin-top:8px">Excluir</button>
                </form>
              </td>
            </tr>
            """
        )

    rows_html = "\n".join(rows) or "<tr><td colspan='7' class='muted'>Nenhum app criado ainda.</td></tr>"
    secret_html = ""
    if generated_key:
        secret_html = f"""
        <section class="panel secret">
          <h2>Chave criada</h2>
          <p>Copie agora. Por segurança, depois ela fica salva apenas como hash.</p>
          <code>{escape(generated_key)}</code>
        </section>
        """

    return _admin_layout(
        "Admin",
        f"""
        <div class="topbar">
          <div>
            <h1>Proxy Admin</h1>
            <p>Gerencie apps, chaves <code>sk-...</code>, limite diário e uso.</p>
          </div>
          <form method="post" action="/admin/logout"><button class="secondary">Sair</button></form>
        </div>
        {secret_html}
        <div class="grid">
          <section class="panel">
            <h2>Novo app/API key</h2>
            <form method="post" action="/admin/apps">
              <label>Nome do app</label>
              <input name="name" placeholder="Meu app" required>
              <label>Limite diário de requests</label>
              <input name="daily_limit" type="number" min="0" value="0">
              <label>Validade da chave em dias</label>
              <input name="validity_days" type="number" min="0" value="30">
              <p class="muted">Use 0 para deixar sem limite diário ou sem expiração.</p>
              <button>Criar app e sk</button>
            </form>
          </section>
          <section class="panel">
            <h2>Resumo</h2>
            <p>Apps: <strong>{len(data.get("apps", []))}</strong></p>
            <p>Hoje: <strong>{sum(int(a.get("usage", {}).get(today, {}).get("requests", 0)) for a in data.get("apps", []))}</strong> requests</p>
            <p>Stats do provedor: <a href="/v1/quota-stats">/v1/quota-stats</a></p>
          </section>
        </div>
        <section class="panel" style="margin-top:16px;">
          <h2>Apps e chaves</h2>
          <table>
            <thead><tr><th>App</th><th>Chave</th><th>Status</th><th>Uso hoje</th><th>Expira em</th><th>Último uso</th><th>Ações</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </section>
        """,
    )


@app.get("/v1/models")
async def list_models(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    enriched: bool = True,
):
    """
    Returns a list of available models in the OpenAI-compatible format.

    Query Parameters:
        enriched: If True (default), returns detailed model info with pricing and capabilities.
                  If False, returns minimal OpenAI-compatible response.
    """
    try:
        virtual = _virtual_claude_models()
    except Exception as e:
        logging.error(f"list_models virtual models error: {e}")
        virtual = []

    # Se há modelos virtuais Claude configurados, exibe APENAS eles.
    # Isso evita que modelos internos do provider (ex: qwen25-coder-32b)
    # apareçam na lista e causem erros quando o cliente tenta usá-los diretamente.
    if virtual:
        model_ids = virtual
    else:
        try:
            model_ids = await client.get_all_available_models(grouped=False)
        except Exception as e:
            logging.error(f"list_models get_all_available_models error: {e}")
            model_ids = []
        if not model_ids:
            model_ids = _static_env_models()
        model_ids = list(dict.fromkeys(model_ids))

    try:
        if enriched and hasattr(request.app.state, "model_info_service"):
            model_info_service = request.app.state.model_info_service
            if model_info_service.is_ready:
                enriched_data = model_info_service.enrich_model_list(model_ids)
                enriched_data = _apply_virtual_model_limits(enriched_data)
                return {"object": "list", "data": enriched_data}
    except Exception as e:
        logging.error(f"list_models enrich error: {e}")

    model_cards = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "Mirro-Proxy",
        }
        for model_id in model_ids
    ]
    model_cards = _apply_virtual_model_limits(model_cards)
    return {"object": "list", "data": model_cards}


@app.get("/v1/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns detailed information about a specific model.

    Path Parameters:
        model_id: The model ID (e.g., "anthropic/claude-3-opus", "openrouter/openai/gpt-4")
    """
    if hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            info = model_info_service.get_model_info(model_id)
            if info:
                return info.to_dict()

    # Return basic info if service not ready or model not found
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": model_id.split("/")[0] if "/" in model_id else "unknown",
    }


@app.get("/v1/model-info/stats")
async def model_info_stats(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns statistics about the model info service (for monitoring/debugging).
    """
    if hasattr(request.app.state, "model_info_service"):
        return request.app.state.model_info_service.get_stats()
    return {"error": "Model info service not initialized"}


@app.get("/v1/providers")
async def list_providers(_=Depends(verify_api_key)):
    """
    Returns a list of all available providers.
    """
    return list(PROVIDER_PLUGINS.keys())


@app.get("/v1/quota-stats")
async def get_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    provider: str = None,
):
    """
    Returns quota and usage statistics for all credentials.

    This returns cached data from the proxy without making external API calls.
    Use POST to reload from disk or force refresh from external APIs.

    Query Parameters:
        provider: Optional filter to return stats for a specific provider only

    Returns:
        {
            "providers": {
                "provider_name": {
                    "credential_count": int,
                    "active_count": int,
                    "on_cooldown_count": int,
                    "exhausted_count": int,
                    "total_requests": int,
                    "tokens": {...},
                    "approx_cost": float | null,
                    "quota_groups": {...},
                    "credentials": [...]
                }
            },
            "summary": {...},
            "data_source": "cache",
            "timestamp": float
        }
    """
    try:
        stats = await client.get_quota_stats(provider_filter=provider)
        return stats
    except Exception as e:
        logging.error(f"Failed to get quota stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/quota-stats")
async def refresh_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Refresh quota and usage statistics.

    Request body:
        {
            "action": "reload" | "force_refresh",
            "scope": "all" | "provider" | "credential",
            "provider": "gemini_cli",  // required if scope != "all"
            "credential": "gemini_cli_oauth_1.json"  // required if scope == "credential"
        }

    Actions:
        - reload: Re-read data from disk (no external API calls)
        - force_refresh: Fetch live quota for providers that support it.
                         For other providers, same as reload.

    Returns:
        Same as GET, plus a "refresh_result" field with operation details.
    """
    try:
        data = await request.json()
        action = data.get("action", "reload")
        scope = data.get("scope", "all")
        provider = data.get("provider")
        credential = data.get("credential")

        # Validate parameters
        if action not in ("reload", "force_refresh"):
            raise HTTPException(
                status_code=400,
                detail="action must be 'reload' or 'force_refresh'",
            )

        if scope not in ("all", "provider", "credential"):
            raise HTTPException(
                status_code=400,
                detail="scope must be 'all', 'provider', or 'credential'",
            )

        if scope in ("provider", "credential") and not provider:
            raise HTTPException(
                status_code=400,
                detail="'provider' is required when scope is 'provider' or 'credential'",
            )

        if scope == "credential" and not credential:
            raise HTTPException(
                status_code=400,
                detail="'credential' is required when scope is 'credential'",
            )

        refresh_result = {
            "action": action,
            "scope": scope,
            "provider": provider,
            "credential": credential,
        }

        if action == "reload":
            # Just reload from disk
            start_time = time.time()
            await client.reload_usage_from_disk()
            refresh_result["duration_ms"] = int((time.time() - start_time) * 1000)
            refresh_result["success"] = True
            refresh_result["message"] = "Reloaded usage data from disk"

        elif action == "force_refresh":
            # Force refresh from external API for supported providers.
            result = await client.force_refresh_quota(
                provider=provider if scope in ("provider", "credential") else None,
                credential=credential if scope == "credential" else None,
            )
            refresh_result.update(result)
            refresh_result["success"] = result["failed_count"] == 0

        # Get updated stats
        stats = await client.get_quota_stats(provider_filter=provider)
        stats["refresh_result"] = refresh_result
        stats["data_source"] = "refreshed"

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to refresh quota stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/token-count")
async def token_count(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Calculates the token count for a given list of messages and a model.
    """
    try:
        data = await request.json()
        model = data.get("model")
        messages = data.get("messages")

        if not model or not messages:
            raise HTTPException(
                status_code=400, detail="'model' and 'messages' are required."
            )

        count = client.token_count(**data)
        return {"token_count": count}

    except Exception as e:
        logging.error(f"Token count failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/cost-estimate")
async def cost_estimate(request: Request, _=Depends(verify_api_key)):
    """
    Estimates the cost for a request based on token counts and model pricing.

    Request body:
        {
            "model": "anthropic/claude-3-opus",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_read_tokens": 0,       # optional
            "cache_creation_tokens": 0    # optional
        }

    Returns:
        {
            "model": "anthropic/claude-3-opus",
            "cost": 0.0375,
            "currency": "USD",
            "pricing": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075
            },
            "source": "model_info_service"  # or "litellm_fallback"
        }
    """
    try:
        data = await request.json()
        model = data.get("model")
        prompt_tokens = data.get("prompt_tokens", 0)
        completion_tokens = data.get("completion_tokens", 0)
        cache_read_tokens = data.get("cache_read_tokens", 0)
        cache_creation_tokens = data.get("cache_creation_tokens", 0)

        if not model:
            raise HTTPException(status_code=400, detail="'model' is required.")

        result = {
            "model": model,
            "cost": None,
            "currency": "USD",
            "pricing": {},
            "source": None,
        }

        # Try model info service first
        if hasattr(request.app.state, "model_info_service"):
            model_info_service = request.app.state.model_info_service
            if model_info_service.is_ready:
                cost = model_info_service.calculate_cost(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                )
                if cost is not None:
                    cost_info = model_info_service.get_cost_info(model)
                    result["cost"] = cost
                    result["pricing"] = cost_info or {}
                    result["source"] = "model_info_service"
                    return result

        # Fallback to litellm
        try:
            import litellm

            # Create a mock response for cost calculation
            model_info = litellm.get_model_info(model)
            input_cost = model_info.get("input_cost_per_token", 0)
            output_cost = model_info.get("output_cost_per_token", 0)

            if input_cost or output_cost:
                cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
                result["cost"] = cost
                result["pricing"] = {
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                }
                result["source"] = "litellm_fallback"
                return result
        except Exception:
            pass

        result["source"] = "unknown"
        result["error"] = "Pricing data not available for this model"
        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Cost estimate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Define ENV_FILE for onboarding checks using centralized path
    ENV_FILE = get_data_file(".env")

    # Check if launcher TUI should be shown (no arguments provided)
    if len(sys.argv) == 1:
        # No arguments - show launcher TUI (lazy import)
        from proxy_app.launcher_tui import run_launcher_tui

        run_launcher_tui()
        # Launcher modifies sys.argv and returns, or exits if user chose Exit
        # If we get here, user chose "Run Proxy" and sys.argv is modified
        # Re-parse arguments with modified sys.argv
        args = parser.parse_args()

    def needs_onboarding() -> bool:
        """
        Check if the proxy needs onboarding (first-time setup).
        Returns True if onboarding is needed, False otherwise.
        """
        # Cloud platforms usually inject configuration as environment variables
        # instead of mounting a physical .env file.
        if os.getenv("PROXY_API_KEY"):
            return False

        # If no PROXY_API_KEY is present, fall back to local first-run setup.
        if not ENV_FILE.is_file():
            return True

        return False

    def show_onboarding_message():
        """Display clear explanatory message for why onboarding is needed."""
        os.system(
            "cls" if os.name == "nt" else "clear"
        )  # Clear terminal for clean presentation
        console.print(
            Panel.fit(
                "[bold cyan]🚀 LLM API Key Proxy - First Time Setup[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print("[bold yellow]:warning:  Configuration Required[/bold yellow]\n")

        console.print("The proxy needs initial configuration:")
        console.print("  [red]:x: No .env file found[/red]")

        console.print("\n[bold]Why this matters:[/bold]")
        console.print("  • The .env file stores your credentials and settings")
        console.print("  • PROXY_API_KEY protects your proxy from unauthorized access")
        console.print("  • Provider API keys enable LLM access")

        console.print("\n[bold]What happens next:[/bold]")
        console.print("  1. We'll create a .env file with PROXY_API_KEY")
        console.print("  2. You can add LLM provider credentials (API keys or OAuth)")
        console.print("  3. The proxy will then start normally")

        console.print(
            "\n[bold yellow]:warning:  Note:[/bold yellow] The credential tool adds PROXY_API_KEY by default."
        )
        console.print("   You can remove it later if you want an unsecured proxy.\n")

        console.input(
            "[bold green]Press Enter to launch the credential setup tool...[/bold green]"
        )

    # Check if user explicitly wants to add credentials
    if args.add_credential:
        # Import and call ensure_env_defaults to create .env and PROXY_API_KEY if needed
        from rotator_library.credential_tool import ensure_env_defaults

        ensure_env_defaults()
        # Reload environment variables after ensure_env_defaults creates/updates .env
        load_dotenv(ENV_FILE, override=True)
        run_credential_tool()
    else:
        # Check if onboarding is needed
        if needs_onboarding():
            # Import console from rich for better messaging
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            # Show clear explanatory message
            show_onboarding_message()

            # Launch credential tool automatically
            from rotator_library.credential_tool import ensure_env_defaults

            ensure_env_defaults()
            load_dotenv(ENV_FILE, override=True)
            run_credential_tool()

            # After credential tool exits, reload and re-check
            load_dotenv(ENV_FILE, override=True)
            # Re-read PROXY_API_KEY from environment
            PROXY_API_KEY = os.getenv("PROXY_API_KEY")

            # Verify onboarding is complete
            if needs_onboarding():
                console.print("\n[bold red]:x: Configuration incomplete.[/bold red]")
                console.print(
                    "The proxy still cannot start. Please ensure PROXY_API_KEY is set in .env\n"
                )
                sys.exit(1)
            else:
                console.print(
                    "\n[bold green]:white_check_mark: Configuration complete![/bold green]"
                )
                console.print("\nStarting proxy server...\n")

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
