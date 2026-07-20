"""Orchestrate a local llama.cpp (Qwen3-VL) server and the batch pipeline.

Starts ``llama-server`` if it is not already running, waits until its
OpenAI-compatible ``/v1/models`` endpoint responds, runs the image batch
extraction (``main.main()``), and finally shuts down the server it started.

Configuration is via environment variables so model paths stay out of source:

    LLAMA_SERVER_BIN   Path to the llama-server binary (default: "llama-server")
    LLAMA_MODEL        Path to the Qwen3-VL .gguf model file (required to start)
    LLAMA_MMPROJ       Path to the vision projector (mmproj) .gguf file (optional)
    LLAMA_HOST         Host to bind/probe (default: "127.0.0.1")
    LLAMA_PORT         Port to bind/probe (default: "8080")
    LLAMA_EXTRA_ARGS   Extra args appended to the server command (optional)
    LLAMA_STARTUP_TIMEOUT  Seconds to wait for readiness (optional; waits
                           indefinitely if unset so first-run downloads finish)

If a server is already reachable at the host/port, this script reuses it and
does not start or stop anything.

Usage:
    uv run python run_pipeline.py               # start server, run batch, stop
    uv run python run_pipeline.py --serve-only   # start server and keep it up
    uv run poe pipeline
    uv run poe serve
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _base_url() -> str:
    host = os.environ.get("LLAMA_HOST", "127.0.0.1")
    port = os.environ.get("LLAMA_PORT", "8080")
    return f"http://{host}:{port}"


def _is_server_ready() -> bool:
    """Return True only when the server is fully ready to serve requests.

    Uses llama-server's /health endpoint, which returns 503 while the model is
    still loading and 200 once it's ready. (The /v1/models endpoint responds
    200 as soon as the process is up, even before the model has loaded, which
    would let requests fire too early and get 503s.)
    """
    url = f"{_base_url()}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def _running_model_id() -> "str | None":
    """Return the model id served by the running server, or None if unknown."""
    url = f"{_base_url()}/v1/models"
    try:
        import json
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())
        models = data.get("data") or data.get("models") or []
        if models:
            return models[0].get("id") or models[0].get("model")
    except Exception:
        pass
    return None


def _warn_if_model_mismatch() -> None:
    """Warn when a reused server isn't serving the configured model."""
    requested = os.environ.get("LLAMA_HF_REPO") or os.environ.get("LLAMA_MODEL")
    running = _running_model_id()
    if requested and running and requested not in running and running not in requested:
        print(
            f"[run_pipeline] WARNING: reusing a running server that serves "
            f"'{running}', but you requested '{requested}'. Stop the running "
            f"server first if you want the configured model."
        )


def _build_server_command() -> list[str]:
    """Assemble the llama-server command from environment variables.

    Two ways to specify the model, in priority order:
      1. LLAMA_HF_REPO  - a Hugging Face GGUF repo; llama-server downloads the
                          model (and, for vision repos, its mmproj projector)
                          into the local llama.cpp cache on first run.
      2. LLAMA_MODEL     - a path to a local .gguf file (+ optional LLAMA_MMPROJ).
    """
    binary = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    # Resolve the binary to an absolute path (accept either a direct path or a
    # name on PATH) and fail early with a clear message if it can't be found.
    resolved = binary if os.path.isfile(binary) else shutil.which(binary)
    if resolved is None:
        raise SystemExit(
            f"llama-server binary not found: '{binary}'.\n"
            f"Set LLAMA_SERVER_BIN to the full path of llama-server.exe (e.g. your "
            f"Vulkan build), or add it to PATH. If LLAMA_SERVER_BIN is set in your "
            f"shell it overrides the project default, so unset it to use the default."
        )
    binary = resolved

    host = os.environ.get("LLAMA_HOST", "127.0.0.1")
    port = os.environ.get("LLAMA_PORT", "8080")

    hf_repo = os.environ.get("LLAMA_HF_REPO")
    mmproj = os.environ.get("LLAMA_MMPROJ")

    if hf_repo:
        # -hf auto-downloads the matching mmproj for multimodal repos.
        command = [binary, "-hf", hf_repo, "--host", host, "--port", str(port)]
        if mmproj:
            command += ["--mmproj", mmproj]
    else:
        model = os.environ.get("LLAMA_MODEL")
        if not model:
            raise SystemExit(
                "No model configured. Set LLAMA_HF_REPO to a Hugging Face GGUF "
                "repo (e.g. Qwen/Qwen3-VL-4B-Instruct-GGUF), or LLAMA_MODEL to a "
                "local .gguf path."
            )
        command = [binary, "-m", model, "--host", host, "--port", str(port)]
        if mmproj:
            command += ["--mmproj", mmproj]

    extra = os.environ.get("LLAMA_EXTRA_ARGS")
    if extra:
        command += shlex.split(extra, posix=(os.name != "nt"))

    return command


def _wait_for_ready(
    server: subprocess.Popen, timeout: "float | None"
) -> None:
    """Poll the server until ready.

    Waits indefinitely when ``timeout`` is ``None`` (the default) so first-run
    model downloads are never cut short. Raises if the server process exits
    before becoming ready, or if an explicit timeout elapses.
    """
    deadline = None if timeout is None else time.time() + timeout
    while True:
        if _is_server_ready():
            return
        exit_code = server.poll()
        if exit_code is not None:
            raise SystemExit(
                f"llama-server exited (code {exit_code}) before becoming ready"
            )
        if deadline is not None and time.time() >= deadline:
            raise SystemExit(
                f"llama-server did not become ready within {timeout:.0f}s "
                f"at {_base_url()}"
            )
        time.sleep(2.0)


def _startup_timeout() -> "float | None":
    """Return the readiness timeout, or None (wait forever) if unset."""
    value = os.environ.get("LLAMA_STARTUP_TIMEOUT")
    return float(value) if value else None


def _serve_only() -> int:
    """Start llama-server in the foreground and keep it running."""
    if _is_server_ready():
        print(f"[run_pipeline] A server is already running at {_base_url()}")
        return 0

    command = _build_server_command()
    print(f"[run_pipeline] Starting llama-server: {' '.join(command)}")
    server = subprocess.Popen(command)
    try:
        return server.wait()
    except KeyboardInterrupt:
        print("\n[run_pipeline] Stopping llama-server")
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Start the llama-server and keep it running (do not run the batch).",
    )
    args = parser.parse_args()

    if args.serve_only:
        return _serve_only()

    # Reuse an already-running server if one is up.
    if _is_server_ready():
        print(f"[run_pipeline] Reusing running llama-server at {_base_url()}")
        _warn_if_model_mismatch()
        from main import main as run_batch
        run_batch()
        return 0

    command = _build_server_command()

    print(f"[run_pipeline] Starting llama-server: {' '.join(command)}")
    print("[run_pipeline] Waiting for server to be ready "
          "(first run downloads the model; this can take a while)...")
    server = subprocess.Popen(command)
    try:
        _wait_for_ready(server, _startup_timeout())
        print(f"[run_pipeline] Server ready at {_base_url()}, starting batch")

        from main import main as run_batch
        run_batch()
        return 0
    finally:
        print("[run_pipeline] Shutting down llama-server")
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
