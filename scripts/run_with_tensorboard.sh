#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
VENV_TENSORBOARD="$ROOT_DIR/.venv/bin/tensorboard"
OUTPUTS_DIR="$ROOT_DIR/outputs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

choose_config_interactively() {
    mapfile -t config_files < <(find "$ROOT_DIR/configs" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort)

    if [[ ${#config_files[@]} -eq 0 ]]; then
        echo "No YAML configs found in $ROOT_DIR/configs"
        exit 1
    fi

    echo "Select a config:" >&2
    local index
    for index in "${!config_files[@]}"; do
        printf "%2d) %s\n" "$((index + 1))" "$(basename "${config_files[$index]}")" >&2
    done

    while true; do
        read -r -p "Enter selection [1-${#config_files[@]}]: " selection >&2

        if [[ "$selection" =~ ^[0-9]+$ ]] && (( selection >= 1 && selection <= ${#config_files[@]} )); then
            CONFIG_PATH="${config_files[$((selection - 1))]}"
            return 0
        fi

        echo "Invalid selection. Enter a number from 1 to ${#config_files[@]}." >&2
    done
}

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 <config_path>"
    echo "   or: $0 -i"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    if [[ "$1" == "-i" ]]; then
        choose_config_interactively
        echo "Selected config: $(basename "$CONFIG_PATH")"
    else
        CONFIG_ARG="$1"
        if [[ "$CONFIG_ARG" = /* ]]; then
            CONFIG_PATH="$CONFIG_ARG"
        else
            CONFIG_PATH="$ROOT_DIR/$CONFIG_ARG"
        fi
    fi
else
    echo "Usage: $0 <config_path>"
    echo "   or: $0 -i"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Missing virtualenv python: $VENV_PYTHON"
    exit 1
fi

if [[ ! -x "$VENV_TENSORBOARD" ]]; then
    echo "Missing tensorboard executable: $VENV_TENSORBOARD"
    echo "Install it with:"
    echo "  $VENV_PYTHON -m pip install tensorboard 'setuptools<81'"
    exit 1
fi

timestamp_now() {
    date '+%Y-%m-%d %H:%M:%S'
}

resolve_tensorboard_host() {
    if [[ -n "${SSH_CONNECTION:-}" ]]; then
        # SSH_CONNECTION format: client_ip client_port server_ip server_port
        local ssh_server_ip
        ssh_server_ip="$(awk '{print $3}' <<<"$SSH_CONNECTION")"
        if [[ -n "$ssh_server_ip" ]]; then
            printf '%s\n' "$ssh_server_ip"
            return 0
        fi
    fi

    printf '127.0.0.1\n'
}

find_free_port() {
    "$VENV_PYTHON" - <<'PY'
import socket

for port in range(6006, 6106):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    raise SystemExit(0)

raise SystemExit("No free port found in range 6006-6105")
PY
}

cleanup() {
    if [[ -n "${TENSORBOARD_PID:-}" ]] && kill -0 "$TENSORBOARD_PID" 2>/dev/null; then
        kill "$TENSORBOARD_PID" 2>/dev/null || true
        wait "$TENSORBOARD_PID" 2>/dev/null || true
    fi
    if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

mkdir -p "$OUTPUTS_DIR"
START_STAMP="$(timestamp_now)"
LAUNCH_LOG="$OUTPUTS_DIR/launch_$(date '+%Y%m%d_%H%M%S').log"
CONFIG_NAME="$(basename "$CONFIG_PATH")"

echo "Starting training with config: $CONFIG_NAME"
echo "Launcher log: $LAUNCH_LOG"

(
    cd "$ROOT_DIR"
    "$VENV_PYTHON" src/train.py "$CONFIG_PATH"
) >"$LAUNCH_LOG" 2>&1 &
TRAIN_PID=$!

RUN_DIR=""
for _ in $(seq 1 120); do
    if grep -q "Output directory:" "$LAUNCH_LOG" 2>/dev/null; then
        RUN_DIR="$(grep -m1 "Output directory:" "$LAUNCH_LOG" | sed 's/^Output directory: //')"
        break
    fi

    RUN_DIR="$(find "$OUTPUTS_DIR" -maxdepth 1 -mindepth 1 -type d -name 'output_*' -newermt "$START_STAMP" | sort | tail -n 1)"
    if [[ -n "$RUN_DIR" ]]; then
        break
    fi

    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "Training exited before creating a run directory."
        cat "$LAUNCH_LOG"
        exit 1
    fi
    sleep 1
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Failed to detect the new run directory."
    cat "$LAUNCH_LOG"
    exit 1
fi

TENSORBOARD_DIR="$RUN_DIR/tensorboard"
mkdir -p "$TENSORBOARD_DIR"

PORT="$(find_free_port)"
TENSORBOARD_HOST="$(resolve_tensorboard_host)"
WATCH_URL="http://$TENSORBOARD_HOST:$PORT"

(
    cd "$ROOT_DIR"
    "$VENV_TENSORBOARD" --logdir "$TENSORBOARD_DIR" --host "$TENSORBOARD_HOST" --port "$PORT"
) >"$RUN_DIR/logs/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!

echo "Run directory: $RUN_DIR"
echo "TensorBoard directory: $TENSORBOARD_DIR"
echo "TensorBoard log: $RUN_DIR/logs/tensorboard.log"
echo "Watch URL: $WATCH_URL"
if [[ -n "${SSH_CONNECTION:-}" ]]; then
    echo "SSH session detected; TensorBoard is bound to the server address for access from the SSH client."
fi
echo
echo "Training output follows. Press Ctrl+C to stop training and TensorBoard."
echo

DEBUG_LOG="$RUN_DIR/logs/debug.log"
if [[ -f "$DEBUG_LOG" ]]; then
    tail --pid="$TRAIN_PID" -n +1 -f "$DEBUG_LOG"
else
    tail --pid="$TRAIN_PID" -n +1 -f "$LAUNCH_LOG" &
    TAIL_PID=$!

    for _ in $(seq 1 30); do
        if [[ -f "$DEBUG_LOG" ]]; then
            kill "$TAIL_PID" 2>/dev/null || true
            wait "$TAIL_PID" 2>/dev/null || true
            tail --pid="$TRAIN_PID" -n +1 -f "$DEBUG_LOG"
            break
        fi

        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            wait "$TAIL_PID" 2>/dev/null || true
            break
        fi

        sleep 1
    done
fi

wait "$TRAIN_PID"
