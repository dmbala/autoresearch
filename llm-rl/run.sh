#!/bin/bash
# Usage: run.sh [host_path] <command>
#   host_path  Directory containing autoresearch-rl.sif (default: current directory)
#   command    Command to run inside /workspace in the container (uses container's Python venv)
#
# Examples:
#   ./run.sh python prepare.py
#   ./run.sh python run.py

SIF="/n/netscratch/kempner_dev/Lab/bdesinghu/images/autoresearch-rl.sif"
REPO="${PWD}"
CACHE_DATA="${PWD}/.cache"
HF_CACHE="${HOME}/.cache/huggingface"

# If the first argument is an existing directory, treat it as the host path
if [ -d "$1" ]; then
    REPO="$1"
    shift
fi

if [ ! -f "$SIF" ]; then
    echo "Error: SIF file not found at ${SIF}" >&2
    exit 1
fi

if [ ! -d "$REPO" ]; then
    echo "Error: repo not found at ${REPO}" >&2
    exit 1
fi

mkdir -p "${CACHE_DATA}"
mkdir -p "${HF_CACHE}"

singularity exec --nv \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CACHE_DIR="${CACHE_DATA}" \
    --env HF_HOME="${HF_CACHE}" \
    --bind "${REPO}:/workspace" \
    --bind "${HF_CACHE}:${HF_CACHE}" \
    "$SIF" \
    bash -c "cd /workspace && $*"
