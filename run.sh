#!/usr/bin/env bash
# Serve Inkling 2 Mini on one NVIDIA Ampere (sm_80) GPU: vLLM's own inkling-2-mini image with this repo's patches
# mounted over it. Nothing is built or installed; stop it with Ctrl-C (or `docker stop inkling2-sm80`).
#
#   MODEL=/path/to/Inkling-2-Mini ./run.sh                  # GPU 0, port 8000, 4 draft tokens
#   MODEL=... GPU=1 PORT=8030 K=7 BIND=0.0.0.0 ./run.sh     # any vLLM flag can follow, e.g. ./run.sh --max-num-seqs 16
#
# MODEL  the downloaded thinkingmachines/Inkling-2-Mini folder (required)
# GPU    GPU index or UUID as nvidia-smi shows it, or a comma list for tensor parallel, e.g. 0,1 (default 0)
# PORT   host port (default 8000); BIND host address (default 127.0.0.1; there is no API key)
# K      DSpark draft tokens, 1-7 (default 4: best on chat, code, long context and several users; 7 for very
#        predictable text); 0 serves without the drafter
# LEN    context length (default 262144)
# CACHE  where compiled kernels are kept between runs (default ~/.cache/inkling2-sm80)
# IMAGE  defaults to the image build these patches were made against (pinned by digest)
# NAME   container name (default inkling2-sm80)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?set MODEL to the downloaded thinkingmachines/Inkling-2-Mini folder}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
BIND="${BIND:-127.0.0.1}"
K="${K:-4}"
LEN="${LEN:-262144}"
CACHE="${CACHE:-${HOME}/.cache/inkling2-sm80}"
# vllm/vllm-openai:inkling-2-mini as published 20 Nov 2026 (vLLM 0.1.dev25855+g7cb5dae22). The patches replace whole
# files of that build; a newer image may need them redone (see patches/sm80.diff).
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}"
VLLM=/usr/local/lib/python3.14/dist-packages/vllm

[[ -f "${MODEL}/config.json" ]] || { echo "no config.json in ${MODEL}" >&2; exit 1; }
mkdir -p "${CACHE}"

mounts=()
while IFS= read -r f; do
    mounts+=(-v "${HERE}/patches/vllm/${f}:${VLLM}/${f}:ro")
done < <(cd "${HERE}/patches/vllm" && find . -name '*.py' | sed 's|^\./||' | sort)

# K=0 serves without the DSpark drafter (frees its weights and KV for context).
spec=()
if [[ "${K}" != 0 ]]; then
    spec=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${K},\"draft_sample_method\":\"probabilistic\"}")
fi

exec docker run --rm --name "${NAME:-inkling2-sm80}" \
    --gpus "\"device=${GPU}\"" --ipc=host \
    -p "${BIND}:${PORT}:8000" \
    -v "${MODEL}:/model:ro" \
    -v "${CACHE}:/root/.cache" \
    -v "${HERE}/site:/sm80site:ro" \
    "${mounts[@]}" \
    -e INKLING_SM80=1 -e PYTHONPATH=/sm80site -e INKLING_SM80_TRACE="${INKLING_SM80_TRACE:-0}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    -e TRITON_CACHE_DIR=/root/.cache/triton -e HF_HUB_OFFLINE=1 \
    "${IMAGE}" /model \
    --served-model-name inkling-2-mini --host 0.0.0.0 --port 8000 \
    --max-model-len "${LEN}" \
    --gpu-memory-utilization 0.90 \
    --language-model-only \
    --reasoning-parser inkling \
    --enable-auto-tool-choice --tool-call-parser inkling \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    ${spec[@]+"${spec[@]}"} \
    "$@"
