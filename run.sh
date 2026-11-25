#!/usr/bin/env bash
# Serve Inkling 2 Mini on NVIDIA Ampere (sm_80) GPUs: vLLM's own inkling-2-mini image with this repo's patches mounted
# over it. The first start compiles the CUDA extension in csrc/ inside that image (about a minute) and keeps it in
# CACHE. Stop it with Ctrl-C (or `docker stop inkling2-sm80`).
#
#   MODEL=/path/to/Inkling-2-Mini ./run.sh                  # GPU 0, port 8000, 4 draft tokens
#   MODEL=... GPU=1 PORT=8030 BIND=0.0.0.0 ./run.sh         # any vLLM flag can follow, e.g. ./run.sh --max-num-seqs 16
#   MODEL=... GPU=0,1 ./run.sh                              # two GPUs, tensor parallel (faster for one user)
#
# MODEL  the downloaded thinkingmachines/Inkling-2-Mini folder (required)
# GPU    one GPU, or two separated by a comma, as index or UUID as nvidia-smi shows them (default 0)
# PORT   host port (default 8000); BIND host address (default 127.0.0.1; there is no API key)
# K      DSpark draft tokens, 1-7 (default 4: best on chat, code, long context and several users)
# LEN    context length (default 262144)
# BATCH  tokens per prefill chunk (default 12288)
# UTIL   share of GPU memory vLLM may take (default 0.96 on one GPU, 0.92 on two); lower it if the GPU also drives a
#        display or runs anything else
# CONV_TAIL  1 (default): the short-conv state caches back only each prefill chunk's tail, which holds about 2.5x
#        the KV tokens; 0 restores vLLM's plain sliding-window layout for them
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
BATCH="${BATCH:-12288}"
CONV_TAIL="${CONV_TAIL:-1}"
CACHE="${CACHE:-${HOME}/.cache/inkling2-sm80}"
# vllm/vllm-openai:inkling-2-mini as published 20 Nov 2026 (vLLM 0.1.dev25855+g7cb5dae22). The patches replace whole
# files of that build; a newer image may need them redone (see patches/sm80.diff).
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}"
VLLM=/usr/local/lib/python3.14/dist-packages/vllm

[[ -f "${MODEL}/config.json" ]] || { echo "no config.json in ${MODEL}" >&2; exit 1; }
IFS=',' read -r -a gpus <<< "${GPU}"
case "${#gpus[@]}" in
    1) split=(); UTIL="${UTIL:-0.96}" ;;
    2) split=(--tensor-parallel-size 2 --disable-custom-all-reduce); UTIL="${UTIL:-0.92}" ;;
    *) echo "GPU: give one or two GPUs" >&2; exit 1 ;;
esac
mkdir -p "${CACHE}"

# The CUDA extension, built once per source / image pair.
sum="$(cat "${HERE}/csrc/inkling_sm80_cuda.cu" "${HERE}/csrc/build.sh" <(echo "${IMAGE}") | sha256sum | cut -c1-16)"
EXT="${CACHE}/sm80ext-${sum}"
if ! ls "${EXT}"/_inkling_sm80_cuda*.so >/dev/null 2>&1; then
    echo "building the sm_80 CUDA extension (once, about a minute)"
    mkdir -p "${EXT}"
    docker run --rm --entrypoint bash \
        -v "${HERE}/csrc:/csrc:ro" -v "${EXT}:/out" -e OUT_DIR=/out \
        "${IMAGE}" /csrc/build.sh
fi

mounts=()
while IFS= read -r f; do
    mounts+=(-v "${HERE}/patches/vllm/${f}:${VLLM}/${f}:ro")
done < <(cd "${HERE}/patches/vllm" && find . -name '*.py' | sed 's|^\./||' | sort)

exec docker run --rm --name "${NAME:-inkling2-sm80}" \
    --gpus "\"device=${GPU}\"" --ipc=host \
    -p "${BIND}:${PORT}:8000" \
    -v "${MODEL}:/model:ro" \
    -v "${CACHE}:/root/.cache" \
    -v "${HERE}/site:/sm80site:ro" \
    -v "${EXT}:/sm80ext:ro" \
    "${mounts[@]}" \
    -e INKLING_SM80=1 -e INKLING_CONV_TAIL="${CONV_TAIL}" -e PYTHONPATH=/sm80site:/sm80ext \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    -e TRITON_CACHE_DIR=/root/.cache/triton -e HF_HUB_OFFLINE=1 \
    "${IMAGE}" /model \
    --served-model-name inkling-2-mini --host 0.0.0.0 --port 8000 \
    "${split[@]}" \
    --max-model-len "${LEN}" \
    --gpu-memory-utilization "${UTIL}" \
    --language-model-only \
    --reasoning-parser inkling \
    --enable-auto-tool-choice --tool-call-parser inkling \
    --max-num-batched-tokens "${BATCH}" \
    --max-num-seqs 8 \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${K},\"draft_sample_method\":\"probabilistic\"}" \
    "$@"
