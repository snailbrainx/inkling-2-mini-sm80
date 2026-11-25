# Inkling 2 Mini on Ampere (sm_80)

Patches that let [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) run, and run
fast, on NVIDIA Ampere GPUs with vLLM.

The model is served by vLLM's own image `vllm/vllm-openai:inkling-2-mini`. That image only supports Hopper and newer
(compute capability 9.0+). This repository mounts patched files over it, plus Triton stand-ins for ops the image only
ships for newer GPUs, and a small CUDA extension that `run.sh` compiles inside the same image on first start. The image
itself is not modified.

Tested on an NVIDIA CMP 170HX: a GA100 (the A100's chip), sm_80, 64 GB, power-capped at 180 W. It should behave the same
on an A100 (also sm_80), but that is untested. Other Ampere chips (sm_86 / sm_89) have less shared memory and are
untested.

## Quick start

```bash
# 1. the model (about 19 GB)
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini
# 2. serve it on GPU 0, port 8000
MODEL=~/models/Inkling-2-Mini ./run.sh
# 3. ask it something
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "chat_template_kwargs": {"reasoning_effort": "none"}}'
```

The first start compiles the CUDA extension (about a minute) and then Triton kernels. Both are cached in
`~/.cache/inkling2-sm80`, so later starts take about two minutes. A few kernels still compile the first time they meet a
new prompt length, so the very first long prompt after a fresh install is slower, and the first time
it serves a new number of users at once, one reply can start up to about 3 s late. Both happen once.

`run.sh` takes its settings from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | (required) | the downloaded model folder |
| `GPU` | `0` | GPU index or UUID; two separated by a comma (`0,1`) split the model over both (tensor parallel) |
| `PORT` / `BIND` | `8000` / `127.0.0.1` | where the OpenAI-compatible API listens. There is no API key, so only bind to `0.0.0.0` on a trusted network |
| `K` | `4` | how many tokens the DSpark drafter proposes per step (1-7), see below |
| `LEN` | `262144` | context length |
| `BATCH` | `12288` | prompt tokens per prefill chunk |
| `UTIL` | `0.96` (one GPU), `0.92` (two) | share of GPU memory vLLM may take; lower it if the GPU also drives a display or runs anything else |
| `CACHE` | `~/.cache/inkling2-sm80` | compiled kernels |
| `IMAGE` | the tested build, pinned by digest | the patches replace whole files of that build |
| `NAME` | `inkling2-sm80` | container name, to run several at once |

Any extra arguments are passed to `vllm serve`, e.g. `./run.sh --max-num-seqs 16`, or for 1M context on a single 64 GB
card with FP8 KV cache (software E4M3FN storage on sm_80, BF16 draft KV):

```bash
MODEL=~/models/Inkling-2-Mini LEN=1048576 BATCH=4096 ./run.sh --kv-cache-dtype fp8
```

Thinking is controlled through the chat template: `"chat_template_kwargs": {"reasoning_effort": "none"}` turns it
off. Even then, as the model card notes, the model still reasons briefly.

## Results

One CMP 170HX at 180 W, PCIe Gen2 x16, DSpark k=4, `reasoning_effort: "none"`. v1 and v2 were measured side by side on
24 Nov 2026, on two identical cards with the same scripts. Each server first ran a throwaway pass of every test, so no
kernel was compiling during the measured run, and every measured prompt was new (no prefix-cache hits).

| | v1 | **v2** | Change |
|---|---:|---:|---:|
| Prompt processing tok/s, 8k / 32k / 128k / 262k | 7,109 / 6,622 / 5,056 / 3,788 | **9,001 / 8,483 / 6,307 / 4,674** | +23% to +28% |
| Time to first token, 32k / 128k / 262k | 4.9 / 25.9 / 69.0 s | **3.9 / 20.8 / 55.9 s** | -19% to -22% |
| Recipe prompts, one user: lists / code / prose (tok/s) | 294 / 212 / 164 | **315 / 246 / 177** | +7% / +16% / +8% |
| Recipe prompts, 2 users (total tok/s) | 504 / 355 / 308 | **545 / 364 / 324** | +8% / +2% / +5% |
| Recipe prompts, 4 users | 827 / 509 / 466 | **905 / 527 / 468** | +9% / +4% / 0% |
| Recipe prompts, 8 users | 1,193 / 713 / 670 | **1,349 / 791 / 717** | +13% / +11% / +7% |
| Recipe prefill test, 2k / 32k prompts (tok/s) | 6,339 / 6,676 | **7,962 / 8,607** | +26% / +29% |
| Chat prompts, one user (tok/s; time per step) | 183; 13.5 ms | **194; 12.5 ms** | +6% |
| Counting 1-200, one user | 302 | **324** | +7% |
| 1k random-word prompts, 1 / 2 / 4 / 8 at once (total tok/s) | 143 / 207 / 312 / 456 | **150 / 227 / 343 / 501** | +5% to +10% |
| Generation after a 32k / 128k / 262k prompt (tok/s) | 144 / 116 / 91 | 140 / 121 / 90 | level |

The recipe prompts are the ones the published GLM and DeepSeek recipes for this card use: count 1-200 (lists), Python
tasks (code) and a hash-map explanation (prose), 400 tokens at temperature 0, median of 5 runs; with several users the
figure is the total of all of them. The last two rows sample at temperature 0.7, which makes the drafter's hit rate,
and with it the speed, swing by about ±10% from run to run; generation after a long prompt is level between v1 and v2
within that noise. Single runs otherwise vary by about ±5% for one user.

**Two GPUs** (`GPU=0,1`, tensor parallel over a PCIe Gen2 x16 pair with peer-to-peer access), v2, measured once:

| | One GPU | Two GPUs |
|---|---:|---:|
| Recipe prompts, one user: lists / code / prose | 315 / 246 / 177 | 338 / 263 / 182 |
| Recipe prompts, 8 users | 1,349 / 791 / 717 | 1,095 / 704 / 679 |
| Chat prompts, one user (time per step) | 194 (12.5 ms) | 198 (11.7 ms) |
| Counting 1-200, one user | 324 | 352 |
| Prompt processing, 32k / 128k / 262k | 8,483 / 6,307 / 4,674 | 7,125 / 6,255 / 5,343 |
| Time to first token at 262k | 55.9 s | 48.7 s |
| Generation after a 262k prompt | 90 | 117 |
| KV cache | 286,563 tokens | 418,253 tokens |

Two GPUs help one user a little, very long prompts more, and hold half as much again of KV cache. With several users at
once one GPU is faster, and two one-GPU servers serve twice as many.

### Why the kind of text matters

Each step, the DSpark drafter guesses the next `K` tokens and the model checks them all in one pass. A step takes
about the same time (about 12.5 ms at k=4) however many guesses are right, so speed comes down to how predictable the text
is:

| Kind of text (k=4, one user) | tok/s | Tokens kept per step |
|---|---:|---:|
| Counting 1-200 | 324 | 3.9 of a possible 5 |
| Lists, structured text | 315 | |
| Code | 246 | |
| Chat answers | 194 | 2.4 |
| Prose | 177 | |
| Random words, temperature 0.7 | 150 | |

Sampling at a higher temperature rejects more guesses than greedy decoding, so it is slower. That is also why the best
`K` depends on the text: long runs of guesses come true when counting (k=7 is fastest there), while on chat, code and
prose the later guesses rarely survive and checking them only costs time. k=4 is the best all-rounder (on v1 against k=3, 5 and
7; on v2 re-checked against k=5).

## Output quality

Every change keeps the same weights and the same maths; only the order of bf16 rounding differs. v2 against v1, same
prompts, temperature 0:

- Perplexity over 16,675 tokens of fixed text: mean negative log-likelihood 0.526 (v2) against 0.527 (v1).
- 55 multi-step arithmetic word problems: 55 correct on both.
- Greedy replies (28 prompts, 300 tokens): 7 identical, the rest identical up to a median of ~300 characters, after which
  the wording drifts (e.g. "asks for description" against "asks to describe"), as it does with any change of kernels.

## What the patches do

**Making it run** (on its own the image refuses sm_80, and some parts would fail silently):

- `quant_config.py`: lowers the minimum capability from 9.0 to 8.0. The experts already go through Humming (bf16 maths
  on NVFP4 / MXFP8 weights), which has sm_80 heuristics.
- `ops/linear.py`, `ops/conv_mix.py`: the TMA linear kernel and the compiled conv/mix op are Hopper / Blackwell only.
  On sm_80 they take the code's SM12x branches (plain Triton) or this repository's kernels.
- `site/sitecustomize.py`: Triton's grid-dependency-control helpers (programmatic dependent launch, sm_90+) become
  empty, and launches run without PDL.
- **Compiled ops without sm_80 kernels.** The image's `_C`, `_C_cache_ops` and `_moe_C` libraries are built for
  sm_100. On sm_80, an op with no kernel launches nothing and leaves `cudaErrorNoKernelImageForDevice` behind for some
  later call to report, far from the cause. Three such ops matter here: `moe_align_block_size` (expert routing),
  `reshape_and_cache_flash` (the drafter's KV cache) and, with two GPUs, `vocab_parallel_embedding`. All have
  replacements in `site/`. Set `INKLING_SM80_TRACE=1` to wrap every compiled op and log the first one that leaves a CUDA
  error.

**Making it fast, v1** (21 Nov 2026):

- Humming expert-GEMM tiles for decode-sized batches, from a 192-config sweep (`ops/moe_sm90.py`).
- Every small bf16 matmul, both vocab heads included, on the image's own low-row Triton GEMV with tiles chosen per
  shape, measured with the weights cycled past the 40 MB L2 as in the model (`ops/gemv.py`). cuBLAS picks split-K
  kernels plus a reduce here.
- One launch for a loop pass's q-side, k and v projections (`attention.py`), two Triton kernels for the loop
  controllers' batched matmuls (`loop.py`), the drafter's Markov correction on the GEMV (`dspark.py`).
- Full-attention blocks wholly beyond the relative-position range skip the bias and mask (`ops/attention_sm90.py`).

**Making it fast, v2** (24 Nov 2026):

- **A CUDA extension** (`csrc/inkling_sm80_cuda.cu`, loaded by `site/inkling_sm80_cuda.py`): the conv/mix step as one
  kernel for decode and a two-pass kernel for prefill; MoE routing (top-k select fused with block alignment, and a
  counting sort for long prefills); the loop controller's bank messages from rank-2 coefficients instead of a
  full fp32 message tensor (much faster prefill, 300-450 MB less memory); the pair norm fused into the controller for
  one-token decode; and five drafter kernels that replace some 80 small launches per step.
- **Humming epilogue** (`site/humming_include/`): the expert activation (softcapped SwiGLU) and routing weight are
  applied inside the first expert GEMM's epilogue, removing 48 launches per step. Expert tiles retuned for real
  routing skew at 1-8 users and for 180 W prefill (`ops/moe_sm90.py`).
- **Attention** (`ops/attention_sm90.py`, `ops/attention.py`, `ops/sconv.py`): far keys through FlashInfer and the
  1,024 near keys (where the relative bias applies) through Triton; retuned local prefill and decode tiles.
- **Small matmuls**: the Triton GEMV now also covers 17-64 rows (4-8 users) for the output projection and drafter
  layers; cuBLAS for the controller's group partials on long prefills (`ops/gemv.py`, `ops/linear.py`, `loop.py`).
- **Defaults**: 12,288-token prefill chunks and 96% of GPU memory (286,563 KV tokens, 1.09x a full 262k context).
- **Two GPUs** work (tensor parallel), with `--disable-custom-all-reduce`.

The v2 kernels were tuned with Neon.

**Tried, no gain:**

- int8 drafter weights: the in-kernel conversion costs more than the halved reads save.
- W8A8 small matmuls: quantizing the activations costs more than it saves, and it changes outputs.
- Marlin 4-bit: no sm_80 kernel in this image.
- Fusing chains of small kernels with grid barriers: slower; these kernels are bound by dependent memory reads.
- A plain CUDA GEMV: 0.45-0.77x the Triton tensor-core GEMV.
- FlashInfer's causal path beyond 64k of context, and a gated local-prefill epilogue: both slower.
- 16,384-token prefill chunks: a full 262k context no longer fits.
- A smaller draft vocabulary: saves up to ~0.7 ms a step, but rare tokens could no longer be drafted.

`patches/sm80.diff` shows every change against the image's files. `tools/trace_kernels.py` sums kernel time from a
vLLM torch-profiler trace: start the server with `--profiler-config`, then POST `/start_profile` and `/stop_profile`.

## Limits

- Text only (`--language-model-only`); the vision tower has not been tried on sm_80.
- The default KV cache is bf16. Passing `--kv-cache-dtype fp8` enables unit-scale E4M3FN KV storage on sm_80 (with the
  DSpark draft cache kept in bf16); at `LEN=1048576 BATCH=4096`, a single 64 GB card holds ~1.6M KV tokens (1.53x 1M
  context). Larger `BATCH` reserves more sliding-window blocks across the 48 SWA layers.
- Two GPUs were tested only as a peer-to-peer pair on PCIe Gen2 x16, with `--disable-custom-all-reduce`.
- Made against the image build of 20 Nov 2026 (vLLM `0.1.dev25855+g7cb5dae22`). Expect to redo the patches when that
  image or upstream vLLM changes these files.

## Licence

Apache-2.0, see `LICENSE`. The files under `patches/vllm/` are modified copies of vLLM source (Apache-2.0, copyright the
vLLM contributors), and their SPDX headers are kept. `site/humming_include/` holds a modified Humming header
(Apache-2.0, copyright the Humming contributors). `NOTICE` lists what was changed. The model weights are not part of
this repository; Inkling 2 Mini is released by Thinking Machines under Apache-2.0.
