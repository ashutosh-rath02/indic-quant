# indic-quant-pilot

A pilot study of how much llama.cpp quantization damages a small multilingual model in Indic languages, and whether an importance matrix calibrated on Indic text reduces that damage compared with an English-calibrated one at the same file size.

## Why

Quantized GGUF models are usually calibrated and evaluated on English. Indic scripts are tokenized into many more tokens per word, and their representations may rely on weights that English calibration data rarely activates. If quantization hurts Indic languages much more than English, and Indic calibration recovers a meaningful share of that loss, a full benchmark (more models, all 22 scheduled languages, larger budgets) is worth building. This repository answers that question cheaply for one 4B model, and the generated report states which of those two conditions hold.

## What it measures

| Measurement | How | Output |
|---|---|---|
| KL divergence per language (core metric) | `llama-perplexity --kl-divergence-base` on the BF16 model saves reference logits for each language's FLORES+ devtest text; `--kl-divergence` scores every quantized variant on the identical token sequence | mean, median, 99th percentile KLD, top-token agreement, PPL ratio |
| MILU accuracy | Official MILU prompt format, k-shot examples from the validation split, next-token log-probabilities of the letters A to D through `llama-server` | accuracy with bootstrap CI |
| Translation | English to X on 200 FLORES+ dev sentences, greedy decoding through the model's chat template | corpus chrF++ (sacrebleu) |
| Tokenizer fertility | `llama-tokenize` on the base GGUF | tokens per word, per character, per grapheme |

Variants (19 for the default config): BF16 base; Q8_0, Q6_K, Q5_K_M without imatrix; Q4_K_M, Q3_K_M, Q2_K each with no imatrix, English imatrix, Indic imatrix, and Indic without Tamil; IQ2_M with the three imatrix arms (llama.cpp refuses IQ2_M without an imatrix). Run `python -m iqp plan` to list them.

Calibration arms all use the same token budget (131072 model tokens, 256 chunks of 512) measured with the model's own tokenizer:

* `en`: English Wikipedia.
* `indic`: equal tokens of Hindi, Bengali, Tamil, Telugu, Marathi Wikipedia, interleaved paragraph by paragraph so any prefix stays balanced. `english_fraction` can mix in English.
* `indic_no_tam`: the same without Tamil, to test whether the benefit transfers to an unseen Indic language.

## Data and disjointness

| Use | Source | Notes |
|---|---|---|
| KLD and fertility text | FLORES+ `devtest` (`openlanguagedata/flores_plus`, gated) | n-way parallel, so languages see the same content |
| Translation | FLORES+ `dev` | a different split from the KLD text |
| MILU | `ai4bharat/MILU` (gated), `test` for scoring, `validation` for shots | 500 questions per language by default |
| Calibration | `wikimedia/wikipedia` 20231101 (default) or `ai4bharat/sangraha` verified | different corpus from all evaluation data |
| Fallback eval | `ai4bharat/IN22-Gen` (gated) | single split; KLD and MT ids are split disjointly |

`prepare_data` enforces disjointness and fails if it is violated: calibration paragraphs that contain an evaluation sentence or share any word 8-gram with one (any language) are dropped; the final calibration files are re-checked for verbatim containment; MT source sentences are removed if they also appear in the KLD text; MILU validation rows that duplicate a test question are dropped before choosing shots. Note that FLORES+ sentences come from Wikinews, Wikivoyage and Wikibooks, so Wikipedia calibration text is a different corpus, but topical overlap is possible; the n-gram filter is the guard.

`python -m iqp validate_data` opens one file from every configured dataset, prints its schema and a sample, and fails loudly on missing columns. Run it first.

## Hardware and disk

* One NVIDIA GPU with 24 GB (a 4B model in BF16 needs about 9 GB plus KV cache). CUDA toolkit with `nvcc`, CMake 3.18 or newer, a C++17 compiler, git, Python 3.10 to 3.13.
* BF16 is used for the base because Gemma 3 and Qwen3 are trained in bfloat16. On GPUs without fast BF16 (pre-Ampere), set `model.base_dtype: f16`.
* RAM: 32 GB recommended (quantization reads the whole BF16 file).

Disk estimate for Gemma 3 4B (default config):

| Item | Size |
|---|---|
| HF safetensors (includes the vision tower) | about 8.6 GB |
| BF16 GGUF (text model) | about 7.8 GB |
| 18 quantized variants | about 40 GB |
| Base logits for KLD, 6 languages x 24 chunks | about 19 GB (Gemma's 262k vocabulary: about 134 MB per chunk at ctx 512) |
| llama.cpp build, venvs, data, results | about 6 GB |
| **Total** | **about 80 GB; plan for 100 GB** |

Set `model.delete_hf_weights_after_convert: true` and `kld.delete_base_logits_when_done: true` to save about 28 GB. Qwen3-4B needs less space for logits (151k vocabulary) but slightly more for weights.

## Time estimates on one 24 GB GPU

These are estimates for an RTX 4090 / A5000 class card with a recent 8 to 16 core CPU. They were not measured: the build sandbox had no GPU.

| Step | Estimate |
|---|---|
| setup (CUDA build of 5 tools, converter venv) | 10 to 25 min |
| download + convert | 10 to 20 min, network bound |
| validate_data + prepare_data | 5 to 15 min (streams one Wikipedia shard per language) |
| build_imatrix (3 arms x 256 chunks) | 5 to 15 min |
| quantize (18 variants, CPU bound) | 30 to 70 min |
| fertility | under 1 min |
| run_kld (6 base runs + 108 variant runs) | 1 to 2 h |
| run_milu (19 variants x 6 languages x 500 questions) | 3 to 6 h |
| run_mt (19 variants x 5 languages x 200 sentences) | 2 to 5 h |
| aggregate, plot, report | under 1 min |
| **Total** | **about 8 to 15 h** |

For a faster first pass, set `milu.sample_cap: 200` and `mt.n_sentences: 100`, or restrict `milu.variants` and `mt.variants` to a list of variant names (KLD remains complete).

## Running it

```bash
unzip indic-quant-pilot.zip && cd indic-quant-pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Accept the terms on huggingface.co for: google/gemma-3-4b-it, openlanguagedata/flores_plus,
# ai4bharat/MILU (and ai4bharat/IN22-Gen if you use the fallback). Then:
export HF_TOKEN=hf_...

.venv/bin/python -m pytest -q                 # unit tests, about 30 s, no GPU needed
.venv/bin/python -m iqp all --dry-run         # whole pipeline with fake tools, writes *_dryrun/
.venv/bin/python -m iqp plan                  # list variants

.venv/bin/python -m iqp setup                 # clone and build llama.cpp (pinned commit) with CUDA
.venv/bin/python -m iqp download
.venv/bin/python -m iqp convert
.venv/bin/python -m iqp validate_data         # check gated dataset access and schemas
.venv/bin/python -m iqp prepare_data
.venv/bin/python -m iqp build_imatrix
.venv/bin/python -m iqp quantize
.venv/bin/python -m iqp fertility
.venv/bin/python -m iqp run_kld
.venv/bin/python -m iqp run_milu
.venv/bin/python -m iqp run_mt
.venv/bin/python -m iqp analysis              # aggregate + plot + report
```

Or all at once, resumable: `./run_pilot.sh` (or `make all`). Every step skips outputs that already exist, so rerunning after an interruption continues where it stopped. `--force` recomputes a step; `python -m iqp all --from run_kld` restarts from a step; `python -m iqp status` shows what is done. Logs go to `logs/<step>.log` and one file per tool invocation under `logs/<step>/`.

Second model: `CONFIG=configs/qwen3-4b.yaml ./run_pilot.sh` (results go to `results/qwen3-4b/`). Qwen3's thinking mode is turned off for translation through `chat_template_kwargs`.

## Outputs

* `results/results_tidy.csv`: one row per model x variant x language x metric (`model, variant, qtype, arm, language, language_name, role, metric, value, uncertainty, n`). Size rows (`file_gib`, `bpw`) have an empty language.
* `results/wide_<metric>.csv`: variant x language tables.
* `results/calibration_deltas.csv`: per calibrated type and language, KLD under each arm and the relative reduction against the English arm.
* `results/variants.csv`: file size and bits per weight (as reported by llama-quantize) for every variant.
* `results/decision.json` and `results/REPORT.md`: findings in plain numbers and the decision.
* `results/plots/`: `kld_heatmap.png`, `retention_vs_bpw.png`, `calibration_delta.png`, `fertility_vs_kld.png`.
* `results/raw/<model>/`: per-run JSON, per-item MILU predictions and MT outputs for inspection.

## Interpreting the metrics

* **Mean KLD** is the average KL divergence between the base and quantized next-token distributions, over the second half of each 512-token chunk. It is the most sensitive and lowest-noise damage signal. Rough reading for a 4B model: below 0.01 is near lossless, 0.05 to 0.1 is noticeable, above 0.3 usually means visible quality loss.
* **99th percentile KLD** shows whether damage is concentrated in a few badly broken predictions.
* **Same top p** is the share of positions where the quantized model's most likely token matches the base model's.
* **PPL ratio** is PPL(Q)/PPL(base). It can hide damage because errors in both directions cancel; prefer KLD.
* **Per-token vs per-word**: KLD is per token. Indic text uses 2 to 5 times more tokens per word, so equal per-token KLD still means more damage per word. Use the fertility table to convert, and the fertility scatter to see whether damage tracks fragmentation.
* **Calibration delta**: positive means the Indic imatrix gives lower KLD than the English imatrix for the same quant type. Variants with the same quant type have the same size to within rounding (the imatrix changes values, not tensor types), so this is an equal-size comparison. The report counts how many languages improved beyond two combined standard errors (a conservative test, since it ignores the positive correlation between arms scored on the same tokens).
* **MILU accuracy** has a 25% chance floor and a few points of sampling noise at 500 questions (see the CI). Small quantization effects will not be visible here; large ones will.
* **chrF++** measures translation and instruction following together; a quantized model that starts explaining instead of translating loses points.

Decision rule (configurable in `decision:`): **strong signal** if the Indic imatrix reduces mean Indic KLD by at least 10% versus the English imatrix in at least 2 of the 4 calibrated types, and the worst Indic language has at least 2x the English KLD at Q4_K_M without imatrix. One condition alone is a **partial signal**.

## Adding models and languages

* **Model**: change `model.hf_id` and `model.short_name` (any architecture `convert_hf_to_gguf.py` supports). Outputs are keyed by `short_name`, so several models can share `work/`.
* **Language**: add its code to `languages.enabled`. Kannada, Malayalam, Gujarati, Odia and Punjabi are already in `languages.registry` with their MILU config names, Wikipedia subsets and Sangraha splits (FLORES+ codes `kan_Knda, mal_Mlym, guj_Gujr, ory_Orya, pan_Guru` exist on the Hub). To include a new language in calibration, add it to the `indic` arm's `langs`. A brand new language needs a registry entry: FLORES+ code, `milu` config name (or disable MILU), `wiki` subset suffix, `sangraha` split and `role`.
* **Quant types**: edit `quant.ladder` and `quant.calibrated_types`. Types listed in `quant.requires_imatrix` get no static arm.

## Known limitations

* One model, one calibration sample per arm, one seed. Differences of a few percent between arms may not replicate.
* KLD is relative to the BF16 GGUF, which itself differs slightly from the HF weights.
* 24 chunks x 255 scored positions per language is about 6k tokens. That is enough for clear differences between quant levels but not for subtle ones. Raise `kld.max_chunks` if disk allows (FLORES+ devtest yields roughly 30k to 90k tokens per language depending on the script).
* Equal chunk counts mean equal token counts, not equal content: one chunk of Tamil covers less text than one chunk of English.
* MILU scoring uses the top 100 next-token candidates. If a letter is missing from that list it is treated as less likely than every listed token; the `all_letters_in_topn` field reports how often this happened.
* `cache_prompt: true` speeds up MILU (shared few-shot prefix) but llama.cpp notes that logits can differ slightly between batch sizes. Set it to false for bit-exact repeatability at some cost in speed.
* Gemma 3 is multimodal; only the text model is converted and evaluated.
* The time estimates above are not measured.

## Verification status

What was checked while building this repository (CPU-only sandbox, no GPU):

* llama.cpp flags and output formats were read from the source at the pinned commit (`tools/perplexity/perplexity.cpp`, `common/arg.cpp`, `tools/quantize/quantize.cpp`, `src/llama-quant.cpp`, `tools/server/README.md`), and llama.cpp was built for CPU from GitHub.
* A real end-to-end CPU smoke run with `Qwen/Qwen3-0.6B` (2 languages, Q4_K_M and IQ2_M, 2 imatrix arms) executed every step of this pipeline with the real binaries: conversion, imatrix, quantization, base logits and KLD, MILU letter scoring and translation through `llama-server`, aggregation, plots and report. Captured outputs from those runs are the parser test fixtures in `tests/fixtures/`.
* The gated datasets (FLORES+, IN22-Gen, MILU) could not be opened. Their file layouts were confirmed from Hub listings and dataset cards, and the MILU column names from the official lm-eval task in `github.com/AI4Bharat/MILU`. The CPU smoke run used a local MILU-shaped file and local Wikipedia text instead. `validate_data` is the check to run first.
* Gemma 3 4B was not downloaded or converted here; it is gated.
