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
| KLD and fertility text | FLORES+ `devtest` (`openlanguagedata/flores_plus`, gated; schema confirmed) | n-way parallel, so languages see the same content |
| Translation | FLORES+ `dev` | a different split from the KLD text |
| MILU | `ai4bharat/MILU` (gated; schema confirmed), `test` for scoring, `validation` for shots | 500 questions per language by default |
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

Set `model.delete_hf_weights_after_convert: true` and `kld.delete_base_logits_when_done: true` to save about 28 GB. With less disk, use the `stream` step (see "Running on Kaggle"), which keeps only one quantized variant on disk at a time. Qwen3-4B needs less space for logits (151k vocabulary) but slightly more for weights.

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

## Running on Kaggle

The Kaggle mode runs the same measurements on one free Kaggle GPU (T4 or P100, 16 GB) with about 20 GB of persisted output space. It differs from the default run in four ways:

* **Streaming variants.** `python -m iqp stream` handles one quantized variant at a time: quantize, KLD on every language, MILU and translation if the variant is selected for them, write results, delete the GGUF. Base logits are computed once per language and deleted once every configured variant has been scored on that language. Peak scratch use for Gemma 3 4B is about 23 GB instead of about 80 GB.
* **F16 base.** Neither the T4 nor the P100 has fast BF16, so `configs/kaggle.yaml` sets `model.base_dtype: f16`. llama.cpp still accumulates the attention products in F32.
* **Smaller budgets.** KLD uses 12 chunks per language (3060 scored tokens) instead of 24. MILU uses 200 questions per language and translation uses 100 sentences, and both run only on 10 of the 19 variants: the base model, Q8_0, and Q4_K_M and Q2_K with every calibration arm. KLD still covers all 19 variants.
* **Two sessions.** A session is limited to 12 hours, and only `/kaggle/working` survives it. Small artifacts (prepared data, imatrices, the KLD plan, size records, measured timings and the built llama.cpp binaries) are written under `/kaggle/working/iqp`, so a second session continues from where the first stopped.

### One-time setup

1. Verify your Kaggle account with a phone number (Settings > Phone verification). GPU and Internet access both require it.
2. On Hugging Face, open each of these pages while logged in and accept the terms: `google/gemma-3-4b-it`, `openlanguagedata/flores_plus`, `ai4bharat/MILU`. Create a read token under Settings > Access Tokens. (For `configs/kaggle-qwen3-4b.yaml` the model is ungated, but the datasets still are.)
3. Create a notebook on Kaggle and import `kaggle/indic_quant_pilot.ipynb` from this repository (File > Import Notebook, or upload it).
4. In the notebook sidebar, under Session options: set Accelerator to **GPU T4 x2** or **GPU P100**, and turn **Internet on**. Choose one GPU type and use it in every session. Only one GPU is used.
5. Add-ons > Secrets: add a secret named `HF_TOKEN` holding the token, and enable it for this notebook.

### Session 1

Click **Save Version**, choose **Save & Run All (Commit)**, and save. This runs the notebook unattended and keeps `/kaggle/working` when it finishes, so you can close the browser. Interactive sessions also work, but they stop when the browser is idle for a while and nothing is saved unless you save a version.

The notebook checks the GPU (`nvidia-smi`), disk (`df -h`) and `nvcc`, reads `HF_TOKEN` from Kaggle Secrets, clones this repository, installs the requirements, and runs:

```
restore, setup, download, convert, validate_data, prepare_data, build_imatrix,
stream --max-hours 10.5, aggregate, plot, report, bundle
```

`validate_data` fails here if one of the gated datasets is not accessible. `stream` stops cleanly before starting a variant that would probably end past 10.5 hours from the start of the session. It estimates each variant from the timings measured so far (times a safety factor of 1.25), logs `Resume next session`, and records the remaining variants in `results/stream_status.json`. Results are written after every language of every variant, so even a session that Kaggle kills at 12 hours loses at most the variant that was running.

### Session 2

1. Open the notebook in the editor (Edit), click **Add Input > Your Work** (or Notebook Output Files), and select this notebook's latest version, which is the output of session 1. It is mounted read-only under `/kaggle/input/<notebook-name>/`.
2. **Save & Run All** again, without changing anything.

The `restore` step searches `/kaggle/input` for `IQP_OUTPUT.json` (or `iqp-output.tar.gz`) and copies state, results and logs into place. `setup` reuses the cached llama.cpp binaries if the commit, the CMake flags, the directory and the CPU type match, and rebuilds otherwise. `prepare_data` and `build_imatrix` find their outputs and skip. The F16 GGUF and the base logits are not carried over (about 17 GB), so session 2 downloads and converts the model again and recomputes the base logits: about 30 to 45 minutes on a T4. `stream` then skips every variant whose results exist and continues. If more variants remain after session 2, attach the session 2 output and run once more. Each output contains everything from earlier sessions.

Without a notebook output, you can also set `paths.restore_from` (or `IQP_RESTORE_FROM`) to any directory or `.tar.gz` holding a previous output. To push the bundle to a private Hugging Face dataset repo instead, set `paths.hf_state_repo: <user>/iqp-state` (needs a write token); `python -m iqp restore --restore-from hf` then pulls it.

`kaggle/run_kaggle.sh` runs the same sequence from a single cell (`!bash kaggle/run_kaggle.sh`); its options are environment variables listed at the top of the file.

### Quotas and time plan

Kaggle gives about 30 GPU hours per week, and a session stops at 12 hours. The pilot needs about 15 to 21 GPU hours for Gemma 3 4B, which fits in two sessions in one week, with a third short session as a reserve.

Time estimates for a T4 with 4 vCPUs. **These are estimates, not measurements**: no Kaggle GPU was available while building this. They scale the 24 GB GPU estimates above by 3 to 4 for GPU work (a T4 has about a quarter of an RTX 4090's memory bandwidth and FP16 throughput) and by about 2 to 3 for CPU work (4 vCPUs instead of 8 to 16 cores). A P100 has more memory bandwidth than a T4 but no tensor cores; expect similar or slightly faster times.

| Step | Session 1 | Session 2 |
|---|---|---|
| setup (CUDA build for sm_60 and sm_75 on 4 vCPUs) | 15 to 30 min | under 1 min if the cached binaries match, else 15 to 30 min |
| download (8.6 GB) and convert to F16 | 10 to 20 min | 10 to 20 min |
| validate_data, prepare_data | 10 to 20 min | under 2 min (restored) |
| build_imatrix (3 arms x 256 chunks of 512) | 10 to 20 min | skipped (restored) |
| base logits (6 languages x 12 chunks) | 3 to 6 min | 3 to 6 min |
| quantize, per variant (CPU bound) | 3 to 6 min; IQ2_M 6 to 12 min | same |
| KLD, per variant (6 languages x 12 chunks) | 3 to 6 min | same |
| MILU, per selected variant (6 x 200 questions, 5-shot) | 20 to 40 min | same |
| translation, per selected variant (5 x 100 sentences) | 15 to 30 min | same |
| aggregate, plot, report, bundle | under 2 min | under 2 min |

Totals: 19 variants x (quantize + KLD), about 3.5 to 6 h, plus 10 variants x (MILU + MT), about 6 to 12 h, plus fixed costs of about 1 to 1.5 h in session 1 and 0.5 to 1 h in session 2. That is about 11 to 19 h of variant work over two sessions of 10.5 h each. The stream order (`stream.priority`) runs the Q4_K_M and Q2_K calibration arms first, so the central comparison is finished in session 1 even if it ends early. If the estimate is too optimistic, a third session completes the rest; lowering `milu.sample_cap` or restricting `milu.variants` / `mt.variants` shortens it.

### Disk plan

| Item, Gemma 3 4B | Where | Size |
|---|---|---|
| HF safetensors, deleted right after conversion | scratch | 8.6 GB, transient |
| F16 GGUF (text model) | scratch | about 7.8 GB |
| base logits, 6 languages x 12 chunks | scratch | 9.0 GiB (133.7 MB per chunk, 262144-token vocabulary) |
| one quantized variant at a time, Q8_0 largest | scratch | at most about 4.1 GB |
| llama.cpp source and build, converter venv | scratch | about 2.5 GB |
| state, results, logs, bundle | /kaggle/working | under 0.5 GB |

Peak scratch is about 23 GB during `stream`, or about 25 GB for a moment during `convert` while the safetensors and the F16 GGUF both exist. Qwen3-4B (151936-token vocabulary) needs 5.2 GiB of logits for the same 12 chunks, about 19 GB peak. Scratch space on Kaggle is not guaranteed, so `download`, `convert`, base logits and each quantization check free space with `shutil.disk_usage` and fail early with a message naming the shortfall. If scratch is too small, set `IQP_WORK_DIR` to another disk in the notebook's configuration cell, or lower `kld.max_chunks`.

### Kaggle configuration notes

* `-DCMAKE_CUDA_ARCHITECTURES=60;75` builds device code for exactly the P100 (compute 6.0) and the T4 (compute 7.5). llama.cpp's default list covers about eight architectures, which on 4 vCPUs roughly triples compile time, and it needs a PTX JIT step for these GPUs. Override with `IQP_CUDA_ARCHS` (for example `75` when you only use T4). CUDA 13 dropped compute 6.0; `setup` removes 60 with a warning in that case.
* `paths.work_dir` defaults to `/tmp/iqp-work` (scratch). `output_root`, `state_dir`, `results_dir` and `logs_dir` live under `/kaggle/working/iqp`. All of them accept `${VAR:-default}`.
* `variants` (top level) restricts the whole run, `milu.variants` and `mt.variants` restrict the two server-based evaluations, and `--variants` on the command line narrows any of `quantize`, `run_kld`, `run_milu`, `run_mt` and `stream` further. Entries are exact names, `base`, or shell patterns such as `Q4_K_M__*` or `*__indic`.
* The stream step is also usable outside Kaggle (`python -m iqp stream --config config.yaml`) on any machine with limited disk.

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
* Gated dataset schemas were later confirmed against the real files with an authorized token. FLORES+: `{dev,devtest}/{code}.jsonl` with fields `id` (integer, from 0) and `text` plus metadata. MILU: `{Language}/{test,validation}-00000-of-00001.parquet` with `question`, `option1` to `option4`, `target` (such as `"option2"`), `is_translated`, `language`, `domain`, `subject` (Hindi test: about 14.8k rows). These match the configured field names. For IN22-Gen (fallback only) the file exists but the per-language column names are not confirmed. The CPU smoke runs used a local MILU-shaped file and local evaluation text. `validate_data` is still the check to run first.
* Kaggle mode, real CPU smoke run with `Qwen/Qwen3-0.6B` (2 vCPUs, no GPU; 2 languages; Q8_0, Q4_K_M with three arms, IQ2_M with two; 4 KLD chunks of 256): session 1 ran restore (nothing found), setup, download, convert (F16, safetensors deleted afterwards, 1.40 GiB freed), prepare_data, build_imatrix, then `stream --max-hours 0.04`, which finished Q4_K_M without an imatrix (64 s: quantize 31 s, KLD 10 s, MILU 5 s, MT 18 s), deleted its GGUF, and stopped before the next variant with the resume message. The persisted output was 58 MB (state 29 MB, of which 25 MB were cached llama.cpp binaries; each imatrix 2.4 MB; prepared data 0.4 MB; bundle 14 MB). Scratch was then removed, the build directory hidden, and the output attached read-only as an input. Session 2 restored 109 files, reused the cached binaries (setup 2 s), downloaded and converted again, skipped prepare_data and build_imatrix, recomputed the base logits, and streamed the remaining 6 variants without repeating the finished one. Measured per-variant times there: quantize 47 to 54 s with an imatrix, 111 to 117 s for IQ2_M, 6 s for Q8_0. Every quantized GGUF was deleted after its results were written, and both logits files were deleted after the last variant. Peak scratch was 3.03 GB (during convert: safetensors plus F16 GGUF); during stream it was the F16 GGUF (1.51 GB), the logits (0.31 GB) and one variant (at most 0.80 GB).
* Nothing in the Kaggle mode has run on a Kaggle GPU yet: the CUDA build for sm_60 and sm_75, T4 and P100 timings, actual Kaggle scratch size, and attaching notebook output as an input are unverified.
* Gemma 3 4B's `config.json` was read (architecture `Gemma3ForConditionalGeneration`, `torch_dtype` bfloat16, text model with 34 layers and hidden size 2560). It is a multimodal checkpoint; `convert_hf_to_gguf.py` writes the text model to the main GGUF (the vision tower goes to a separate mmproj file only when `--mmproj` is passed). The 4B weights were not downloaded or converted here.
