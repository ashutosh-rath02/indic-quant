# Pilot results: Gemma 3 4B (instruction-tuned), llama.cpp, Kaggle T4

Run `gemma3-4b-kaggle`, 19 variants, all completed in one session (about 4 hours of streaming after setup).
Reference model is F16. KLD is the mean KL divergence of the quantized model against F16 on FLORES+ devtest
(12 chunks of 512 tokens per language). Downstream scores use 200 MILU questions and 100 FLORES+ dev sentences
per language, so single-language differences below about 0.07 accuracy are within noise.

## 1. Quantization damage is uneven across languages

Mean KLD, static quantization (no importance matrix):

| Quant | English | Hindi | Tamil | Telugu | Bengali | Marathi |
|---|---|---|---|---|---|---|
| Q8_0 | 0.003 | 0.005 | 0.006 | 0.006 | 0.008 | 0.009 |
| Q6_K | 0.010 | 0.017 | 0.024 | 0.020 | 0.031 | 0.033 |
| Q5_K_M | 0.030 | 0.045 | 0.060 | 0.055 | 0.082 | 0.100 |
| Q4_K_M | 0.094 | 0.135 | 0.166 | 0.179 | 0.223 | 0.272 |
| Q3_K_M | 0.238 | 0.311 | 0.402 | 0.439 | 0.575 | 0.626 |
| Q2_K | 1.053 | 1.256 | 1.469 | 1.431 | 1.996 | 2.171 |

At Q4_K_M, Marathi loses 2.9 times and Bengali 2.4 times as much as English. The ratio is stable across quant levels
and remains 2.4 to 3.2 times at Q4_K_M with English calibration, so calibration does not remove it.

## 2. Calibration language (equal file size per quant type)

Mean KLD with an importance matrix. Columns: English, Hindi, Bengali, Tamil, Telugu, Marathi.

| Variant | eng | hin | ben | tam | tel | mar | Indic mean (5) |
|---|---|---|---|---|---|---|---|
| Q4_K_M en | 0.052 | 0.075 | 0.142 | 0.102 | 0.120 | 0.168 | 0.121 |
| Q4_K_M indic | 0.051 | 0.073 | 0.129 | 0.097 | 0.102 | 0.170 | 0.114 |
| Q4_K_M indic_no_tam | 0.050 | 0.078 | 0.137 | 0.092 | 0.105 | 0.167 | 0.116 |
| Q3_K_M en | 0.151 | 0.214 | 0.401 | 0.306 | 0.298 | 0.463 | 0.336 |
| Q3_K_M indic | 0.151 | 0.212 | 0.390 | 0.269 | 0.283 | 0.429 | 0.317 |
| Q3_K_M indic_no_tam | 0.155 | 0.208 | 0.376 | 0.265 | 0.286 | 0.429 | 0.313 |
| Q2_K en | 0.493 | 0.603 | 1.036 | 0.755 | 0.783 | 1.076 | 0.851 |
| Q2_K indic | 0.484 | 0.567 | 0.967 | 0.714 | 0.759 | 0.998 | 0.801 |
| Q2_K indic_no_tam | 0.492 | 0.556 | 0.966 | 0.718 | 0.752 | 1.017 | 0.802 |
| IQ2_M en | 0.467 | 0.614 | 1.012 | 0.846 | 0.816 | 1.169 | 0.891 |
| IQ2_M indic | 0.496 | 0.580 | 0.924 | 0.773 | 0.729 | 1.065 | 0.814 |
| IQ2_M indic_no_tam | 0.498 | 0.578 | 0.934 | 0.789 | 0.736 | 1.065 | 0.820 |

Findings:

* Any importance matrix beats none. At Q4_K_M, English calibration lowers the Indic mean KLD from 0.195 to 0.121 (38 percent).
* Indic calibration beats English calibration by 6 percent (Q4_K_M, Q3_K_M, Q2_K) to 9 percent (IQ2_M) on the Indic mean.
* Held-out Tamil transfers: with Tamil excluded from calibration, Tamil KLD is within noise of the arm that included it
  (Q2_K 0.718 vs 0.714) and well below English calibration (0.755).
* The cost lands on English at the lowest bit width: IQ2_M English KLD 0.467 (en) vs 0.496 (indic).

## 3. Downstream metrics (mean over the five Indic languages)

| Variant | MILU accuracy | chrF++ (eng to X) |
|---|---|---|
| F16 | 0.413 | 46.9 |
| Q8_0 static | 0.413 | 46.7 |
| Q4_K_M static | 0.403 | 46.4 |
| Q4_K_M en | 0.404 | 46.5 |
| Q4_K_M indic | 0.410 | 46.5 |
| Q4_K_M indic_no_tam | 0.409 | 46.4 |
| Q2_K static | 0.308 | 33.2 |
| Q2_K en | 0.353 | 42.8 |
| Q2_K indic | 0.338 | 44.1 |
| Q2_K indic_no_tam | 0.337 | 44.5 |

At Q4_K_M all arms are indistinguishable from F16. At Q2_K calibration recovers most of the loss, and Indic calibration
adds 1.3 to 1.7 chrF++ over English. MILU favors English calibration at Q2_K, but within noise.

## Verdict and limits

Pipeline verdict: partial signal. The per-language gap is large and consistent. The Indic calibration effect is real
but modest (6 to 9 percent KLD), with a small downstream gain visible only at 2-bit. One model, a small KLD token budget
and 200 MILU questions per language limit the claim; the Qwen3-4B replication (`configs/kaggle-qwen3-4b.yaml`) is the
next check.
