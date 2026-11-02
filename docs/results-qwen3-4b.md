# Pilot results: Qwen3-4B, llama.cpp, Kaggle T4

Run `qwen3-4b-kaggle`, same protocol as the Gemma 3 4B run (19 variants, F16 reference, 12 chunks of 512 tokens per
language for KLD, 200 MILU questions and 100 FLORES+ sentences per language). All 19 variants completed in one session,
about 4.9 hours in total. Columns below: English, Hindi, Bengali, Tamil, Telugu, Marathi.

## 1. Quantization damage by language (static, no importance matrix)

| Quant | eng | hin | ben | tam | tel | mar |
|---|---|---|---|---|---|---|
| Q8_0 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 |
| Q6_K | 0.007 | 0.007 | 0.006 | 0.007 | 0.005 | 0.008 |
| Q5_K_M | 0.018 | 0.022 | 0.019 | 0.017 | 0.014 | 0.022 |
| Q4_K_M | 0.048 | 0.054 | 0.054 | 0.052 | 0.044 | 0.064 |
| Q3_K_M | 0.204 | 0.201 | 0.210 | 0.204 | 0.152 | 0.259 |
| Q2_K | 0.868 | 0.900 | 0.868 | 0.854 | 0.702 | 1.135 |

Unlike Gemma 3 4B, Qwen3-4B shows almost no per-language gap in KLD. The worst language (Marathi) is 1.3 times English
at Q4_K_M and Q2_K, and Telugu is below English. Caution: Qwen3 splits Indic text into many tokens (6.7 tokens per word
for Marathi), and KLD is averaged per token, so many easy tokens can dilute the per-token damage. Downstream scores
(section 3) are the fairer cross-language check.

## 2. Calibration language

| Variant | eng | hin | ben | tam | tel | mar | Indic mean (5) |
|---|---|---|---|---|---|---|---|
| Q4_K_M en | 0.035 | 0.049 | 0.042 | 0.045 | 0.032 | 0.051 | 0.0438 |
| Q4_K_M indic | 0.038 | 0.046 | 0.037 | 0.039 | 0.029 | 0.051 | 0.0404 |
| Q4_K_M indic_no_tam | 0.040 | 0.047 | 0.039 | 0.039 | 0.029 | 0.052 | 0.0412 |
| Q3_K_M en | 0.119 | 0.144 | 0.151 | 0.150 | 0.107 | 0.178 | 0.146 |
| Q3_K_M indic | 0.129 | 0.127 | 0.130 | 0.135 | 0.100 | 0.156 | 0.130 |
| Q3_K_M indic_no_tam | 0.131 | 0.125 | 0.132 | 0.136 | 0.101 | 0.158 | 0.130 |
| Q2_K en | 0.355 | 0.444 | 0.455 | 0.465 | 0.389 | 0.564 | 0.463 |
| Q2_K indic | 0.397 | 0.416 | 0.443 | 0.430 | 0.373 | 0.573 | 0.447 |
| Q2_K indic_no_tam | 0.403 | 0.442 | 0.425 | 0.442 | 0.374 | 0.581 | 0.453 |
| IQ2_M en | 0.421 | 0.513 | 0.505 | 0.540 | 0.440 | 0.680 | 0.536 |
| IQ2_M indic | 0.497 | 0.453 | 0.460 | 0.469 | 0.373 | 0.564 | 0.464 |
| IQ2_M indic_no_tam | 0.505 | 0.446 | 0.465 | 0.466 | 0.369 | 0.552 | 0.458 |

* Indic calibration lowers the Indic mean KLD relative to English calibration by 8 percent (Q4_K_M), 11 percent (Q3_K_M),
  3.5 percent (Q2_K) and 13 percent (IQ2_M).
* Any importance matrix beats none: at Q2_K, English calibration cuts the Indic mean from 0.892 to 0.463 (48 percent).
* Held-out Tamil transfers (Q3_K_M 0.136 vs 0.135; IQ2_M 0.466 vs 0.469; English calibration 0.150 and 0.540).
* English pays at low bit width: IQ2_M English KLD 0.421 (en) vs 0.497 (indic), 18 percent worse.

## 3. Downstream (mean over the five Indic languages)

| Variant | MILU accuracy | chrF++ (eng to X) |
|---|---|---|
| F16 | 0.443 | 31.4 |
| Q8_0 static | 0.448 | 31.8 |
| Q4_K_M static | 0.439 | 29.9 |
| Q4_K_M en | 0.439 | 30.1 |
| Q4_K_M indic | 0.435 | 30.4 |
| Q4_K_M indic_no_tam | 0.432 | 30.1 |
| Q2_K static | 0.301 | 6.0 |
| Q2_K en | 0.363 | 13.2 |
| Q2_K indic | 0.344 | 14.0 |
| Q2_K indic_no_tam | 0.349 | 13.6 |

Qwen3-4B is a weak Indic translator even at F16 (chrF++ 31.4 against 46.9 for Gemma). Q4_K_M is indistinguishable from
F16. At Q2_K calibration recovers part of the loss and Indic calibration adds 0.4 to 0.9 chrF++ over English; MILU
favors English calibration, within noise (200 questions per language).
