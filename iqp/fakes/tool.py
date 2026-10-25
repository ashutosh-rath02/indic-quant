"""Fake llama.cpp command line tools for --dry-run.

Installed as small wrapper scripts named llama-perplexity, llama-quantize, llama-imatrix,
llama-tokenize and convert_hf_to_gguf.py. Each one accepts the same flags the pipeline passes
to the real tool and prints output in the real tool's format (taken from captured runs), so
the real parsers are exercised end to end. Model "GGUF" files are small JSON documents.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

from iqp.fakes.common import QTYPE_BPW, h01, toy_kld, toy_token_count

FAKE_VOCAB = 4096


def _args(argv: list[str]) -> tuple[dict, list[str]]:
    flags: dict[str, str | bool] = {}
    pos: list[str] = []
    bool_flags = {"--kl-divergence", "--show-count", "--ids", "--no-bos", "--log-disable", "--jinja", "--no-webui"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in bool_flags:
            flags[a] = True
            i += 1
        elif a.startswith("-") and not _is_number(a):
            flags[a] = argv[i + 1] if i + 1 < len(argv) else ""
            i += 2
        else:
            pos.append(a)
            i += 1
    return flags, pos


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _model(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        print(f"llama_model_load: error loading model: {path}", file=sys.stderr)
        sys.exit(1)


def _lang_from_path(p: str) -> str:
    return Path(p).stem


def perplexity(argv):
    f, _ = _args(argv)
    m = _model(f["-m"])
    ctx = int(f.get("-c", 512))
    if f.get("--kl-divergence"):
        lf = f["--kl-divergence-base"]
        with open(lf, "rb") as fh:
            assert fh.read(8) == b"_logits_"
            n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", fh.read(12))
            trailer = json.loads(fh.read().decode())
        lang = trailer["language"]
        print(f"0.00.600.000 I kl_divergence: computing over {n_chunk} chunks, n_ctx={n_ctx}, batch_size={ctx}, n_seq=1")
        count = n_chunk * (n_ctx - 1 - n_ctx // 2)
        if count < 100:
            return 0
        k = toy_kld(m["qtype"], m["arm"], lang.split("_")[0], m.get("model", ""))
        ppl_b = trailer["ppl"]
        lr = 0.55 * k
        ppl_q = ppl_b * math.exp(lr)
        same = max(30.0, 100.0 * math.exp(-1.15 * math.sqrt(k)))
        unc = k / math.sqrt(count) * 2.2
        print("\nchunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p")
        for c in range(1, n_chunk + 1):
            print(f"{c:4d}      {ppl_q:8.4f} ±    {ppl_q*0.1:6.4f}       {lr:7.5f} ±    {lr*0.3:7.5f}       "
                  f"{k:7.5f} ±    {unc:7.5f}     {100*math.sqrt(k)/3:5.3f} ±  0.500 %    {same:6.3f} ±  1.000 %")
        print("\n====== Perplexity statistics ======")
        print(f"Mean PPL(Q)                   : {ppl_q:10.6f} ± {ppl_q*0.05:10.6f}")
        print(f"Mean PPL(base)                : {ppl_b:10.6f} ± {ppl_b*0.05:10.6f}")
        print("Cor(ln(PPL(Q)), ln(PPL(base))):  98.50%")
        print(f"Mean ln(PPL(Q)/PPL(base))     : {lr:10.6f} ± {lr*0.2:10.6f}")
        print(f"Mean PPL(Q)/PPL(base)         : {math.exp(lr):10.6f} ± {math.exp(lr)*lr*0.2:10.6f}")
        print(f"Mean PPL(Q)-PPL(base)         : {ppl_q-ppl_b:10.6f} ± {(ppl_q-ppl_b)*0.3:10.6f}")
        print("\n====== KL divergence statistics ======")
        print(f"Mean    KLD: {k:10.6f} ± {unc:10.6f}")
        for lab, mult in (("Maximum", 30), ("99.9%  ", 18), ("99.0%  ", 8.5), ("95.0%  ", 3.8), ("90.0%  ", 2.5),
                          ("Median ", 0.42), ("10.0%  ", 0.04), (" 5.0%  ", 0.015), (" 1.0%  ", 0.002),
                          (" 0.1%  ", 0.0008), ("Minimum", 0.0006)):
            print(f"{lab} KLD: {k*mult:10.6f}")
        print("\n====== Token probability statistics ======")
        print(f"Mean    Δp: {-10*k:6.3f} ± 0.300 %")
        print(f"RMS Δp    : {100*math.sqrt(k)/3:6.3f} ± 0.500 %")
        print(f"Same top p: {same:6.3f} ± 1.000 %")
        return 0
    text = Path(f["-f"]).read_text(encoding="utf-8")
    n_tok = toy_token_count(text)
    if n_tok < 2 * ctx:
        print(f"perplexity: you need at least {2*ctx} tokens to evaluate perplexity with a context of {ctx}")
        print(f"perplexity: the data file you provided tokenizes to only {n_tok} tokens")
        return 1
    n_chunk = min(int(f.get("--chunks", -1)) if int(f.get("--chunks", -1)) > 0 else 10**9, n_tok // ctx)
    lang = _lang_from_path(f["-f"])
    ppl = 6.0 + 12.0 * h01(lang, m.get("model", ""))
    print(f"0.00.400.000 I perplexity: calculating perplexity over {n_chunk} chunks, n_ctx={ctx}, batch_size={ctx}, n_seq=1")
    lf = f.get("--kl-divergence-base")
    if lf:
        with open(lf, "wb") as fh:
            fh.write(b"_logits_")
            fh.write(struct.pack("<Iii", ctx, FAKE_VOCAB, n_chunk))
            fh.write(json.dumps({"language": lang, "ppl": ppl}).encode())
    print("".join(f"[{i}]{ppl:.4f}," for i in range(1, n_chunk + 1)))
    print(f"0.01.000.000 I Final estimate: PPL = {ppl:.4f} +/- {ppl*0.08:.5f}")
    return 0


def quantize(argv):
    f, pos = _args(argv)
    src, dst, qtype = pos[0], pos[1], pos[2]
    m = _model(src)
    needs = qtype.startswith("IQ1") or qtype.startswith("IQ2") or qtype in ("IQ3_XXS", "Q2_K_S")
    if needs and "--imatrix" not in f:
        print("\n============================================================================")
        print(" ERROR: this quantization requires an importance matrix!")
        print("============================================================================")
        return 1
    arm = "static"
    if "--imatrix" in f:
        arm = json.loads(Path(f["--imatrix"]).read_text())["arm"]
    bpw = QTYPE_BPW[qtype] + 0.02 * (h01(qtype, arm) - 0.5)
    n_params = m.get("n_params", 3.88e9)
    out = {"fake_gguf": True, "model": m.get("model"), "qtype": qtype, "arm": arm, "n_params": n_params}
    Path(dst).write_text(json.dumps(out))
    orig_mib = n_params * 16 / 8 / 2**20
    print(f"llama_model_quantize_impl: model size  = {orig_mib:8.2f} MiB (16.00 BPW)")
    print(f"llama_model_quantize_impl: quant size  = {n_params*bpw/8/2**20:8.2f} MiB ({bpw:.2f} BPW)")
    print("\nllama_quantize: quantize time =  1000.00 ms")
    return 0


def imatrix(argv):
    f, _ = _args(argv)
    _model(f["-m"])
    ctx = int(f.get("-c", 512))
    text = Path(f["-f"]).read_text(encoding="utf-8")
    n_tok = toy_token_count(text)
    if n_tok < 2 * ctx:
        print(f"compute_imatrix: you need at least {2*ctx} tokens for a context of {ctx} tokens")
        return 1
    want = int(f.get("--chunks", -1))
    n_chunk = n_tok // ctx if want <= 0 else min(want, n_tok // ctx)
    print("0.00.430.000 I compute_imatrix: tokenizing the input ..")
    print(f"0.00.440.000 I compute_imatrix: computing over {n_chunk} chunks, n_ctx={ctx}, batch_size={ctx}, n_seq=1")
    print("".join(f"[{i}]8.0000," for i in range(1, n_chunk + 1)))
    print("Final estimate: PPL = 8.0000 +/- 0.50000")
    Path(f["-o"]).write_text(json.dumps({"fake_imatrix": True, "arm": Path(f["-f"]).stem, "chunks": n_chunk}))
    return 0


def tokenize(argv):
    f, _ = _args(argv)
    _model(f["-m"])
    text = Path(f["-f"]).read_text(encoding="utf-8") if "-f" in f else f.get("-p", "")
    n = toy_token_count(text)
    if f.get("--ids"):
        print("[" + ", ".join(str(int(h01(i) * FAKE_VOCAB)) for i in range(n)) + "]")
    if f.get("--show-count"):
        print(f"Total number of tokens: {n}")
    return 0


def convert(argv):
    f, pos = _args(argv)
    hf = Path(pos[0])
    c = json.loads((hf / "config.json").read_text())
    Path(f["--outfile"]).write_text(json.dumps({"fake_gguf": True, "model": c.get("_name_or_path", "fake"),
                                                "qtype": f["--outtype"].upper(), "arm": "base",
                                                "n_params": c.get("n_params", 3.88e9)}))
    print(f"INFO:hf-to-gguf:Model successfully exported to {f['--outfile']}")
    return 0


TOOLS = {"llama-perplexity": perplexity, "llama-quantize": quantize, "llama-imatrix": imatrix,
         "llama-tokenize": tokenize, "convert_hf_to_gguf.py": convert}


def main():
    name = os.environ.get("IQP_FAKE_TOOL") or Path(sys.argv[0]).name
    sys.exit(TOOLS[name](sys.argv[1:]))


if __name__ == "__main__":
    main()
