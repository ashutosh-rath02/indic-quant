"""Steps run_milu and run_mt (both use a managed llama-server per variant)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from iqp import textproc as tp
from iqp.scoring import bootstrap_ci, chrf_pp, milu_score_response
from iqp.server import LlamaServer
from iqp.util import LOG, StepError, atomic_write_json, read_jsonl, write_jsonl


def _server_cls(cfg):
    if cfg.dry_run:
        from iqp.fakes.server import FakeLlamaServer
        return FakeLlamaServer
    return LlamaServer


# ------------------------------------------------------------------ MILU

def milu_request(prompt: str, cfg) -> dict:
    """Raw-text completion (no chat template), one token, logits before sampling."""
    return {"prompt": prompt, "n_predict": 1, "n_probs": int(cfg.raw["milu"]["n_probs"]), "temperature": 0.0,
            "cache_prompt": bool(cfg.raw["server"]["cache_prompt"]), "seed": int(cfg.raw["server"]["seed"]),
            "post_sampling_probs": False}


def step_run_milu(cfg, force: bool = False) -> None:
    m = cfg.raw["milu"]
    if not m.get("enabled", True):
        LOG.info("MILU disabled")
        return
    raw = cfg.raw_results_dir / "milu"
    Server = _server_cls(cfg)
    for v in cfg.select_variants(m.get("variants", "all")):
        todo = [c for c in cfg.langs if force or not (raw / v.name / f"{c}.json").exists()]
        if not todo:
            LOG.info("skip (exists): MILU %s", v.name)
            continue
        model = cfg.gguf_path(v)
        if not model.exists():
            LOG.warning("missing %s; skipping MILU", model.name)
            continue
        with Server(cfg, model, cfg.logs_dir / "run_milu" / f"server_{v.name}.log") as srv:
            for code in todo:
                path = cfg.milu_path(code, "test")
                if not path.exists():
                    raise StepError(f"MILU prompts missing: {path}; run prepare_data")
                items = read_jsonl(path)

                def one(it):
                    r = milu_score_response(srv.post("/completion", milu_request(it["prompt"], cfg)))
                    return {"idx": it["idx"], "answer": it["answer"], "pred": r["pred"],
                            "correct": int(r["pred"] == it["answer"]), "letters_present": r["letters_present"],
                            "k_shot": it["k_shot"], "subject": it.get("subject")}

                with ThreadPoolExecutor(int(cfg.raw["server"]["client_threads"])) as ex:
                    rows = list(ex.map(one, items))
                write_jsonl(raw / v.name / f"{code}.items.jsonl", rows)
                acc = sum(r["correct"] for r in rows) / len(rows)
                lo, hi = bootstrap_ci([r["correct"] for r in rows], seed=1)
                cov = sum(r["letters_present"] == 4 for r in rows) / len(rows)
                atomic_write_json(raw / v.name / f"{code}.json", {
                    "variant": v.name, "language": code, "n": len(rows), "accuracy": acc,
                    "ci_low": lo, "ci_high": hi, "all_letters_in_topn": cov,
                    "k_shot_mean": sum(r["k_shot"] for r in rows) / len(rows)})
                LOG.info("MILU %-22s %s acc=%.3f [%.3f, %.3f] n=%d", v.name, code, acc, lo, hi, len(rows))


# ------------------------------------------------------------------ MT

def mt_request(cfg, source: str, target_name: str) -> dict:
    mt = cfg.raw["mt"]
    content = mt["prompt"].format(source=source, target=target_name)
    return {"messages": [{"role": "user", "content": content}], "temperature": 0.0,
            "max_tokens": int(mt["max_tokens"]), "seed": int(cfg.raw["server"]["seed"]),
            "cache_prompt": bool(cfg.raw["server"]["cache_prompt"]),
            # Qwen3 thinking mode off; ignored by templates that do not use the flag.
            "chat_template_kwargs": {"enable_thinking": False}}


def step_run_mt(cfg, force: bool = False) -> None:
    mt = cfg.raw["mt"]
    if not mt.get("enabled", True):
        LOG.info("MT disabled")
        return
    rows = read_jsonl(cfg.eval_mt_path())
    targets = [c for c in cfg.langs if c != "eng_Latn"]
    raw = cfg.raw_results_dir / "mt"
    Server = _server_cls(cfg)
    for v in cfg.select_variants(mt.get("variants", "all")):
        todo = [c for c in targets if force or not (raw / v.name / f"{c}.json").exists()]
        if not todo:
            LOG.info("skip (exists): MT %s", v.name)
            continue
        model = cfg.gguf_path(v)
        if not model.exists():
            LOG.warning("missing %s; skipping MT", model.name)
            continue
        with Server(cfg, model, cfg.logs_dir / "run_mt" / f"server_{v.name}.log") as srv:
            for code in todo:
                name = cfg.lang_name(code)

                def one(r):
                    resp = srv.post("/v1/chat/completions", mt_request(cfg, r["eng_Latn"], name))
                    raw_out = resp["choices"][0]["message"].get("content") or ""
                    return {"id": r["id"], "src": r["eng_Latn"], "ref": r[code], "raw": raw_out,
                            "hyp": tp.clean_translation(raw_out),
                            "finish_reason": resp["choices"][0].get("finish_reason")}

                with ThreadPoolExecutor(int(cfg.raw["server"]["client_threads"])) as ex:
                    outs = list(ex.map(one, rows))
                write_jsonl(raw / v.name / f"{code}.items.jsonl", outs)
                score = chrf_pp([o["hyp"] for o in outs], [o["ref"] for o in outs])
                atomic_write_json(raw / v.name / f"{code}.json", {
                    "variant": v.name, "language": code, "n": len(outs), "chrfpp": score,
                    "truncated": sum(o["finish_reason"] == "length" for o in outs)})
                LOG.info("MT %-22s eng->%s chrF++=%.2f", v.name, code, score)
