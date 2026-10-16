"""Run a MOSS-Transcribe-Diarize checkpoint over hospital/dev/ (the 10
real-recording, human-verified examples) and report MER. Standalone version
of GenerativeMERTrainer's eval loop, for scoring an already-saved checkpoint
directory (e.g. one of the LR-sweep outputs) rather than during training.

Usage:
    python eval_dev_mer.py --model_path /path/to/checkpoint --label lr_1e-5
"""

import argparse
import json
import os
import sys

import torch

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../moss_repeat_repro

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPRO_ROOT, "finetune_official"))
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages, generate_transcription
from mer_utils import compute_mer, compute_mer_by_language, normalize_for_eval
from production_pipeline import production_transcribe, real_production_transcribe

DEV_MANIFEST = os.path.join(REPRO_ROOT, "data", "dev_audio", "manifest.jsonl")
HOTWORDS_PATH = os.path.join(REPRO_ROOT, "data", "hotword_benchmark", "hotwords.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None,
                    help="Required unless --real_production_url is set (real-production mode ignores it "
                         "for inference; only --label matters then).")
    p.add_argument("--label", default=None)
    p.add_argument("--max_new_tokens", type=int, default=1200)
    p.add_argument("--use_hotword_prompt", action="store_true",
                    help="Append '熱詞提示：...' with this dev example's hotwords.json entry (only 7/10 dev "
                         "ids have one -- 738/895/837/743/739/926/998; the other 3 fall back to DEFAULT_PROMPT).")
    p.add_argument("--production_pipeline", action="store_true",
                    help="Use the real production pipeline (KWS-detected hotwords, not ground truth, + "
                         "hotword_correct post-processing) instead of --use_hotword_prompt's idealized "
                         "ground-truth-hint mode. Mutually exclusive with --use_hotword_prompt.")
    p.add_argument("--real_production_url", default=None,
                    help="If set (e.g. http://localhost:18000), transcribe via the REAL moss-asr-deploy "
                         "Docker stack (vLLM + proxy) instead of --production_pipeline's local HF-transformers "
                         "simulation. Requires the containers to already be running. Mutually exclusive with "
                         "--production_pipeline/--use_hotword_prompt -- the real stack IS the production "
                         "pipeline, its own KWS+hotword-correct logic always applies. --model_path is ignored "
                         "for inference in this mode (only used for the output --label default); the actual "
                         "model is whichever checkpoint the running vLLM container was started against.")
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=default_device)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="Repetition penalty for generation (e.g. 1.05 or 1.1).")
    args = p.parse_args()
    if args.real_production_url and (args.production_pipeline or args.use_hotword_prompt):
        raise SystemExit("--real_production_url is mutually exclusive with --production_pipeline/--use_hotword_prompt")
    if not args.real_production_url and not args.model_path:
        raise SystemExit("--model_path is required unless --real_production_url is set")

    label = args.label or (os.path.basename(os.path.normpath(args.model_path)) if args.model_path else "real_production")

    if args.real_production_url:
        model = processor = device = None
    else:
        from transformers import AutoModelForCausalLM
        from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

        device = torch.device(args.device)
        processor = MossTranscribeDiarizeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, dtype=torch.bfloat16
        ).to(device)
        if args.repetition_penalty != 1.0:
            model.generation_config.repetition_penalty = args.repetition_penalty
        model.eval()

    rows = [json.loads(l) for l in open(DEV_MANIFEST, encoding="utf-8") if l.strip()]
    hotwords_map = json.load(open(HOTWORDS_PATH, encoding="utf-8")) if args.use_hotword_prompt else {}

    total_errors = total_ref_len = 0
    lang_errors = {"zh": 0, "en": 0, "digit": 0}
    lang_ref_len = {"zh": 0, "en": 0, "digit": 0}
    per_example = []
    for row in rows:
        prompt = DEFAULT_PROMPT
        if args.use_hotword_prompt:
            kws = sorted(set(hotwords_map.get(row["id"], [])))
            if kws:
                prompt = DEFAULT_PROMPT + "热词提示：" + ", ".join(kws)
        if args.real_production_url:
            hyp_raw = real_production_transcribe(row["wav"], args.real_production_url)
        elif args.production_pipeline:
            hyp_raw = production_transcribe(model, processor, row["wav"], device, max_new_tokens=args.max_new_tokens)
        else:
            messages = build_transcription_messages(row["wav"], prompt)
            result = generate_transcription(model, processor, messages, device=device, dtype=torch.bfloat16,
                                             max_new_tokens=args.max_new_tokens)
            hyp_raw = result["text"]
        hyp_text = normalize_for_eval(hyp_raw, is_asr_output=True)
        ref_text = normalize_for_eval(row["text"])  # already untagged plain text
        errors, ref_len = compute_mer(ref_text, hyp_text)
        total_errors += errors
        total_ref_len += ref_len
        by_lang = compute_mer_by_language(ref_text, hyp_text)  # same default normalize=True as compute_mer() above
        for lang in ("zh", "en", "digit"):
            lang_errors[lang] += by_lang[lang][0]
            lang_ref_len[lang] += by_lang[lang][1]
        per_example.append({"id": row["id"], "mer": errors / ref_len if ref_len else None,
                             "ref": ref_text, "hyp": hyp_text,
                             "hyp_raw": hyp_raw})  # untouched, with [start][Sxx]...[end] tags intact

    mer = total_errors / total_ref_len if total_ref_len else 0.0
    mer_by_lang = {lang: (lang_errors[lang] / lang_ref_len[lang] if lang_ref_len[lang] else None)
                   for lang in ("zh", "en", "digit")}
    print(f"\n=== dev MER: {label} ===")
    print(f"n={len(rows)}  overall MER={mer:.2%}")
    for lang in ("zh", "en", "digit"):
        v = mer_by_lang[lang]
        print(f"  {lang} MER={v:.2%} (ref_len={lang_ref_len[lang]})" if v is not None else f"  {lang} MER=N/A (ref_len=0)")
    for ex in per_example:
        print(f"  [{ex['id']}] mer={ex['mer']:.2%}" if ex["mer"] is not None else f"  [{ex['id']}] mer=N/A")

    out_dir = os.path.join(REPRO_ROOT, "outputs", "dev_mer", label)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "overall_mer": mer, "mer_by_language": mer_by_lang,
                    "per_example": per_example}, f, ensure_ascii=False, indent=2)
    print(f"full results -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
