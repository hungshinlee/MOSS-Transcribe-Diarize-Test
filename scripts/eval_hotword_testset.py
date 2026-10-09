"""Run a MOSS-Transcribe-Diarize checkpoint over the hotword_benchmark test
set. This script only does two things: (1) generate ASR output for each
benchmark audio and write it into the exact format hotword_benchmark's own
evaluate.py --candidate expects (<candidate_dir>/<audio_id>/transcription.json,
{"text": ...}), and (2) call evaluate.py's own evaluate() function, UNMODIFIED,
to score it. All scoring logic lives in evaluate.py -- there is exactly one
evaluation program, and it is not duplicated or reimplemented here.

Reports the three fields evaluate()'s summary dict already contains:
  - keyword_recall     <- summary["recall"]: substring hit rate vs
                           hotwords.json, occurrence-aware (a keyword required
                           3x only gets full credit if matched 3x).
  - mer                <- summary["mer"]: vs pseudo_transcripts.json
                           (reference-only, not ground truth).
  - keyword_false_alarm <- summary["excess"]: occurrence count (not deduped
                           to a set) of a required keyword being matched more
                           times than hotwords.json asked for -- evaluate.py's
                           own "疑似誤報" figure, used as-is.

evaluate.py's separate --predicted-keywords path (kw_precision/kw_recall/
kw_f1) is NOT used: that path scores a system that explicitly predicts a
keyword-spotting list, comparing deduped sets. Our model only outputs free
transcribed text, so that path doesn't apply to it.

This is a *separate, purely observational* test set from hospital/dev/ (the
10-example real-recording dev set used to actually pick a checkpoint) --
"跟 dev 一樣的時間跑，但就只要算指標讓我觀察效果就好". It is NOT used to choose
a checkpoint, just logged alongside it so trends are visible.

run_hotword_eval() is the reusable core (takes an already-loaded model +
processor, batches generation via batch_generate.batched_transcribe) --
called both by this file's CLI and by finetune_with_eval.py's
GenerativeMERTrainer for the periodic in-training observational pass.

Data source: /home/wago/meetingminutes/hot_word/hotword_benchmark/
  - audio/<id>.wav           -- 66 primary audio files (variant clips like
                                 503_Transamine.wav are ignored; only the
                                 plain <id>.wav matching hotwords.json's keys
                                 is used)
  - hotwords.json            -- ground-truth target hotwords per audio_id
  - pseudo_transcripts.json  -- ASR-generated (not human-verified) reference
                                 text, used only for the MER "抓漏參考" metric

Usage:
    python eval_hotword_testset.py --model_path /path/to/checkpoint --label baseline
    python eval_hotword_testset.py --model_path outputs/full_run/checkpoint-500 --label epoch2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../moss_repeat_repro

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPRO_ROOT, "finetune_official"))

sys.path.insert(0, os.path.join(REPRO_ROOT, "data", "hotword_benchmark"))
import evaluate as hw_eval  # noqa: E402  (module name matches upstream file evaluate.py)

from batch_generate import batched_transcribe  # noqa: E402
from mer_utils import compute_mer_by_language, strip_tags  # noqa: E402  (shared with finetune_with_eval.py's compute_metrics)
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT  # noqa: E402
from production_pipeline import _apply_hotword_correction, _CC_S2TW, _load_samples_16k, detect_keywords_from_samples, real_production_transcribe  # noqa: E402

BENCHMARK_ROOT = os.path.join(REPRO_ROOT, "data", "hotword_benchmark")
AUDIO_DIR = os.path.join(BENCHMARK_ROOT, "audio")
HOTWORDS_PATH = os.path.join(BENCHMARK_ROOT, "hotwords.json")
ALL_HOTWORDS_PATH = os.path.join(BENCHMARK_ROOT, "all_hotwords.json")
PSEUDO_PATH = os.path.join(BENCHMARK_ROOT, "pseudo_transcripts.json")
TRUE_TRANSCRIPTS_PATH = os.path.join(BENCHMARK_ROOT, "true_transcripts", "true_transcripts.json")

_HOTWORDS_MAP = None
_ALL_HOTWORDS = None
_PSEUDO_MAP = None
_TRUE_MAP = None
_AUDIO_IDS = None
_SIBLING_MAP = None


def _load_benchmark_data():
    global _HOTWORDS_MAP, _ALL_HOTWORDS, _PSEUDO_MAP, _TRUE_MAP, _AUDIO_IDS, _SIBLING_MAP
    if _HOTWORDS_MAP is None:
        _HOTWORDS_MAP = json.load(open(HOTWORDS_PATH, encoding="utf-8"))
        _ALL_HOTWORDS = json.load(open(ALL_HOTWORDS_PATH, encoding="utf-8"))
        _PSEUDO_MAP = json.load(open(PSEUDO_PATH, encoding="utf-8"))
        # evaluate.py's evaluate() now also takes a true_map (human-verified
        # transcripts, only 10/66 currently annotated) -- an upstream addition
        # to this shared, not-our-own file. Missing ids just get "" (evaluate.py
        # treats that as "no true-transcript reference", contributes (0,0) to
        # the true_transcribe MER, doesn't affect the pseudo-based mer/recall).
        _TRUE_MAP = json.load(open(TRUE_TRANSCRIPTS_PATH, encoding="utf-8")) if os.path.exists(TRUE_TRANSCRIPTS_PATH) else {}
        _AUDIO_IDS = sorted(_HOTWORDS_MAP.keys(), key=int)
        _SIBLING_MAP = hw_eval.build_sibling_superset_map(_HOTWORDS_MAP)
    return _HOTWORDS_MAP, _ALL_HOTWORDS, _PSEUDO_MAP, _TRUE_MAP, _AUDIO_IDS, _SIBLING_MAP


def run_hotword_eval(model, processor, device, *, batch_size: int = 8, max_new_tokens: int = 1200,
                      candidate_dir: str = None, max_audio_duration_sec: float = 180.0,
                      use_hotword_prompt: bool = False, production_pipeline: bool = False,
                      real_production_url: str = None, repetition_penalty: float = 1.0) -> dict:
    """Batched inference + scoring over the 66-example hotword benchmark.
    Returns {"keyword_recall", "mer", "keyword_false_alarm", "n_audio"}.
    Writes per-audio transcription.json under candidate_dir if given (kept
    for inspection); uses a throwaway temp dir otherwise.

    A handful of these recordings are much longer than anything in our own
    training/dev data (up to ~11 minutes vs. our 120s training cap) --
    batching one of those in with others padded it up to ~25k+ audio tokens
    and crashed the SDPA prefill kernel outright (reproduced twice). Since
    this benchmark is purely observational, audio over max_audio_duration_sec
    is skipped rather than risk crashing the whole eval; evaluate.py already
    treats a missing transcription.json as "no output" (a recall miss), so
    no other special-casing is needed. Skipped ids are printed for visibility.

    use_hotword_prompt=True appends the official hotword-hint syntax (see
    README's "Custom Prompt and Hotwords" section: "...熱詞提示：熱詞1, 熱詞2")
    to DEFAULT_PROMPT, using that audio's own target hotwords from
    hotwords.json -- i.e. the actual deployment-time usage pattern, not the
    fixed no-hint prompt every training example used. Lets us check whether
    this capability (which the base model documents support for) survived
    fine-tuning on a single fixed prompt.
    """
    hotwords_map, _all_hotwords, pseudo_map, true_map, audio_ids, _sibling_map = _load_benchmark_data()

    if candidate_dir is None:
        import tempfile
        candidate_dir = tempfile.mkdtemp(prefix="hotword_eval_")
    os.makedirs(candidate_dir, exist_ok=True)

    kept_ids, audio_paths = [], []
    skipped = []
    for aid in audio_ids:
        path = os.path.join(AUDIO_DIR, f"{aid}.wav")
        if sf.info(path).duration > max_audio_duration_sec:
            skipped.append(aid)
            continue
        kept_ids.append(aid)
        audio_paths.append(path)
    if skipped:
        print(f"[eval_hotword_testset] skipping {len(skipped)} audio > {max_audio_duration_sec}s: {skipped}")

    if real_production_url:
        # The real stack does its own KWS + prompt-building + hotword-correct
        # internally (server.py's _transcribe()) -- no local prompt building
        # or post-processing needed, just one HTTP call per audio file. The
        # proxy only accepts WAV; AUDIO_DIR's files already are .wav.
        hyps = [real_production_transcribe(path, real_production_url) for path in tqdm(audio_paths, desc="real production")]
    else:
        prompts = None
        if production_pipeline:
            # Real KWS detection per audio (proxy/kws.py, unmodified) instead of
            # the ground-truth hotwords_map -- reflects what a user would actually
            # see, not an idealized "given the right answer" upper bound.
            prompts = []
            for path in audio_paths:
                samples, sr = _load_samples_16k(path)
                detected = detect_keywords_from_samples(samples, sr)
                keywords = sorted({k for k, _offset in detected})
                prompts.append(DEFAULT_PROMPT + ("热词提示：" + ", ".join(keywords) if keywords else ""))
        elif use_hotword_prompt:
            prompts = []
            for aid in kept_ids:
                kws = sorted(set(hotwords_map.get(aid, [])))
                prompt = DEFAULT_PROMPT + ("热词提示：" + ", ".join(kws) if kws else "")
                prompts.append(prompt)

        hyps = batched_transcribe(model, processor, audio_paths, prompts, max_new_tokens=max_new_tokens,
                                   batch_size=batch_size, device=device, repetition_penalty=repetition_penalty)
        if production_pipeline:
            # Real post-processing (server.py order: s2tw convert, then hotword_correct
            # per-segment, tags preserved) -- applied AFTER generation, before scoring.
            hyps = [_apply_hotword_correction(_CC_S2TW.convert(hyp)) for hyp in hyps]

    lang_errors = {"zh": 0, "en": 0, "digit": 0}
    lang_ref_len = {"zh": 0, "en": 0, "digit": 0}
    for aid, hyp in zip(kept_ids, hyps):
        text = strip_tags(hyp)
        out_dir = os.path.join(candidate_dir, aid)
        os.makedirs(out_dir, exist_ok=True)
        # evaluate.py's own contract only reads "text" (untagged) -- "text_raw"
        # is extra, ignored by evaluate.py, kept for manual timestamp/speaker
        # verification against the original audio.
        with open(os.path.join(out_dir, "transcription.json"), "w", encoding="utf-8") as f:
            json.dump({"text": text, "text_raw": hyp}, f, ensure_ascii=False)
        # Additive, does NOT feed into or replace evaluate()'s own scoring below
        # -- pseudo_map is ASR-generated (not human-verified) reference text,
        # same caveat that already applies to evaluate()'s own "mer" number.
        pseudo_ref = pseudo_map.get(aid)
        if pseudo_ref:
            by_lang = compute_mer_by_language(pseudo_ref, text)
            for lang in ("zh", "en", "digit"):
                lang_errors[lang] += by_lang[lang][0]
                lang_ref_len[lang] += by_lang[lang][1]

    summary, _ = hw_eval.evaluate(Path(candidate_dir), hotwords_map, pseudo_map, true_map)

    mer_by_language = {lang: (lang_errors[lang] / lang_ref_len[lang] if lang_ref_len[lang] else None)
                        for lang in ("zh", "en", "digit")}

    return {
        "keyword_recall": summary["recall"],
        "mer": summary["mer"],
        "mer_by_language": mer_by_language,  # vs pseudo_transcripts.json, same reference-only caveat as "mer"
        "keyword_false_alarm": summary["excess"],
        "n_audio": summary["audio_count"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default=None,
                    help="checkpoint dir or HF hub id. Required unless --real_production_url is set.")
    p.add_argument("--label", default=None, help="tag for this run, used in the candidate_dir name")
    p.add_argument("--output_root", default=os.path.join(REPRO_ROOT, "outputs", "hotword_testset"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1200)
    p.add_argument("--max_audio_duration_sec", type=float, default=180.0,
                    help="Skip audio longer than this. Pass a very large value (e.g. 1e9) to include everything.")
    p.add_argument("--use_hotword_prompt", action="store_true",
                    help="Append each audio's own target hotwords to the prompt using the official "
                         "'熱詞提示：' hint syntax, instead of the fixed no-hint DEFAULT_PROMPT.")
    p.add_argument("--production_pipeline", action="store_true",
                    help="Use the real production pipeline (KWS-detected hotwords, not ground truth, + "
                         "hotword_correct post-processing) instead of --use_hotword_prompt's idealized "
                         "ground-truth-hint mode. Mutually exclusive with --use_hotword_prompt.")
    p.add_argument("--real_production_url", default=None,
                    help="If set (e.g. http://localhost:18000), transcribe via the REAL moss-asr-deploy "
                         "Docker stack (vLLM + proxy) instead of --production_pipeline's local HF-transformers "
                         "simulation. Requires the containers to already be running. Mutually exclusive with "
                         "--production_pipeline/--use_hotword_prompt. --model_path/--batch_size are ignored "
                         "for inference in this mode.")
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=default_device)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="Repetition penalty for generation (e.g. 1.05 or 1.1 to prevent repeat degeneration loops).")
    args = p.parse_args()
    if args.real_production_url and (args.production_pipeline or args.use_hotword_prompt):
        raise SystemExit("--real_production_url is mutually exclusive with --production_pipeline/--use_hotword_prompt")
    if not args.real_production_url and not args.model_path:
        raise SystemExit("--model_path is required unless --real_production_url is set")

    label = args.label or (os.path.basename(os.path.normpath(args.model_path)) if args.model_path else "real_production")
    candidate_dir = os.path.join(args.output_root, label)

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
        model.eval()

    result = run_hotword_eval(model, processor, device, batch_size=args.batch_size, candidate_dir=candidate_dir,
                               max_new_tokens=args.max_new_tokens, max_audio_duration_sec=args.max_audio_duration_sec,
                               use_hotword_prompt=args.use_hotword_prompt, production_pipeline=args.production_pipeline,
                               real_production_url=args.real_production_url, repetition_penalty=args.repetition_penalty)

    print(f"\n=== hotword test set: {label} ===")
    print(f"audio count: {result['n_audio']}")
    print(f"keyword recall (substring vs hotwords.json): {result['keyword_recall']:.2%}")
    print(f"MER (vs pseudo_transcripts.json, reference-only, not ground truth): {result['mer']:.2%}")
    for lang in ("zh", "en", "digit"):
        v = result["mer_by_language"][lang]
        print(f"  {lang} MER (vs pseudo_transcripts.json): {v:.2%}" if v is not None else f"  {lang} MER: N/A (ref_len=0)")
    print(f"keyword false alarm (occurrence count, not deduped): {result['keyword_false_alarm']}")

    with open(os.path.join(candidate_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, **result}, f, ensure_ascii=False, indent=2)
    print(f"full results -> {candidate_dir}/")


if __name__ == "__main__":
    main()
