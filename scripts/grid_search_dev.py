import json
import os
import sys
import torch

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPRO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPRO_ROOT, "finetune_official"))
sys.path.insert(0, os.path.join(REPRO_ROOT, "data", "hotword_benchmark"))

from transformers import AutoModelForCausalLM
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages, generate_transcription
from mer_utils import compute_mer, compute_mer_by_language, normalize_for_eval

DEV_MANIFEST = os.path.join(REPRO_ROOT, "data", "dev_audio", "manifest.jsonl")
MODEL_PATH = os.path.join(REPRO_ROOT, "model", "checkpoint-8000-deploy")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading processor and model on {device}...")
    processor = MossTranscribeDiarizeProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    rows = [json.loads(l) for l in open(DEV_MANIFEST, encoding="utf-8") if l.strip()]
    penalties = [1.00, 1.03, 1.05, 1.08, 1.10, 1.12]

    results = []
    print("\n--- Starting Dev Set Repetition Penalty Sweep ---")
    for rp in penalties:
        model.generation_config.repetition_penalty = rp
        total_errors = total_ref_len = 0
        lang_errors = {"zh": 0, "en": 0, "digit": 0}
        lang_ref_len = {"zh": 0, "en": 0, "digit": 0}

        for row in rows:
            messages = build_transcription_messages(row["wav"], DEFAULT_PROMPT)
            res = generate_transcription(model, processor, messages, device=device, dtype=torch.bfloat16, max_new_tokens=1200)
            hyp_text = normalize_for_eval(res["text"], is_asr_output=True)
            ref_text = normalize_for_eval(row["text"])
            errs, rlen = compute_mer(ref_text, hyp_text)
            total_errors += errs
            total_ref_len += rlen
            by_lang = compute_mer_by_language(ref_text, hyp_text)
            for lang in ("zh", "en", "digit"):
                lang_errors[lang] += by_lang[lang][0]
                lang_ref_len[lang] += by_lang[lang][1]

        overall_mer = total_errors / total_ref_len if total_ref_len else 0.0
        zh_mer = lang_errors["zh"] / lang_ref_len["zh"] if lang_ref_len["zh"] else 0.0
        en_mer = lang_errors["en"] / lang_ref_len["en"] if lang_ref_len["en"] else 0.0
        d_mer = lang_errors["digit"] / lang_ref_len["digit"] if lang_ref_len["digit"] else 0.0
        results.append({
            "rp": rp,
            "overall_mer": overall_mer,
            "zh_mer": zh_mer,
            "en_mer": en_mer,
            "digit_mer": d_mer
        })
        print(f"rp={rp:.2f} | Overall MER: {overall_mer:.2%} | zh: {zh_mer:.2%} | en: {en_mer:.2%} | digit: {d_mer:.2%}")

    print("\n=== Summary Table (10 Dev Examples) ===")
    print(f"{'Penalty':<8} | {'Overall MER':<12} | {'zh MER':<10} | {'en MER':<10} | {'digit MER':<10}")
    print("-" * 58)
    for r in results:
        print(f"{r['rp']:<8.2f} | {r['overall_mer']:<12.2%} | {r['zh_mer']:<10.2%} | {r['en_mer']:<10.2%} | {r['digit_mer']:<10.2%}")

if __name__ == "__main__":
    main()
