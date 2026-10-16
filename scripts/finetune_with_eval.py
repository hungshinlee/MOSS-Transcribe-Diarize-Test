#!/usr/bin/env python3
"""Wrapper around the official finetune.py that adds eval_dataset support
(the official script only wires up train_dataset -- see ../official/finetune.py)
plus a real generative MER metric (actual model.generate(), not teacher-forced)
so checkpoint quality can be tracked/selected by something more meaningful than
raw cross-entropy loss.

Reuses the official ScriptArguments/ConversationDataset/DataCollator classes
as-is so behavior stays identical to the official script for everything else.

Eval always goes through GenerativeMERTrainer: real autoregressive generation
on eval_dataset (meant to be the 10-example real dev set, hence cheap even
though generation is much slower per-example than a single forward pass).
Teacher-forced MER was removed -- it's an optimistic proxy (model always sees
ground-truth history, unlike real generation) and does not reflect actual
inference-time behavior.

With --metric_for_best_model mer --greater_is_better False
--load_best_model_at_end --save_total_limit N, Trainer's own checkpoint
rotation keeps the best-by-MER checkpoint safe automatically -- no separate
post-hoc pruning needed.

Usage (same CLI as finetune.py, plus --val_jsonl):
    python scripts/finetune_with_eval.py \
      --train_jsonl data/train_full.jsonl --val_jsonl data/dev.jsonl \
      --model_name_or_path /home/wago/moss-asr/models/MOSS-Transcribe-Diarize \
      --output_dir outputs/full_run/1e-5_v2 \
      --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --max_length 6144 \
      --learning_rate 1e-5 --max_steps 8000 \
      --eval_strategy steps --eval_steps 200 --save_strategy steps --save_steps 200 \
      --save_only_model True \
      --metric_for_best_model mer --greater_is_better False --load_best_model_at_end \
      --hotword_test_enabled True --hotword_batch_size 1 --hotword_max_audio_duration_sec 1e9 \
      --gen_batch_size 8 \
      --bf16 --gradient_checkpointing --report_to wandb --run_name lr_1e-5_v2 \
      --logging_steps 200

Note: --output_dir must be a fresh/empty directory -- main() calls
trainer.train(resume_from_checkpoint=True), which resumes from whatever
checkpoint already lives there instead of starting from the base pretrained
weights. --save_total_limit is intentionally omitted above so every
checkpoint is kept (paired with --save_only_model to keep disk usage
manageable, since optimizer/scheduler/rng state isn't needed for post-hoc
MER/keyword comparison, only for resuming training).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "finetune_official"))
from pathlib import Path
from torch.utils.data import Dataset
from finetune import DataCollator, ScriptArguments as _BaseScriptArguments
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from mer_utils import compute_mer, normalize_for_eval
from batch_generate import batched_transcribe
from eval_hotword_testset import run_hotword_eval
import json

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConversationDataset(Dataset):
    """Conversation dataset with robust root-relative and parent-relative path resolution."""

    def __init__(self, path: str):
        manifest = Path(path).expanduser().resolve()
        self.samples = []
        with manifest.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                self.samples.append(self._parse(json.loads(line), manifest.parent, line_no))
        if not self.samples:
            raise ValueError(f"No samples found in {manifest}")

    @staticmethod
    def _parse(row: dict, root: Path, line_no: int) -> dict[str, str]:
        conversation = row.get("conversation") or []
        expected = [("user", "text"), ("user", "audio"), ("assistant", "text")]
        if not isinstance(conversation, list) or not all(isinstance(item, dict) for item in conversation):
            raise ValueError(f"Line {line_no}: conversation must be a list of messages")
        actual = [(item.get("role"), item.get("message_type")) for item in conversation]
        if actual != expected:
            raise ValueError(f"Line {line_no}: expected user/text, user/audio, assistant/text")

        prompt, audio_path, target = (item.get("content") for item in conversation)
        if not all(isinstance(value, str) and value.strip() for value in (prompt, audio_path, target)):
            raise ValueError(f"Line {line_no}: prompt, audio path, and target must be non-empty strings")

        audio = Path(audio_path).expanduser()
        if not audio.is_absolute():
            candidate1 = (Path(REPRO_ROOT) / audio).resolve()
            candidate2 = (root / audio).resolve()
            audio = candidate1 if candidate1.exists() else candidate2
        return {
            "audio": str(audio),
            "prompt": prompt.strip(),
            "target": target.strip(),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.samples[index]



@dataclass
class ScriptArguments(_BaseScriptArguments):
    val_jsonl: str = field(default=None, metadata={"help": "Conversation-format val manifest (optional)."})
    gen_max_new_tokens: int = field(default=1200, metadata={"help": "Cap for GenerativeMERTrainer's generate() calls."})
    gen_batch_size: int = field(default=8, metadata={"help": "Batched generation batch size for GenerativeMERTrainer."})
    hotword_test_enabled: bool = field(
        default=False,
        metadata={"help": "Also run the 66-example hotword_benchmark test set at every eval (observational only, "
                           "never used for --metric_for_best_model)."},
    )
    hotword_batch_size: int = field(
        default=1,
        metadata={"help": "Batched generation batch size for the hotword_benchmark test set, kept separate from "
                           "gen_batch_size (which is sized for the short 10-example real dev set). Long benchmark "
                           "audio batched together can pad up to 25000+ tokens and crash the SDPA prefill kernel "
                           "(reproduced twice) -- default of 1 is the only size verified safe for the full "
                           "unrestricted-duration set."},
    )
    hotword_max_audio_duration_sec: float = field(
        default=180.0,
        metadata={"help": "Skip hotword_benchmark audio longer than this during the periodic in-training test. "
                           "Pass a very large value (e.g. 1e9) to include everything."},
    )
    # P3: Architectural & Fine-Tuning Improvements
    freeze_whisper_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze Whisper encoder weights to preserve acoustic robustness and prevent catastrophic forgetting."},
    )
    freeze_vq_adaptor: bool = field(
        default=False,
        metadata={"help": "Freeze VQ adaptor weights between audio encoder and language model."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use PEFT LoRA for language model fine-tuning instead of full parameter fine-tuning."},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank r."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA scaling alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate."})
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj",
        metadata={"help": "Comma-separated list of target module names for LoRA."},
    )


class GenerativeMERTrainer(Trainer):
    """Overrides evaluate() entirely (not just prediction_step) to sidestep
    Trainer's generic eval-loop tensor concat/padding machinery, which
    assumes same-shaped batches -- real generation produces a different
    length per example (stops at EOS or gen_max_new_tokens). Uses
    batch_generate.batched_transcribe so the (small) eval_dataset is decoded
    in one or a few batched model.generate() calls, not one at a time.

    Each eval call does two things:
      1. Generative MER on `self.eval_dataset` (meant to be the 10-example
         real dev set) -- this is `eval_mer`, used for
         --metric_for_best_model / checkpoint selection.
      2. If `hotword_test_enabled`, the 66-example hotword_benchmark test set
         (keyword recall / MER-vs-pseudo / false-alarm precision-recall-F1)
         -- purely observational (logged as `eval_test_*`, never used to pick
         a checkpoint; see eval_hotword_testset.py's module docstring).
    """

    def __init__(self, *args, gen_max_new_tokens: int = 1200, gen_batch_size: int = 8,
                 hotword_test_enabled: bool = False, hotword_batch_size: int = 1,
                 hotword_max_audio_duration_sec: float = 180.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_batch_size = gen_batch_size
        self.hotword_test_enabled = hotword_test_enabled
        self.hotword_batch_size = hotword_batch_size
        self.hotword_max_audio_duration_sec = hotword_max_audio_duration_sec

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        model = self.model
        processor = self.processing_class
        was_training = model.training
        model.eval()

        # generate() wants KV-caching; training keeps use_cache=False because
        # it's incompatible with gradient checkpointing -- flip it back for
        # the duration of this eval only.
        prev_use_cache = model.config.use_cache
        prev_text_use_cache = model.config.text_config.use_cache
        model.config.use_cache = True
        model.config.text_config.use_cache = True

        try:
            audio_paths = [dataset[i]["audio"] for i in range(len(dataset))]
            prompts = [dataset[i]["prompt"] for i in range(len(dataset))]
            refs = [dataset[i]["target"] for i in range(len(dataset))]
            hyps = batched_transcribe(model, processor, audio_paths, prompts,
                                       max_new_tokens=self.gen_max_new_tokens,
                                       batch_size=self.gen_batch_size, device=model.device)
            total_errors = total_ref_len = 0
            for ref, hyp in zip(refs, hyps):
                pred_text = normalize_for_eval(hyp, is_asr_output=True)
                true_text = normalize_for_eval(ref)
                errors, ref_len = compute_mer(true_text, pred_text)
                total_errors += errors
                total_ref_len += ref_len
            mer = total_errors / total_ref_len if total_ref_len else 0.0
            metrics = {f"{metric_key_prefix}_mer": mer, f"{metric_key_prefix}_mer_n_examples": len(audio_paths)}

            if self.hotword_test_enabled:
                test_result = run_hotword_eval(model, processor, model.device,
                                                batch_size=self.hotword_batch_size,
                                                max_new_tokens=self.gen_max_new_tokens,
                                                max_audio_duration_sec=self.hotword_max_audio_duration_sec)
                metrics.update({f"{metric_key_prefix}_test_{k}": v for k, v in test_result.items()})
        finally:
            model.config.use_cache = prev_use_cache
            model.config.text_config.use_cache = prev_text_use_cache
            if was_training:
                model.train()

        self.log(metrics)
        return metrics


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]

    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
    )
    train_dataset = ConversationDataset(script_args.train_jsonl)
    eval_dataset = ConversationDataset(script_args.val_jsonl) if script_args.val_jsonl else None
    collator = DataCollator(processor, script_args.max_length)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=script_args.attn_implementation,
    )
    model.tie_weights()
    model.config.use_cache = False
    model.config.text_config.use_cache = False

    # P3: Freeze Whisper Encoder
    if script_args.freeze_whisper_encoder:
        encoder = getattr(getattr(model, "model", model), "whisper_encoder", None)
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad = False
            print("[P3] Successfully froze Whisper encoder (whisper_encoder).")

    # P3: Freeze VQ Adaptor if requested
    if script_args.freeze_vq_adaptor:
        adaptor = getattr(getattr(model, "model", model), "vq_adaptor", None)
        if adaptor is not None:
            for p in adaptor.parameters():
                p.requires_grad = False
            print("[P3] Successfully froze VQ adaptor (vq_adaptor).")

    # P3: LoRA Adaptation
    if script_args.use_lora:
        from peft import LoraConfig, get_peft_model
        targets = [m.strip() for m in script_args.lora_target_modules.split(",") if m.strip()]
        modules_to_save = []
        if not script_args.freeze_vq_adaptor:
            adaptor = getattr(getattr(model, "model", model), "vq_adaptor", None)
            if adaptor is not None:
                modules_to_save.append("vq_adaptor")
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=targets,
            modules_to_save=modules_to_save if modules_to_save else None,
            lora_dropout=script_args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        print("[P3] Successfully initialized PEFT LoRA model:")
        model.print_trainable_parameters()
    else:
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[P3] Model parameter status: {trainable_p:,} trainable / {total_p:,} total ({trainable_p/total_p:.2%})")

    trainer = GenerativeMERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        gen_max_new_tokens=script_args.gen_max_new_tokens,
        gen_batch_size=script_args.gen_batch_size,
        hotword_test_enabled=script_args.hotword_test_enabled,
        hotword_batch_size=script_args.hotword_batch_size,
        hotword_max_audio_duration_sec=script_args.hotword_max_audio_duration_sec,
    )
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
