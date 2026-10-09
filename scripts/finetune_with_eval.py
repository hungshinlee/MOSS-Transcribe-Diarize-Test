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
from finetune import ConversationDataset, DataCollator, ScriptArguments as _BaseScriptArguments
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from mer_utils import compute_mer, normalize_for_eval
from batch_generate import batched_transcribe
from eval_hotword_testset import run_hotword_eval


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
