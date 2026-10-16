"""Batched transcription for MOSS-Transcribe-Diarize: generate for many audio
files in one (or a few) forward passes instead of looping model.generate()
one example at a time. Used by GenerativeMERTrainer's periodic eval (dev set
+ hotword test set) and by the standalone eval scripts, so a 10- or 66-example
pass doesn't cost one full autoregressive decode per example sequentially.

The processor's own __call__ always right-pads (see
processing_moss_transcribe_diarize.py's __call__), which is fine for training
(loss is masked) but wrong for batched generation -- a causal LM needs the
"next token to predict" at the same index for every row in the batch, which
means LEFT padding. This module builds a batch via the normal processor call
and then re-packs each row's padding from the end to the front, leaving the
real token content (audio-token span included) untouched and contiguous --
audio features are matched to rows via audio_chunk_mapping, not by input_ids
position, so shifting the pad block doesn't disturb that.
"""

import soundfile as sf
import soxr
import torch
from tqdm import tqdm

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages


def _load_audio(path: str, target_sr: int):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)
    return audio


def _left_pad_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int):
    """Convert a right-padded [B, L] batch to left-padded, preserving each
    row's real content (and its internal audio-token span) unchanged."""
    B, L = input_ids.shape
    new_ids = torch.full_like(input_ids, pad_token_id)
    new_mask = torch.zeros_like(attention_mask)
    for i in range(B):
        real_len = int(attention_mask[i].sum().item())
        new_ids[i, L - real_len:] = input_ids[i, :real_len]
        new_mask[i, L - real_len:] = 1
    return new_ids, new_mask


@torch.no_grad()
def batched_transcribe(model, processor, audio_paths: list, prompts=None, *, max_new_tokens: int = 1200,
                        batch_size: int = 8, device=None, repetition_penalty: float = 1.0,
                        no_repeat_ngram_size: int = 0) -> list:
    """Returns one generated text string per audio path, in the SAME order as
    the input `audio_paths` (internally processed longest-duration-first so
    each batch pads against similar-length neighbors instead of wasting
    compute padding short clips up to whatever the longest one in an
    arbitrarily-ordered chunk happens to be)."""
    device = device or next(model.parameters()).device
    if prompts is None:
        prompts = [DEFAULT_PROMPT] * len(audio_paths)
    sample_rate = int(processor.feature_extractor.sampling_rate)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0

    order = sorted(range(len(audio_paths)), key=lambda i: sf.info(audio_paths[i]).duration, reverse=True)
    sorted_paths = [audio_paths[i] for i in order]
    sorted_prompts = [prompts[i] for i in order]

    sorted_results = []
    for start in tqdm(range(0, len(sorted_paths), batch_size), desc="batch transcribing"):
        chunk_paths = sorted_paths[start:start + batch_size]
        chunk_prompts = sorted_prompts[start:start + batch_size]
        audios = [_load_audio(p, sample_rate) for p in chunk_paths]
        chat_texts = [
            processor.apply_chat_template(
                build_transcription_messages(path, prompt), tokenize=False, add_generation_prompt=True
            )
            for path, prompt in zip(chunk_paths, chunk_prompts)
        ]

        batch = processor(text=chat_texts, audio=audios, return_tensors="pt")
        input_ids, attention_mask = _left_pad_batch(
            batch["input_ids"], batch["attention_mask"], pad_token_id
        )
        gen_kwargs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "input_features": batch["input_features"].to(device),
            "audio_feature_lengths": batch["audio_feature_lengths"].to(device),
            "audio_chunk_mapping": batch["audio_chunk_mapping"].to(device),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": repetition_penalty,
        }
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        generated = model.generate(**gen_kwargs)
        prompt_len = input_ids.shape[1]
        for row in generated[:, prompt_len:]:
            text = processor.tokenizer.decode(row, skip_special_tokens=True)
            sorted_results.append(text)

    results = [None] * len(audio_paths)
    for orig_idx, text in zip(order, sorted_results):
        results[orig_idx] = text
    return results
