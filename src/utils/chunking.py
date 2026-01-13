from typing import Tuple

import torch


def chunk_text(
    tokenizer,
    text: str,
    max_len: int,
    stride: int,
    max_windows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if max_windows <= 0:
        raise ValueError("max_windows must be > 0")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    window_len = max_len - special_tokens
    if window_len <= 0:
        raise ValueError("max_len too small for special tokens")

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        tokens = []

    step = window_len - stride
    if step <= 0:
        step = window_len

    windows = []
    for start in range(0, max(len(tokens), 1), step):
        if len(windows) >= max_windows:
            break
        chunk = tokens[start:start + window_len]
        input_ids = tokenizer.build_inputs_with_special_tokens(chunk)
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        elif len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        windows.append((input_ids, attention_mask))

        if start + window_len >= len(tokens):
            break

    if not windows:
        empty_ids = [tokenizer.pad_token_id] * max_len
        empty_mask = [0] * max_len
        windows = [(empty_ids, empty_mask)]

    input_ids = torch.tensor([w[0] for w in windows], dtype=torch.long)
    attention_mask = torch.tensor([w[1] for w in windows], dtype=torch.long)
    win_mask = torch.ones(len(windows), dtype=torch.long)

    if len(windows) < max_windows:
        pad_windows = max_windows - len(windows)
        pad_ids = torch.full((pad_windows, max_len), tokenizer.pad_token_id, dtype=torch.long)
        pad_mask = torch.zeros((pad_windows, max_len), dtype=torch.long)
        input_ids = torch.cat([input_ids, pad_ids], dim=0)
        attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
        win_mask = torch.cat([win_mask, torch.zeros(pad_windows, dtype=torch.long)], dim=0)

    return input_ids, attention_mask, win_mask
