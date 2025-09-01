#!/usr/bin/env python3
"""
M4/MPS micro-benchmark for MiniLM fine-tuning throughput.
Measures forward+backward steps/sec on synthetic data.
"""
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("BENCH_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SEQ_LEN = int(os.environ.get("BENCH_SEQ_LEN", "128"))
BATCH = int(os.environ.get("BENCH_BATCH", "64"))
STEPS = int(os.environ.get("BENCH_STEPS", "100"))

class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, size=8192, seq_len=128):
        self.size = size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        # random tokens in vocab range [100, 20000)
        ids = torch.randint(100, 20000, (self.seq_len,), dtype=torch.long)
        attn = torch.ones(self.seq_len, dtype=torch.long)
        label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
        return {"input_ids": ids, "attention_mask": attn, "labels": label}

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.to(device)
    model.train()

    ds = RandomTextDataset(tokenizer, size=STEPS * BATCH * 2, seq_len=SEQ_LEN)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Warmup
    it = iter(dl)
    for _ in range(5):
        batch = next(it)
        for k in ("input_ids", "attention_mask", "labels"):
            batch[k] = batch[k].to(device)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step(); optim.zero_grad(set_to_none=True)

    # Timed steps
    steps = 0
    tokens = 0
    start = time.time()
    with torch.autocast(device_type=("mps" if device.type == "mps" else "cpu"), dtype=torch.float16, enabled=(device.type=="mps")):
        while steps < STEPS:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            for k in ("input_ids", "attention_mask", "labels"):
                batch[k] = batch[k].to(device)
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step(); optim.zero_grad(set_to_none=True)
            steps += 1
            tokens += int(batch["input_ids"].numel())
    elapsed = time.time() - start

    sps = (STEPS * BATCH) / elapsed
    tps = tokens / elapsed
    print(f"Steps: {STEPS}, Batch: {BATCH}, Seq: {SEQ_LEN}")
    print(f"Throughput: {sps:.2f} samples/sec, {tps:.0f} tokens/sec")
    # ETA for 100k x 5 epochs
    total_samples = 100_000 * 5
    eta_sec = total_samples / sps
    print(f"ETA for 100k x 5 epochs: {eta_sec/60:.1f} minutes")

if __name__ == "__main__":
    main()
