"""Cross-encoder matcher for uncertain (record, S1) pairs.

Fine-tunes a transformer that reads both raw records side by side
("name | address" [SEP] "name | address") and predicts whether they are the same
business. Trained per cross-fitting fold on the stage-1 *uncertain* pairs only
(0.02 <= p1 <= 0.98): that is where every remaining matching error lives.
Runs on GPU when available, otherwise CPU.

    python ce_train.py --data work/ce/train_uncertain.parquet --fold 0 --n 90000 --out work/ce/model_f0
"""
import argparse
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def batches(t1, tq, y, tok, bs, max_len, order):
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        enc = tok([t1[k] for k in j], [tq[k] for k in j], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield enc, (torch.tensor(y[j], dtype=torch.float32) if y is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fold", type=int, required=True, help="train on queries of this cross-fitting fold")
    ap.add_argument("--n", type=int, default=0, help="subsample this many training pairs (0 = all)")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.manual_seed(a.fold)
    d = pl.read_parquet(a.data, columns=["fold", "y", "t1", "tq"]).filter(pl.col("fold") == a.fold)
    if a.n and d.height > a.n:
        d = d.sample(a.n, seed=a.fold)
    t1, tq, y = d["t1"].to_list(), d["tq"].to_list(), d["y"].to_numpy().astype(np.float32)
    print(f"[ce] device={dev} fold={a.fold} train pairs={len(y):,} pos={y.mean():.3f} model={a.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, ignore_mismatched_sizes=True).to(dev)
    steps_per_epoch = -(-len(y) // a.bs)
    total = int(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total + 1, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t, step, rng = time.time(), 0, np.random.default_rng(a.fold)
    while step < total:
        for enc, yb in batches(t1, tq, y, tok, a.bs, a.max_len, rng.permutation(len(y))):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(**enc).logits.squeeze(-1).float(), yb.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0 or step == total:
                el = time.time() - t
                print(f"[ce] step {step}/{total} loss {loss.item():.4f} {el:.0f}s eta {el / step * (total - step) / 60:.0f}min", flush=True)
            if step >= total:
                break
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"[ce] saved {a.out} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
