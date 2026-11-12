"""Score (record, S1) text pairs with a fine-tuned cross-encoder.

    python ce_score.py --model work/ce/model_f0 --data work/ce/test_uncertain.parquet --out work/ce/test_ce_f0.parquet
Optional --fold k scores only queries of fold k (out-of-fold scoring for training data).
On CPU the model is int8 dynamically quantised and batches are length-sorted.
"""
import argparse
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--rows", default=None, help="optional parquet of (q_rid, s1_rid) to restrict scoring to")
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pl.read_parquet(a.data, columns=["q_rid", "s1_rid", "fold", "t1", "tq"])
    if a.fold is not None:
        d = d.filter(pl.col("fold") == a.fold)
    if a.rows:
        d = d.join(pl.read_parquet(a.rows).select("q_rid", "s1_rid"), on=["q_rid", "s1_rid"], how="semi")
    d = d.with_columns((pl.col("t1").str.len_chars() + pl.col("tq").str.len_chars()).alias("_len")).sort("_len")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).eval()
    if dev == "cpu":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        model = model.half().to(dev)
    t1, tq = d["t1"].to_list(), d["tq"].to_list()
    out = np.empty(len(t1), dtype=np.float32)
    t = time.time()
    with torch.inference_mode():
        for i in range(0, len(t1), a.bs):
            enc = tok(t1[i:i + a.bs], tq[i:i + a.bs], truncation=True, max_length=a.max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            out[i:i + a.bs] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if (i // a.bs) % 200 == 0:
                el = time.time() - t
                done = min(i + a.bs, len(t1))
                print(f"[score] {done:,}/{len(t1):,} {done / max(el, 1e-9):.0f} pairs/s eta {(len(t1) - done) / max(done / max(el, 1e-9), 1e-9) / 60:.0f}min", flush=True)
    d.select("q_rid", "s1_rid").with_columns(pl.Series("ce", out)).write_parquet(a.out)
    print(f"[score] wrote {a.out} ({len(t1):,} pairs, {time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
