"""Stage 2: pairwise features for (S2/S3 query record, S1 candidate) pairs.

All features are language-agnostic similarity signals (no country one-hot), so the
model transfers to countries unseen in training (France in the test set).
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from blocking import KEY_TYPES

FEAT_REC_COLS = ["rid", "src", "name_core", "name_cat", "name_skel", "core_toks", "legal",
                 "addr", "addr_toks", "nums", "nonlatin"]


OCR_FROM, OCR_TO = list("0134578"), list("oleastb")


def repair_ocr(recs: pl.DataFrame) -> pl.DataFrame:
    """Undo OCR-style digit-for-letter noise inside name tokens that also contain
    letters ("m0unt 8aptist" -> "mount baptist", "nunezr0yalsierra")."""
    from normalize import _skeleton
    fix = (pl.when(pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"[0-9]"))
             .then(pl.element().str.replace_many(OCR_FROM, OCR_TO)).otherwise(pl.element()))
    recs = recs.with_columns(pl.col("core_toks").list.eval(fix))
    recs = recs.with_columns(pl.col("core_toks").list.join(" ").alias("name_core"),
                             pl.col("core_toks").list.join("").alias("name_cat"))
    return recs.with_columns(_skeleton(pl.col("name_cat")).alias("name_skel"))


class RecordStore:
    """Normalised records of one split plus per-record TF-IDF vectors (rows = rid)."""

    def __init__(self, recs: pl.DataFrame):
        t = time.time()
        assert (recs["rid"].to_numpy() == np.arange(recs.height)).all()
        self.recs = recs = repair_ocr(recs)
        self.name_vec, self.name_info = self._tfidf(recs["name_core"])
        self.addr_vec, self.addr_info = self._tfidf(recs["addr"])
        print(f"[feat] tf-idf vectors built ({time.time() - t:.0f}s)")

    @staticmethod
    def _tfidf(col: pl.Series):
        docs = col.to_list()
        vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True, dtype=np.float32,
                              min_df=1, norm=None)
        x = vec.fit_transform(docs).tocsr()
        info = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)   # total IDF mass
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel()).astype(np.float32)
        norms[norms == 0] = 1.0
        x = sparse.diags(1.0 / norms).dot(x).tocsr().astype(np.float32)
        return x, info

    def gather(self, rids: np.ndarray, suffix: str) -> pl.DataFrame:
        return self.recs.select(pl.all().gather(rids)).rename(lambda c: c + suffix)


def _rowdot(x, a, b):
    return np.asarray(x[a].multiply(x[b]).sum(axis=1)).ravel().astype(np.float32)


def _cp(scorer, a, b):
    return (process.cpdist(a, b, scorer=scorer, workers=-1) / (100.0 if scorer is not JaroWinkler.normalized_similarity else 1.0)).astype(np.float32)


def pair_features(store: RecordStore, pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: q_rid, s1_rid, bscore, brank, nA..nG (complete candidate groups per q_rid)."""
    qi = pairs["q_rid"].to_numpy().astype(np.int64)
    si = pairs["s1_rid"].to_numpy().astype(np.int64)
    q = store.gather(qi, "_q")
    s = store.gather(si, "_s")
    df = pl.concat([pairs, q, s], how="horizontal")

    qn, sn = df["name_core_q"].to_list(), df["name_core_s"].to_list()
    qc, sc = df["name_cat_q"].to_list(), df["name_cat_s"].to_list()
    qk, sk = df["name_skel_q"].to_list(), df["name_skel_s"].to_list()
    qa, sa = df["addr_q"].to_list(), df["addr_s"].to_list()
    f = {
        "n_ratio": _cp(fuzz.ratio, qn, sn),
        "n_tsort": _cp(fuzz.token_sort_ratio, qn, sn),
        "n_tset": _cp(fuzz.token_set_ratio, qn, sn),
        "n_partial": _cp(fuzz.partial_ratio, qn, sn),
        "n_cat_jw": _cp(JaroWinkler.normalized_similarity, qc, sc),
        "n_cat_partial": _cp(fuzz.partial_ratio, qc, sc),
        "n_skel_ratio": _cp(fuzz.ratio, qk, sk),
        "n_skel_partial": _cp(fuzz.partial_ratio, qk, sk),
        "a_ratio": _cp(fuzz.ratio, qa, sa),
        "a_tset": _cp(fuzz.token_set_ratio, qa, sa),
        "a_partial": _cp(fuzz.partial_ratio, qa, sa),
        "n_cos": _rowdot(store.name_vec, qi, si),
        "a_cos": _rowdot(store.addr_vec, qi, si),
        "n_info_q": store.name_info[qi], "n_info_s": store.name_info[si],
        "a_info_q": store.addr_info[qi], "a_info_s": store.addr_info[si],
    }
    # first numbers (house numbers) fuzzy agreement
    qf = df["nums_q"].list.first().fill_null("").to_list()
    sf = df["nums_s"].list.first().fill_null("").to_list()
    f["num1_ratio"] = _cp(fuzz.ratio, qf, sf)
    df = df.with_columns(**{k: pl.Series(v) for k, v in f.items()})

    inter = lambda a, b: pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = lambda a, b: pl.col(a).list.set_union(pl.col(b)).list.len()
    df = df.with_columns(
        (inter("core_toks_q", "core_toks_s") / union("core_toks_q", "core_toks_s").clip(1)).alias("n_jac"),
        inter("core_toks_q", "core_toks_s").alias("n_common"),
        pl.col("core_toks_q").list.len().alias("n_len_q"),
        pl.col("core_toks_s").list.len().alias("n_len_s"),
        (inter("addr_toks_q", "addr_toks_s") / union("addr_toks_q", "addr_toks_s").clip(1)).alias("a_jac"),
        pl.col("addr_toks_q").list.len().alias("a_len_q"),
        pl.col("addr_toks_s").list.len().alias("a_len_s"),
        inter("nums_q", "nums_s").alias("num_common"),
        (inter("nums_q", "nums_s") / union("nums_q", "nums_s").clip(1)).alias("num_jac"),
        pl.col("nums_q").list.len().alias("num_len_q"),
        pl.col("nums_s").list.len().alias("num_len_s"),
        (pl.col("nums_q").list.first() == pl.col("nums_s").list.first()).fill_null(False).alias("num1_eq"),
        (pl.col("nums_q").list.first().str.slice(0, 9).cast(pl.Int64, strict=False)
         - pl.col("nums_s").list.first().str.slice(0, 9).cast(pl.Int64, strict=False))
          .abs().log1p().fill_null(-1).alias("num1_logdiff"),
        (pl.col("nums_q").list.first().str.len_chars() == pl.col("nums_s").list.first().str.len_chars())
          .fill_null(False).alias("num1_len_eq"),
        pl.col("nums_q").list.first().is_in(pl.col("nums_s")).fill_null(False).alias("num1_q_in_s"),
        (pl.col("nums_s").list.first().str.starts_with(pl.col("nums_q").list.first())
         | pl.col("nums_q").list.first().str.starts_with(pl.col("nums_s").list.first()))
          .fill_null(False).alias("num1_prefix"),
        pl.col("nums_s").list.first().is_in(pl.col("nums_q")).fill_null(False).alias("num1_s_in_q"),
        (pl.col("legal_q") == pl.col("legal_s")).alias("legal_eq"),
        ((pl.col("legal_q") != "") & (pl.col("legal_s") != "") & (pl.col("legal_q") != pl.col("legal_s"))).alias("legal_conflict"),
        (pl.col("name_cat_q") == pl.col("name_cat_s")).alias("n_exact"),
        (pl.col("name_cat_s").str.starts_with(pl.col("name_cat_q")) | pl.col("name_cat_q").str.starts_with(pl.col("name_cat_s"))).alias("n_prefix"),
        (pl.col("addr_q").str.len_chars() == 0).alias("a_empty_q"),
        pl.col("nonlatin_q").alias("nonlatin"),
        (pl.col("src_q") == 3).alias("is_s3"),
    )
    # digit-position edit type of the first house numbers: noise hits the leading digits,
    # look-alike "twin" businesses differ in the trailing digits (P(true) 0.98 vs 0.33-0.46)
    qa_, sa_ = pl.col("nums_q").list.first().fill_null(""), pl.col("nums_s").list.first().fill_null("")
    both = (qa_ != "") & (sa_ != "")
    same_len = both & (qa_.str.len_chars() == sa_.str.len_chars())
    L = qa_.str.len_chars()
    diff_i = [(qa_.str.slice(i, 1) != sa_.str.slice(i, 1)) & (pl.lit(i) < L) for i in range(9)]
    ham = pl.sum_horizontal([d.cast(pl.Int8) for d in diff_i])
    first_diff = pl.coalesce([pl.when(d).then(pl.lit(i)) for i, d in enumerate(diff_i)] + [pl.lit(9)])
    last_diff = pl.coalesce([pl.when(diff_i[i]).then(pl.lit(i)) for i in range(8, -1, -1)] + [pl.lit(-1)])
    df = df.with_columns(
        (both & (qa_ != sa_) & (qa_.str.ends_with(sa_) | sa_.str.ends_with(qa_))).alias("num1_suffix"),
        pl.when(same_len).then(ham).otherwise(-1).alias("num1_hamming"),
        pl.when(same_len & (ham > 0)).then(first_diff).otherwise(-1).alias("num1_diff_from_left"),
        pl.when(same_len & (ham > 0)).then(L - 1 - last_diff).otherwise(-1).alias("num1_diff_from_right"),
        (same_len & (ham == 1) & (first_diff == L - 1)
         & ((qa_.str.slice(0, 9).cast(pl.Int64, strict=False) - sa_.str.slice(0, 9).cast(pl.Int64, strict=False)).abs() <= 2)
         ).fill_null(False).alias("num1_last_digit_near"),
        (inter("nums_q", "nums_s") / pl.col("nums_q").list.len().clip(1)).alias("num_cov_q"),
        (inter("nums_q", "nums_s") / pl.col("nums_s").list.len().clip(1)).alias("num_cov_s"),
    )
    # query-level context: how does this candidate compare with its competitors?
    ctx = []
    for c in ["n_cos", "a_cos", "n_tset", "a_tset", "bscore", "num_jac"]:
        second = (pl.when(pl.len().over("q_rid") > 1)
                    .then(pl.col(c).top_k(2).min().over("q_rid")).otherwise(0.0))
        ctx += [(pl.col(c) - pl.col(c).max().over("q_rid")).alias(f"{c}_gap"),
                (pl.col(c) - second).alias(f"{c}_gap2")]
    df = df.with_columns(ctx + [pl.len().over("q_rid").alias("n_cand")])
    return df.select("q_rid", "s1_rid", *FEATURES).with_columns(
        pl.col(FEATURES).cast(pl.Float32))


FEATURES = (
    ["bscore", "brank"] + [f"n{k}" for k in KEY_TYPES]
    + ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_cat_jw", "n_cat_partial", "n_skel_ratio",
       "n_skel_partial", "a_ratio", "a_tset", "a_partial", "n_cos", "a_cos", "n_info_q", "n_info_s",
       "a_info_q", "a_info_s", "num1_ratio", "n_jac", "n_common", "n_len_q", "n_len_s", "a_jac",
       "a_len_q", "a_len_s", "num_common", "num_jac", "num_len_q", "num_len_s", "num1_eq",
       "num1_logdiff", "num1_len_eq", "num1_q_in_s", "num1_prefix", "num1_suffix", "num1_hamming",
       "num1_diff_from_left", "num1_diff_from_right", "num1_last_digit_near", "num_cov_q", "num_cov_s",
       "num1_s_in_q", "legal_eq", "legal_conflict", "n_exact", "n_prefix", "a_empty_q", "nonlatin",
       "is_s3"]
    + [f"{c}_{g}" for c in ["n_cos", "a_cos", "n_tset", "a_tset", "bscore", "num_jac"] for g in ["gap", "gap2"]]
    + ["n_cand", "s1_top1_cnt"]
)
