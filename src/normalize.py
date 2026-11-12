"""Record loading and text normalisation (vectorised with polars).

Every record gets:
  name_toks   canonical name tokens (legal forms unified, e.g. pvt/praivet -> private)
  core_toks   name tokens minus legal forms / stop words
  name_core   core tokens joined by spaces
  name_cat    core tokens concatenated (matches domains like "eelegal.com")
  name_skel   consonant skeleton of name_cat (robust to transliteration)
  skel_toks   per-token skeletons (blocking keys)
  legal       sorted legal-form tokens joined by spaces
  addr        canonical address string (abbreviations unified to short forms)
  addr_toks   informative alphabetic address tokens
  nums        numbers found in the address (leading zeros stripped)
  nonlatin    1 if the original name was not in Latin script
Country is carried through as an opaque string label (open set).
"""
import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------- name vocab
# Canonical legal forms. Keys are variants seen in the data (including anyascii
# transliterations of Indic-script "Private Limited" / "LLP").
LEGAL_MAP = {
    "pvt": "private", "praivet": "private", "praibhet": "private", "piraivet": "private",
    "praivrr": "private", "pra": "private", "prv": "private", "pte": "private", "privet": "private",
    "ltd": "limited", "limitet": "limited", "limirrd": "limited", "li": "limited", "lmtd": "limited",
    "limted": "limited", "limitedd": "limited",
    "elelpi": "llp", "incorporated": "inc", "corporation": "corp", "company": "co",
    "cie": "co", "compagnie": "co", "etablissements": "ets", "etablissement": "ets",
}
LEGAL_TOKENS = {
    "private", "limited", "llp", "llc", "inc", "corp", "co", "lp", "pc", "pllc", "plc", "ltda",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "selarl", "ets", "gmbh", "ag",
}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "l", "d", "dba",
             "com", "www", "a", "an", "en", "aur", "evam"}

# --------------------------------------------------------------- address vocab
# Map to SHORT canonical forms so that e.g. street/saint/st collide consistently
# on both sides regardless of country.
ADDR_MAP = {
    "street": "st", "str": "st", "saint": "st", "sainte": "st", "ste": "st",
    "road": "rd", "avenue": "av", "ave": "av", "drive": "dr", "lane": "ln", "boulevard": "bd",
    "blvd": "bd", "court": "ct", "place": "pl", "highway": "hwy", "parkway": "pkwy",
    "circle": "cir", "terrace": "ter", "trail": "trl", "square": "sq", "apartment": "apt",
    "appartement": "apt", "suite": "ste", "floor": "flr", "fl": "flr", "building": "bldg",
    "near": "nr", "opposite": "opp", "opp": "opp", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "rue": "r", "impasse": "imp", "chemin": "ch", "chem": "ch", "allee": "all", "route": "rte",
    "quai": "qu", "faubourg": "fbg", "residence": "res", "township": "twp", "towship": "twp",
    "mount": "mt", "fort": "ft", "point": "pt", "sector": "sec", "number": "no",
    "house": "h", "hno": "h", "colony": "col", "nagar": "ngr", "calcutta": "kolkata",
    "bombay": "mumbai", "madras": "chennai", "bengaluru": "bangalore", "gurugram": "gurgaon",
}
# Region / state names (and Indic transliterations) -> short codes
STATE_PHRASES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    # India (English + anyascii of native-script spellings)
    "maharashtra": "mh", "mharastr": "mh", "delhi": "dl", "dilli": "dl", "uttar pradesh": "up",
    "uttr prdes": "up", "karnataka": "ka", "krnatk": "ka", "tamil nadu": "tn", "tmilnatu": "tn",
    "west bengal": "wb", "pscimbng": "wb", "gujarat": "gj", "gujrat": "gj", "telangana": "tg",
    "telmgan": "tg", "haryana": "hr", "hriyana": "hr", "rajasthan": "rj", "rajsthan": "rj",
    "kerala": "kl", "kerlm": "kl", "bihar": "br", "madhya pradesh": "mp", "mdhy prdes": "mp",
    "andhra pradesh": "ap", "amdhrprdes": "ap", "punjab": "pb", "pmjab": "pb", "odisha": "od",
    "orissa": "od", "od isa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "himachal pradesh": "hp", "goa": "ga", "assam": "as", "jammu and kashmir": "jk",
}
ADDR_NULLS = ["<null>", "null", "n/a", "none", "nan"]
# Very common address words that carry no identity information on their own
ADDR_GENERIC = {"no", "h", "flr", "st", "rd", "av", "dr", "ln", "r", "po", "box", "pmb", "unit",
                "apt", "ste", "bldg", "nr", "opp", "n", "s", "e", "w", "ngr", "col", "sec",
                "near", "plot", "door", "flat", "shop", "block", "ground", "city", "dist",
                "district", "tq", "taluka", "village", "post", "cdp", "twp", "county"}

ASCII_ONLY = r"^[\x00-\x7F]*$"


def _translit(col: pl.Expr) -> pl.Expr:
    """anyascii-transliterate only the strings that are not already ASCII."""
    return (pl.when(col.str.contains(ASCII_ONLY))
            .then(col)
            .otherwise(col.map_elements(anyascii, return_dtype=pl.String, skip_nulls=True)))


def _skeleton(col: pl.Expr) -> pl.Expr:
    """Phonetic consonant skeleton: unify aspirates / similar consonants, drop vowels
    after the first letter, and collapse repeats. Makes 'bharat green' and the
    anyascii of 'भारत ग्रीन' (bharat grin) collide."""
    e = col
    for a, b in [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("dh", "d"), ("th", "t"),
                 ("sh", "s"), ("ch", "c"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("c", "k"),
                 ("z", "j"), ("w", "v"), ("x", "ks")]:
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"m([bgkjtdpf])", "n$1")         # anusvara rendered as m
    e = e.str.slice(0, 1) + e.str.slice(1).str.replace_all(r"[aeiouyh]", "")
    for ch in "bcdfgjklmnprstvx":
        e = e.str.replace_all(ch + "+", ch)
    return e


def load_source(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       null_values=[]).with_columns(
        pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""))


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Add normalised columns to a raw source frame."""
    name = _translit(pl.col("business_name")).str.to_lowercase()
    name = (name.str.replace_all(r"https?://|www\.", " ")
                .str.replace_all(r"\.(com|in|net|org|co|fr|biz|info|us)\b", " ")
                .str.replace_all("&", " and ", literal=True)
                .str.replace_all(r"[.'`’]", "")
                .str.replace_all(r"[^a-z0-9]+", " ")
                .str.strip_chars())
    addr = _translit(pl.col("business_address")).str.to_lowercase()
    for n in ADDR_NULLS:
        addr = addr.str.replace_all(n, " ", literal=True)
    addr = (addr.str.replace_all(r"(\d+)(st|nd|rd|th)\b", "$1")
                .str.replace_all(r"[^a-z0-9]+", " ")
                .str.replace_all(r"(\d)([a-z])", "$1 $2")
                .str.replace_all(r"([a-z])(\d)", "$1 $2"))
    addr = (pl.lit(" ") + addr + pl.lit(" ")).str.replace_all(" ", "  ", literal=True)
    phrases = sorted(STATE_PHRASES, key=len, reverse=True)
    addr = addr.str.replace_many([" " + p.replace(" ", "  ") + " " for p in phrases],
                              [f" {STATE_PHRASES[p]} " for p in phrases])
    addr = addr.str.replace_all(r"\s+", " ").str.strip_chars()

    out = df.with_columns(
        name.alias("_name"), addr.alias("_addr"),
        pl.col("business_name").str.contains(r"[^\x00-\x7FÀ-ɏ‘-‟]")
          .cast(pl.Int8).alias("nonlatin"),
    )
    out = out.with_columns(
        pl.col("_name").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "").replace(LEGAL_MAP)).alias("name_toks"),
        pl.col("_addr").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "").replace(ADDR_MAP)).alias("_atoks"),
    )
    stop = list(LEGAL_TOKENS | NAME_STOP)
    out = out.with_columns(
        pl.col("name_toks").list.eval(pl.element().filter(~pl.element().is_in(stop))).alias("core_toks"),
        pl.col("name_toks").list.eval(pl.element().filter(pl.element().is_in(list(LEGAL_TOKENS))))
          .list.unique().list.sort().list.join(" ").alias("legal"),
        pl.col("_atoks").list.join(" ").alias("addr"),
        pl.col("_atoks").list.eval(
            pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0"))
          .list.eval(pl.element().filter(pl.element() != ""))
          .list.unique(maintain_order=True).alias("nums"),
        pl.col("_atoks").list.eval(
            pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$")
                                & ~pl.element().is_in(list(ADDR_GENERIC)))
        ).list.unique(maintain_order=True).alias("addr_toks"),
    )
    # A name that is entirely legal forms/stop words keeps its raw tokens as core
    out = out.with_columns(
        pl.when(pl.col("core_toks").list.len() == 0).then(pl.col("name_toks"))
          .otherwise(pl.col("core_toks")).alias("core_toks"))
    out = out.with_columns(
        pl.col("core_toks").list.join(" ").alias("name_core"),
        pl.col("core_toks").list.join("").alias("name_cat"),
        pl.col("core_toks").list.eval(_skeleton(pl.element())).alias("skel_toks"),
    )
    out = out.with_columns(_skeleton(pl.col("name_cat")).alias("name_skel"))
    return out.drop("_name", "_addr", "_atoks")


def load_split(data_dir, split: str) -> pl.DataFrame:
    """Load + normalise all three sources of a split into one frame.
    Adds src (1/2/3) and a dense integer row id `rid`."""
    frames = []
    for s in (1, 2, 3):
        df = load_source(f"{data_dir}/{split}/{split}_source{s}.tsv")
        frames.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")))
    df = pl.concat(frames)
    df = normalize(df)
    return df.with_row_index("rid")
