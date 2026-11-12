"""Builds the project report PDF (reportlab). All numbers come from measured results of this project.
Final edition: from the first look at the data to the submitted package (v10c)."""
import math

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Body", F + "segoeui.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", F + "segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", F + "segoeuii.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F + "consola.ttf"))
registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

INK = colors.HexColor("#1f2933"); MUTED = colors.HexColor("#52606d"); ACCENT = colors.HexColor("#1d4ed8")
RULE = colors.HexColor("#cbd2d9"); SOFT = colors.HexColor("#f0f4f8"); GOOD = colors.HexColor("#15803d"); BAD = colors.HexColor("#b91c1c")
AMBER = colors.HexColor("#b45309")

S = {
    "title": ParagraphStyle("t", fontName="Body-Bold", fontSize=25, leading=31, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("st", fontName="Body", fontSize=12.5, leading=18, textColor=MUTED),
    "part": ParagraphStyle("pt", fontName="Body-Bold", fontSize=22, leading=28, textColor=ACCENT, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=22, textColor=ACCENT, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=12.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", fontName="Body", fontSize=9.6, leading=14, textColor=INK, spaceAfter=5),
    "b": ParagraphStyle("b", fontName="Body", fontSize=9.6, leading=13.5, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "small": ParagraphStyle("s", fontName="Body", fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("c", fontName="Body", fontSize=8.3, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Body-Bold", fontSize=8.3, leading=10.8, textColor=INK),
    "mono": ParagraphStyle("m", fontName="Mono", fontSize=7.6, leading=10, textColor=INK),
    "callout": ParagraphStyle("co", fontName="Body", fontSize=9.6, leading=14, textColor=INK, backColor=SOFT,
                              borderPadding=(7, 9, 7, 9), spaceBefore=4, spaceAfter=9, borderColor=RULE, borderWidth=0.5),
}

story = []
P = lambda t, s="p": story.append(Paragraph(t, S[s]))
from reportlab.platypus import CondPageBreak
H1 = lambda t: (story.append(CondPageBreak(6 * cm)), story.append(Paragraph(t, S["h1"])))
H2 = lambda t: (story.append(CondPageBreak(3.5 * cm)), story.append(Paragraph(t, S["h2"])))
H3 = lambda t: story.append(Paragraph(t, S["h3"]))
SP = lambda h=6: story.append(Spacer(1, h))
BR = lambda: story.append(PageBreak())
M = lambda t: f"<font face='Mono'>{t}</font>"


def B(items):
    for it in items:
        story.append(Paragraph(it, S["b"], bulletText="•"))
    SP(3)


def CALL(t):
    story.append(Paragraph(t, S["callout"]))


def INTUITION(t):
    CALL("<b>Intuition.</b> " + t)


def T(rows, widths, head=True):
    data = [[Paragraph(str(c), S["cellb" if (head and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
          ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(st))
    story.append(t)
    SP(8)


def CODE(t):
    story.append(Preformatted(t, S["mono"]))
    SP(6)


def PART(num, title, blurb):
    BR()
    SP(150)
    P(f"Part {num}", "subtitle")
    P(title, "part")
    P(blurb, "subtitle")
    BR()


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=INK, strokeWidth=0.9))
    a = math.atan2(y2 - y1, x2 - x1); L = 5
    d.add(Polygon([x2, y2, x2 - L * math.cos(a - 0.4), y2 - L * math.sin(a - 0.4),
                   x2 - L * math.cos(a + 0.4), y2 - L * math.sin(a + 0.4)], fillColor=INK, strokeColor=INK))


def pipeline_diagram():
    w, h = 17 * cm, 12.2 * cm
    d = Drawing(w, h)
    boxes = [
        ("Raw TSVs", "S1 / S2 / S3 records\n(name, address, country)"),
        ("0 · Normalise", "transliterate, canonicalise,\nnumbers, skeletons, OCR repair"),
        ("1 · Key blocking", "7 hashed key families\ntop-10 S1 per record"),
        ("2 · Pair features", "72 similarity + competition\nfeatures per pair"),
        ("3 · Stage-1 LightGBM", "2-fold cross-fitted\npair classifier → p1"),
        ("Assign + prune", "best S1 per record, p1 ≥ 0.02\n→ candidate_pairs.tsv"),
        ("Expansion (v10b/c)", "weak records (p1 < 0.5): top-30,\nIndia top-50, re-score"),
        ("4 · Cross-encoders", "MiniLM v7 + v8 read raw text\nof uncertain pairs → ce"),
        ("5 · Stacker3", "LightGBM on p1, margin,\n2 CE scores, 72 feats → ps"),
        ("6 · Stage-2 veto", "group / twin model p2;\nsoft (US, India), full (France)"),
        ("7 · Calibrate", "per-country EM label shift\n(France extra ×0.5)"),
        ("8 · Decide", "per-S1 expected-F0.5\nset → matching_results.tsv"),
    ]
    bw, bh, gapx, gapy = 5.1 * cm, 1.75 * cm, 0.75 * cm, 1.0 * cm
    pos = {}
    for i, (t, sub) in enumerate(boxes):
        row, col = divmod(i, 3)
        if row % 2 == 1:
            col = 2 - col
        x = 0.2 * cm + col * (bw + gapx)
        y = h - 0.3 * cm - bh - row * (bh + gapy)
        pos[i] = (x, y)
        fill = (colors.HexColor("#dbeafe") if i in (4, 7, 8, 9) else
                colors.HexColor("#fef3c7") if i in (5, 6) else colors.HexColor("#dcfce7") if i == 11 else SOFT)
        d.add(Rect(x, y, bw, bh, rx=5, ry=5, fillColor=fill, strokeColor=ACCENT, strokeWidth=0.8))
        d.add(String(x + bw / 2, y + bh - 0.52 * cm, t, fontName="Body-Bold", fontSize=8.8, textAnchor="middle", fillColor=INK))
        for j, line in enumerate(sub.split("\n")):
            d.add(String(x + bw / 2, y + bh - 0.98 * cm - j * 0.34 * cm, line, fontName="Body", fontSize=7.2, textAnchor="middle", fillColor=MUTED))
    for a_ in range(len(boxes) - 1):
        (x1, y1), (x2, y2) = pos[a_], pos[a_ + 1]
        if abs(y1 - y2) < 1:
            if x2 > x1:
                arrow(d, x1 + bw, y1 + bh / 2, x2, y2 + bh / 2)
            else:
                arrow(d, x1, y1 + bh / 2, x2 + bw, y2 + bh / 2)
        else:
            arrow(d, x1 + bw / 2, y1, x2 + bw / 2, y2 + bh)
    d.add(String(0.35 * cm, 0.15 * cm, "Blue = learned models · amber = candidate set · green = final output. Every stage after pruning runs only on the 3.81 candidates per S1.",
                 fontName="Body-Italic", fontSize=7.4, fillColor=MUTED))
    return d


def score_chart():
    """Public leaderboard and held-out macro F0.5 by version (one shared y-axis: both are the same metric)."""
    vers = ["v2", "v3", "v4a", "v5a", "v6", "v7b", "v9", "v10a", "v10b", "v10c"]
    lb = [0.972, 0.965, 0.973, 0.973, 0.973612, 0.977, 0.979189, 0.979396, 0.980128, 0.980385]
    ho = [0.9812, 0.9833, 0.9813, None, 0.9815, 0.9840, 0.9853, 0.9853, 0.9861, 0.9864]
    w, h = 17 * cm, 8.6 * cm
    d = Drawing(w, h)
    x0, x1, y0, y1 = 1.5 * cm, w - 2.3 * cm, 1.0 * cm, h - 0.6 * cm
    lo, hi = 0.962, 0.992
    X = lambda i: x0 + (x1 - x0) * i / (len(vers) - 1)
    Y = lambda v: y0 + (y1 - y0) * (v - lo) / (hi - lo)
    for v in [0.965, 0.970, 0.975, 0.980, 0.985, 0.990]:
        d.add(Line(x0, Y(v), x1, Y(v), strokeColor=RULE, strokeWidth=0.4))
        d.add(String(x0 - 4, Y(v) - 3, f"{v:.3f}", fontName="Body", fontSize=7, textAnchor="end", fillColor=MUTED))
    bench = 0.990748
    d.add(Line(x0, Y(bench), x1, Y(bench), strokeColor=BAD, strokeWidth=0.8, strokeDashArray=[3, 2]))
    d.add(String(x1 + 3, Y(bench) - 3, "benchmark", fontName="Body", fontSize=7, fillColor=BAD))
    for i, v in enumerate(vers):
        d.add(String(X(i), y0 - 12, v, fontName="Body", fontSize=7.4, textAnchor="middle", fillColor=INK))
    for series, col, name in [(ho, AMBER, "held-out validation"), (lb, ACCENT, "public leaderboard")]:
        pts = [(X(i), Y(v)) for i, v in enumerate(series) if v is not None]
        for (a, b), (c, e) in zip(pts[:-1], pts[1:]):
            d.add(Line(a, b, c, e, strokeColor=col, strokeWidth=2))
        for (a, b) in pts:
            d.add(Circle(a, b, 3.2, fillColor=col, strokeColor=colors.white, strokeWidth=1.2))
        d.add(String(pts[-1][0] + 6, pts[-1][1] - 3, name, fontName="Body-Bold", fontSize=7.2, fillColor=col))
    d.add(String(X(9), Y(0.980385) - 12, "0.980385", fontName="Body-Bold", fontSize=7.4, textAnchor="middle", fillColor=ACCENT))
    d.add(String(X(1), Y(0.965) - 12, "0.965 (v3)", fontName="Body", fontSize=7.2, textAnchor="middle", fillColor=INK))
    d.add(String(X(5) + 6, Y(0.977) - 11, "cross-encoder (v7b)", fontName="Body", fontSize=7.2, textAnchor="start", fillColor=INK))
    return d


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Business Entity Resolution · Amazon ML Challenge 2026 · Team Vyrox · project report (final edition)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


# ====================================================================================== TITLE
SP(70)
P("Business Entity Resolution", "title")
P("Amazon ML Challenge 2026 · the complete build, from the first look at the data to the final submission", "subtitle")
SP(8)
P("Team <b>Vyrox</b>: B Sujan Kumar, Apeksha Sanjay Hudali (BMS College of Engineering, Bengaluru)", "subtitle")
SP(16)
P("Final edition, 2026-09-27. This report is written to build intuition. It explains the problem statement in depth, "
  "the dataset file by file with real rows, every stage of the pipeline and the reasoning behind it, every version we built "
  "(including the ones that failed and why), and every technical term we used, with an explanation of what it means "
  "in general and what it means in this project.")
SP(10)
T([["Snapshot", ""],
   ["Final submission", "<b>v10c</b> · public leaderboard <b>0.980385</b> (our best of 14 leaderboard uploads)"],
   ["Held-out validation", "<b>0.98637</b> on 441,785 Source-1 entities never used for training (US 0.98860 · India 0.98302)"],
   ["Benchmark", "0.990748 (not reached; section 23 explains where the remaining error is and what would close it)"],
   ["Candidate set", "<b>3.81 candidates per Source-1 entity</b> (6,595,983 pairs), only ~10% above the 3.46 true links per entity"],
   ["Final output", "5,772,673 matches for 1,732,544 Source-1 entities · official validator: PASS"],
   ["Models", "LightGBM (MIT) + fine-tuned MiniLM-L6 cross-encoders (Apache-2.0, 22.7 M parameters)"],
   ["Compute", "Intel Core i7-1355U laptop (10 cores / 12 threads), 16 GB RAM, 512 GB SSD, no GPU (AWS GPU quota requests were never approved)"]],
  [3.8, 12.7])
BR()

# ====================================================================================== CONTENTS
H1("Contents")
toc = [
    ("Part I · Understanding the problem", ["1. The problem in one page", "2. Deep dive into the problem statement",
                                            "3. The metric and the leaderboard: how the score is calculated and what moves it", "4. The dataset: every file, its size, structure and real rows",
                                            "5. What exploring the data revealed"]),
    ("Part II · What we built", ["6. Architecture and the funnel of numbers", "7. Stage 0 · normalisation", "8. Stage 1 · blocking (candidate generation)",
                                 "9. Stage 2 · the 72 pair features", "10. Stage 3 · stage-1 LightGBM, assignment and learned pruning",
                                 "11. Stage 4 · transformer cross-encoders", "12. Stage 5 · the stacker", "13. Stage 6 · the group / twin veto",
                                 "14. Stage 7 · calibration under test shift", "15. Stage 8 · the per-entity expected-F0.5 decision",
                                 "16. Candidate expansion for weak records (v10b, v10c)", "17. The life of one entity: a France example end to end",
                                 "18. How we validated (and why validation alone was not enough)"]),
    ("Part III · The journey", ["19. Version history at a glance (with chart)", "20. Every version: what, why, result, lesson",
                                "21. Failures, bugs and dead ends (and what each taught us)", "22. Diagnostics that drove the decisions",
                                "23. Where the remaining error lives, and what would close the gap"]),
    ("Part IV · Engineering and delivery", ["24. Engineering on a 16 GB laptop", "25. Infrastructure: AWS, cost protection, GPU requests",
                                            "26. The submission package and the compliance checks", "27. Code map and how to reproduce"]),
    ("Part V · Terminology", ["28. Glossary: every term, in depth"]),
]
for part, items in toc:
    P(f"<b>{part}</b>", "p")
    for it in items:
        story.append(Paragraph(it, ParagraphStyle("toc", parent=S["b"], fontSize=9.2, leading=12.2, spaceAfter=0), bulletText=" "))
    SP(3)

# ====================================================================================== PART I
PART("I", "Understanding the problem", "What we are asked to do, how it is scored, what the data looks like, and what the data taught us before we built anything.")

H1("1. The problem in one page")
P("Businesses are described in three independent data feeds, called <b>sources</b>. The sources never share an identifier. "
  "Each record has only three fields: a <b>business name</b>, a <b>business address</b> and a <b>country</b>. Both text fields are "
  "noisy: misspelled, abbreviated, re-ordered, sometimes written in an Indian script, sometimes empty. The task, called <b>entity "
  "resolution</b> (also <i>record linkage</i> or <i>deduplication</i>), is to decide which records describe the same real business.")
B(["<b>Source 1 (S1)</b> is the reference list. It is clean-ish and deduplicated: each business appears at most once.",
   "<b>Source 2 (S2)</b> and <b>Source 3 (S3)</b> are messy feeds. A business can appear 0 to about 6 times in them, and many S2/S3 "
   "records belong to businesses that are not in S1 at all (distractors).",
   "<b>Output:</b> for every S1 business, the list of all S2/S3 record IDs that are the same business, possibly an empty list."])
INTUITION("think of S1 as a phone book and S2/S3 as ten million handwritten notes. Each note is about one business, "
          "maybe misspelled, maybe in Hindi or Telugu, maybe with half the address missing. For every phone-book entry you must "
          "collect exactly its own notes, and never a note about the look-alike shop two doors down. You cannot read every note "
          "against every entry (that would be 17 trillion comparisons), so you first need a fast way to shortlist, then a careful way "
          "to decide.")
H2("The two sub-problems")
T([["Sub-problem", "Question", "Our answer (in one line)"],
   ["Candidate generation (blocking)", "Which few S1 entities could this record possibly belong to?",
    "hashed keys built from rare name/address tokens → top-10 S1 per record, then a model keeps only the best one"],
   ["Matching", "Of those candidates, which pairs are truly the same business?",
    "gradient-boosted trees + transformer cross-encoders + group-level twin veto + a metric-aware decision per S1"]],
  [4, 5.5, 7])
H2("Why this is hard")
B(["<b>Same business, different text:</b> 'Geis Assets Inc.' vs 'GEIS  ASSETS' vs 'Geis-Assets Inc.' vs a Telugu-script spelling.",
   "<b>Different business, same text:</b> a <i>twin</i> such as 'Adams Made, Inc., 833 Gostlin Street' vs 'Adams Made, Inc, 84 Gostlin Street' "
   "(real train rows, and <b>not</b> the same business).",
   "<b>The noise overlaps:</b> genuine records also get house-number typos ('447 Chatterton Drive' → '44 CHATTERTON DR' <i>is</i> a true match). "
   "Telling a typo from a twin is the heart of the precision problem.",
   "<b>Scale:</b> 26.4 million records in total; everything had to run on a laptop.",
   "<b>Shift:</b> the test set contains a country (France) that never appears in training, and more look-alike records than training."])

H1("2. Deep dive into the problem statement")
H2("2.1 Inputs")
T([["Item", "Detail"],
   ["Training data", "train_source1/2/3.tsv plus train_ground_truth.tsv (the answer key for training S1 entities)"],
   ["Test data", "test_source1/2/3.tsv, no answer key. Our predictions are scored on the leaderboard"],
   ["Fields per record", M("entity_id") + ", " + M("business_name") + ", " + M("business_address") + ", " + M("country")],
   ["Countries", "training: US, India · test: US, India and <b>France</b> (unseen). Country is an open label: the solution must not assume a fixed list"]],
  [3.4, 13.1])
H2("2.2 Required outputs")
T([["File", "Columns", "Meaning"],
   ["matching_results.tsv", M("source1_entity_id, matched_entity_ids"), "one row per test S1 entity; comma-separated S2/S3 IDs (empty if none). This file is scored"],
   ["candidate_pairs.tsv", M("source1_entity_id, candidate_entity_ids"), "one row per test S1 entity; the candidate set the final model actually scored. Reviewed for the final ranking"]],
  [3.6, 5.6, 7.3])
H2("2.3 Rules and constraints")
T([["Rule", "What it means in practice", "How it shaped our design"],
   ["Every S1 must appear, once", "1,732,544 rows in each output file, no duplicates", "outputs are written from the full S1 list, then validated"],
   ["Only S2/S3 IDs that exist in test", "no self-matches, no invented IDs, no duplicates inside a list", "assignment is one record → one S1, so duplicates are impossible"],
   ["No external data or APIs", "no geocoders, business registries, web lookups or entity-resolution services", "everything learned from the provided files"],
   ["Model licence and size", "final models must be MIT or Apache-2.0 and at most 8 B parameters", "LightGBM (MIT), MiniLM cross-encoder (Apache-2.0, 22.7 M)"],
   ["Country is open", "France appears only in test", "no country feature anywhere; country is used only to scope blocking keys"],
   ["Smaller candidate sets rank higher", "blocking must scale; candidate_pairs.tsv is reviewed with the code", "learned pruning: 3.81 candidates per S1"],
   ["5 leaderboard uploads per day", "each upload is precious; public leaderboard = a subset of test", "decide offline first; use uploads only for measured bets and probes"],
   ["Final package", "zip with output/, code (src, README, requirements), filled documentation", "Vyrox_submission.zip, reproducible end to end"],
   ["Deadline", "2026-09-27 23:59 IST", "final version frozen and validated on the day"]],
  [3.6, 6.2, 6.7])
H2("2.4 Structural facts hidden in the statement")
B(["<b>At most one S1 per record.</b> The statement implies, and the training answer key confirms, that every S2/S3 record belongs to at "
   "most one S1 business. So a record's candidates <i>compete</i>: if it clearly matches one S1, it cannot match another. We exploit "
   "this everywhere (assignment, competition features, pruning).",
   "<b>S1 is deduplicated.</b> Two S1 entries are never the same business, so two very similar S1 entries are informative: a record "
   "that fits both equally well is ambiguous.",
   "<b>Scoring is per S1 entity.</b> The unit of decision is the whole group of records attached to an S1, not an isolated pair."])

H1("3. The metric and the leaderboard: how the score is calculated and what moves it")
H2("3.1 The formula")
P("For each S1 entity we compare our predicted set of IDs with the true set:")
CODE("TP = |pred ∩ true|      FP = |pred − true|      FN = |true − pred|\n"
     "precision P = TP / (TP + FP)           recall R = TP / (TP + FN)\n"
     "F_β = (1 + β²)·P·R / (β²·P + R)        with β = 0.5:\n"
     "F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP)\n"
     "final score = mean of F0.5 over ALL S1 entities (macro average)\n"
     "special case: true set empty (a 'singleton') → predict empty = 1.0, predict anything = 0.0")
H2("3.2 Worked examples (truth = {a, b, c} unless stated)")
T([["We predict", "TP / FP / FN", "F0.5", "Lesson"],
   ["{a, b, c}", "3 / 0 / 0", "1.000", "perfect"],
   ["{a, b} (missed one)", "2 / 0 / 1", "0.909", "a miss costs ~9%"],
   ["{a, b, c, x} (one extra wrong)", "3 / 1 / 0", "0.789", "an extra wrong ID costs ~21%, more than twice a miss"],
   ["{a, b, x} (swapped one)", "2 / 1 / 1", "0.667", "swapping is worst of both"],
   ["{ } (nothing)", "0 / 0 / 3", "0.000", "silence scores zero when matches exist"],
   ["truth {a}: predict {a, x}", "1 / 1 / 0", "0.556", "small groups are fragile"],
   ["truth { } (singleton): predict { }", "-", "1.000", "saying 'nothing' can be perfect"],
   ["truth { }: predict {x}", "-", "0.000", "one wrong merge wipes out a singleton"]],
  [5.2, 2.6, 1.6, 7.1])
B(["<b>β = 0.5 means precision matters more.</b> In the formula a false positive (FP) has weight 1 and a false negative (FN) weight 0.25: a "
   "wrong merge hurts about four times as much as a missed match in the denominator.",
   "<b>Macro average</b>: every S1 counts equally, whether it has 0 or 6 matches. Small groups and singletons (5.6% of training S1) carry "
   "disproportionate weight.",
   "<b>The brief's example</b>: predict {a, b, c}, truth {a, c} → P = 2/3, R = 1 → F0.5 = 0.714."])
INTUITION("the metric punishes over-confidence. When unsure, leaving a record out is usually cheaper than adding it. But the best choice "
          "depends on the whole group: adding a 60%-likely record to a group of two sure records is a bad bet, while adding it to an "
          "otherwise empty group can be good. Section 15 turns this into an exact rule.")

H2("3.3 How the leaderboard computes our score, step by step")
T([["Step", "What happens"],
   ["1. Upload", "we upload matching_results.tsv (and candidate_pairs.tsv). Only the matching file is scored; the candidate file is reviewed for the final ranking"],
   ["2. Validation gate", "the file is checked first: every test S1 exactly once, only S2/S3 IDs that exist in test, no duplicates. A file that fails is "
    "<b>rejected and not scored at all</b>. A valid file shows the status SCORED"],
   ["3. Public subset", "the public leaderboard uses a hidden subset of the test S1 entities (the rest is kept for the private / final evaluation). "
    "We do not know which entities are in it or how large it is"],
   ["4. Per-entity F0.5", "for every S1 in the subset: TP, FP, FN between our list and the hidden true list → F0.5 (1.0 for a correctly empty singleton)"],
   ["5. Macro average", "the leaderboard score is the plain mean of those per-entity scores, so every S1 counts equally"]], [3.2, 13.3])
CALL("<b>So the leaderboard score is literally 'the average entity score'.</b> Our 0.980385 means: on average, an S1 entity in the public subset "
     "got 98.04% of the perfect F0.5. An entity with one wrong extra record among three true ones scores 0.789, a perfect one 1.0, and "
     "the average over about a million entities lands at 0.98.")
H2("3.4 What one mistake costs: it depends on the group size")
P("Assume all other records of the entity are right. Then one mistake changes that entity's F0.5 from 1.0 to:")
T([["True group size n", "one missed match (FN)", "cost", "one wrong extra record (FP)", "cost"],
   ["0 (singleton)", "-", "-", "0.000", "<b>1.000</b>"],
   ["1", "0.000", "<b>1.000</b>", "0.556", "0.444"],
   ["2", "0.833", "0.167", "0.714", "0.286"],
   ["3 (most common)", "0.909", "0.091", "0.789", "0.211"],
   ["4", "0.938", "0.063", "0.833", "0.167"],
   ["6", "0.962", "0.038", "0.882", "0.118"]], [3.6, 3.4, 2, 4.4, 3.1])
B(["<b>A wrong extra record costs 2-3× a miss</b> in the same group (0.211 vs 0.091 for n = 3): the β = 0.5 effect.",
   "<b>Small groups are fragile.</b> Missing the only match of a 1-record entity, or adding anything to a singleton, costs the whole entity (1.0).",
   "<b>A wrong attachment hurts twice.</b> Because each record belongs to one S1, attaching a record to the wrong S1 is an FP for that S1 "
   "<i>and</i> an FN for the right one."])
H2("3.5 From one mistake to the leaderboard number")
P("The leaderboard is an average over N entities, so one entity's loss moves the score by loss / N. With the full test set "
  "(N = 1,732,544) this gives a feeling for the scale:")
T([["To move the leaderboard by +0.001 you must fix about ...", "Count", "Why"],
   ["wrong merges into singletons (cost 1.0 each)", "1,733", "0.001 × 1,732,544 / 1.0"],
   ["wrong extra records in 3-record groups (cost 0.211)", "8,200", "0.001 × 1,732,544 / 0.211"],
   ["missed matches in 3-record groups (cost 0.091)", "19,000", "0.001 × 1,732,544 / 0.091"],
   ["wrong attachments (FP in one group + FN in another, n = 3)", "5,700", "0.001 × 1,732,544 / (0.211 + 0.091)"]], [8.4, 2, 6.1])
P("This is why our late gains look small: v10c changed the best S1 of 262,946 test records (most changes are neutral: the record "
  "was rejected before and after), and moved the leaderboard by +0.00026, about 450 entity-equivalents of improvement.", "small")
H2("3.6 How each country weighs in")
T([["Country", "Test S1 entities", "Share of the score", "Consequence"],
   ["India", "809,986", "46.8%", "the biggest lever; hardest (scripts, transliteration)"],
   ["US", "663,106", "38.3%", "best-understood country"],
   ["France", "259,452", "15.0%", "unseen in training; a change of x in France moves the leaderboard by 0.15·x"]], [2.4, 3, 3, 8.1])
P("Example: the France soft-veto probe lost 0.0012 on the leaderboard. If the loss came from France alone, France's own score fell by "
  "about 0.0012 / 0.15 = 0.008, a large drop, which is why we reverted it.", "small")
H2("3.7 Reference points on the same scale")
T([["Submission", "Score", "Meaning"],
   ["predict an empty list for every S1", "≈ 0.056", "only singletons score (5.6% in training)"],
   ["our first upload (v2)", "0.972", ""],
   ["our final upload (v10c)", "<b>0.980385</b>", "held-out validation 0.9864"],
   ["benchmark", "0.990748", ""],
   ["blocking ceiling (held-out, perfect matcher on our candidates)", "0.9926", "the most our candidate set allows"]], [7.4, 2.6, 6.5])
H2("3.8 Why the leaderboard is lower than our validation")
B(["<b>Distribution shift.</b> Test has more distractors (~40% vs 26%), more look-alike twins (13.5% vs 10.9% twin-like attachments in the "
   "US), more uncertain records (8.0% vs 4.4% in India) and an unseen country. More uncertain records means more chances for FPs, the "
   "expensive error. Validation is measured on training-like data.",
   "<b>The gap</b> was 0.018 at v3 and 0.006 at v10c. Changes that make the model more robust (veto-only, cross-encoders, calibration) "
   "shrink it; changes that exploit training-specific patterns (v3's promotion) widen it.",
   "<b>How validation gains transferred:</b> cross-encoder +0.0025 held-out → +0.0034 leaderboard; v10b expansion +0.0008 → +0.0007; "
   "v10c +0.00025 → +0.00026; v3 +0.0021 → −0.007. Recall and text-model improvements transferred; the group-promotion trick did not."])
H2("3.9 What moved our leaderboard score, and by how much")
T([["Change", "Δ leaderboard", "Type of effect"],
   ["v2 → v3: stage 2 allowed to promote weak records", "<font color='#b91c1c'>−0.0070</font>", "more FPs on test (twins promoted)"],
   ["v3 → v4a: veto only + learned pruning", "<font color='#15803d'>+0.0080</font>", "FPs removed"],
   ["v4a → v5a: EM calibration", "≈ 0", "borderline links half right, half wrong"],
   ["v5a → v6: digit-position features", "≈ +0.0006", "slightly fewer twin FPs"],
   ["v6 → v7b: cross-encoder + stacker", "<font color='#15803d'>+0.0034</font>", "fewer FPs and FNs on uncertain pairs"],
   ["v7b → v9: stronger cross-encoder, soft veto", "<font color='#15803d'>+0.0022</font>", "fewer vetoed true matches"],
   ["v9 → probe: soft veto in France", "<font color='#b91c1c'>−0.0012</font>", "France FPs"],
   ["v9 → v10a: stricter France", "+0.0002", "France FPs removed"],
   ["v10a → v10b: candidate expansion (top-30)", "+0.0007", "blocking misses recovered (FNs)"],
   ["v10b → probes: India / US ×0.7", "−0.0001 / −0.00002", "true matches lost (FNs) outweighed FPs removed"],
   ["v10b → v10c: India top-50", "+0.00026", "more blocking misses recovered"]], [7.6, 3, 5.9])
H2("3.10 How much of a difference is real? (noise on the public subset)")
P("The public leaderboard averages over a <b>sample</b> of entities, so it carries sampling noise. We measured the spread of per-entity "
  "scores on our 441,785 held-out entities (final pipeline): <b>88.2%</b> of entities score a perfect 1.0, <b>0.34%</b> score 0, and the "
  "standard deviation of the per-entity F0.5 is <b>0.0675</b>. The standard error of an average over N entities is std / √N. We do not "
  "know the size of the public subset, so the table shows three cases (assuming it is a random sample):")
T([["Public subset", "Entities N", "Noise on the absolute score (1 SE)", "Noise on the difference between two similar versions (1 SE)"],
   ["25% of test", "433,136", "±0.00010", "±0.00004"],
   ["50% of test", "866,272", "±0.00007", "±0.00003"],
   ["100% of test", "1,732,544", "±0.00005", "±0.00002"]], [2.8, 2.6, 4.8, 6.3])
B(["<b>Why the difference is less noisy than the score.</b> Two versions are scored on the <i>same</i> entities (a paired comparison). "
   "v10c and v10a differ on only 0.78% of held-out entities (std of the per-entity difference 0.0236), so everything else cancels out.",
   "<b>Reading our results.</b> Steps of +0.0007 (v10b) and +0.00026 (v10c) are many standard errors: real. The India ×0.7 probe (−0.0001) "
   "is a real but tiny loss; the US ×0.7 probe (−0.00002) is a tie within noise, so we kept the simpler setting.",
   "<b>Public vs private.</b> The private (final) score may differ from the public one by about one or two standard errors (≈ ±0.0001-0.0002). "
   "The 0.006 gap between validation and leaderboard is far larger than this: it is caused by the distribution shift, not by noise."])
H2("3.11 Where the score is lost, by true group size (held-out, final pipeline)")
T([["True matches per S1", "Entities", "Mean F0.5", "Share not perfect", "Score lost (of 0.0136 total)"],
   ["0 (singleton)", "24,797", "0.9882", "1.2%", "0.0007"],
   ["1", "23,844", "<b>0.9611</b>", "4.2%", "0.0021"],
   ["2", "75,040", "0.9836", "7.6%", "0.0028"],
   ["3", "106,526", "0.9877", "10.6%", "<b>0.0030</b>"],
   ["4", "96,957", "0.9888", "13.7%", "0.0025"],
   ["5", "63,982", "0.9895", "16.3%", "0.0015"],
   ["6 or more", "50,639", "0.9901", "20.1%", "0.0011"]], [3.4, 2.4, 2.4, 3.3, 5])
B(["<b>1-record entities have the lowest average</b> (0.961): a single miss or a single wrong addition costs the entity 0.44-1.0.",
   "<b>Large groups are rarely perfect</b> (20% of 6+ groups have some error) but each error is cheap, so they lose little.",
   "<b>Most of the score is lost in the common sizes 2-4</b>, simply because most entities are that size.",
   "Singletons are handled well: only 1.2% receive a wrong merge, thanks to the expected-F rule's 'predict nothing' option."])
H1("4. The dataset: every file, its size, structure and real rows")
P("All files are tab-separated text (TSV) with a header row, UTF-8 encoded. Tabs are used because addresses and ID lists contain commas. "
  "Row counts exclude the header.")
T([["File", "Rows", "Size", "Countries (rows)"],
   ["train/train_source1.tsv", "2,206,821", "210 MB", "US 1,323,633 · India 883,188"],
   ["train/train_source2.tsv", "5,034,616", "489 MB", "US 3,016,817 · India 2,017,799"],
   ["train/train_source3.tsv", "5,285,603", "504 MB", "US 3,170,056 · India 2,115,547"],
   ["train/train_ground_truth.tsv", "2,206,821", "127 MB", "one row per training S1 entity"],
   ["test/test_source1.tsv", "1,732,544", "175 MB", "India 809,986 · US 663,106 · <b>France 259,452</b>"],
   ["test/test_source2.tsv", "4,887,273", "509 MB", "India 2,312,565 · US 1,871,330 · France 703,378"],
   ["test/test_source3.tsv", "5,082,316", "506 MB", "India 2,405,000 · US 1,945,701 · France 731,615"],
   ["<b>Total</b>", "<b>26.4 M rows</b>", "<b>2.5 GB</b>", "train 12.5 M records · test 11.7 M records"]],
  [4.5, 2.4, 1.7, 7.9])
H2("4.1 Columns of a source file")
T([["Column", "Type", "Meaning", "Example"],
   ["entity_id", "string", "unique record ID; the prefix names the source", "S1-925783039 · S2-764573417 · S3-202863386"],
   ["business_name", "string", "the business name, noisy", "Orelee's Barbershop"],
   ["business_address", "string, may be empty", "street, city, state in any order, noisy", "1795 Westchester Drive, High Point, NC"],
   ["country", "string", "open label", "US · India · France"]], [2.9, 2.4, 5, 6.2])
H2("4.2 The first rows of every file (exactly as stored)")
P("Indian-script text cannot be drawn by the PDF fonts, so it is shown in brackets with its transliteration (what our normaliser turns it into).", "small")
H3("train_source1.tsv (the reference list)")
T([["entity_id", "business_name", "business_address", "country"],
   ["S1-925783039", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US"],
   ["S1-773889195", "Prime Money", "17560 Ellis Road, Tahlequah, OK", "US"],
   ["S1-377745466", "B+ Retail Inc", "1712 Montebello Avenue, Phoenix, AZ", "US"]], [2.6, 4.2, 7.9, 1.8])
H3("train_source2.tsv (messy feed: upper-case addresses, abbreviations, Indian scripts)")
T([["entity_id", "business_name", "business_address", "country"],
   ["S2-166376419", "[Devanagari] → <i>ram marketimg praivet limited</i>", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"],
   ["S2-764573417", "-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC", "US"],
   ["S2-639257739", "[Devanagari] → <i>adity proprtij elelpi</i> (Aditya Properties LLP)", "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh", "India"]],
  [2.6, 5.2, 6.9, 1.8])
H3("train_source3.tsv (messy feed: state names spelled out, domains, empty addresses)")
T([["entity_id", "business_name", "business_address", "country"],
   ["S3-202863386", "wilfordhancock.com", "Mack Rd, Haltom City, Texas", "US"],
   ["S3-859268022", "International South Consultants Private Ltd", "<i>(empty)</i>", "India"],
   ["S3-22467283", "LLC Moncada Léarning Center", "5780 Fawn Ct, Fort Worth, Texas", "US"]], [2.6, 5.2, 6.9, 1.8])
H3("train_ground_truth.tsv (the answer key; one row per training S1)")
CODE("source1_entity_id   matched_entity_ids\n"
     "S1-965667           S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364\n"
     "S1-55344266         S2-249013014,S2-197070651,S3-478195123,S3-384364074\n"
     "S1-343815751        S2-790675320,S2-479876582,S3-878454467\n"
     "S1-625699510        (empty: a singleton, 'Engracia Fuhrmann Wells LLC, 3452 Brinkley Road, Unit 201, Temple Hills, MD')")
H3("test_source1.tsv (includes France)")
T([["entity_id", "business_name", "business_address", "country"],
   ["S1-714132312", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX", "US"],
   ["S1-106407869", "Vision Partners Corp", "IA, Iowa City, 1064 Newton Rd, Unit 11", "US"],
   ["S1-156285671", "&lt;&lt; Team Ecole", "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France"]],
  [2.6, 3.6, 8.5, 1.8])
H3("test_source2.tsv and test_source3.tsv")
T([["entity_id", "business_name", "business_address", "country"],
   ["S2-192345572", "Brahma Infosoft", "COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka", "India"],
   ["S2-566025912", "Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"],
   ["S2-158121477", "SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord", "France"],
   ["S3-462677478", "[Devanagari] → <i>modrn phainems</i> (Modern Finance)", "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR", "India"],
   ["S3-374810425", "Shri Sai Infratech Co", "3/115, East Delhi, DL", "India"],
   ["S3-198586129", "Fractales Amis Groupe S.A.S", "23 Rue Icmre, La Teste-de-buch, Gironde", "France"]], [2.6, 4.6, 7.5, 1.8])
H3("Our output files (first rows of the submitted v10c files)")
CODE("candidate_pairs.tsv\n"
     "source1_entity_id   candidate_entity_ids\n"
     "S1-714132312        S2-187020300,S2-637340732,S3-625880872,S3-867809779\n"
     "S1-106407869        S2-705547832,S3-585937637,S3-613056593\n\n"
     "matching_results.tsv  (same S1 rows, same order; every match is also a candidate)\n"
     "source1_entity_id   matched_entity_ids")
H2("4.3 Real match groups: what 'the same business' looks like")
H3("US: five records, one business")
T([["Record", "Name", "Address"],
   ["<b>S1</b>", "Geis Assets Inc.", "IN, 7857 Melbourne Road, Indianapolis"],
   ["S2", "GEIS  ASSETS", "#7857 MELBOURNE RD, IN, INDIANAPOLIS"],
   ["S2", "Geis Inc. Assets", "#7857 MELBOURNE RD, INDIANAPOLIS, IN"],
   ["S2", "The Geis Assets Inc.", "#7857 MELBOURNE ROAD, INDIANAPOLIS, IN"],
   ["S3", "Geis Assets (Inc.)", "Melbourne Rd, Indianapolis Towship, Indiana"],
   ["S3", "Geis-Assets Inc.", "Melbourne Road, Indianapolis Towship, Indiana"]], [2, 5.8, 8.7])
P("Word re-ordering, legal-suffix noise, '#' prefixes, abbreviations (RD/ROAD), a missing house number, state code vs state name, a typo "
  "('Towship'). All five must be collected.", "small")
H3("India: Telugu script and legal forms")
T([["Record", "Name", "Address"],
   ["<b>S1-860376386</b>", "White Industries Private Limited", "3D, Parameswara Apts, 6-3-626, Anand Nagar Khairatabad, Hyderabad, Telangana"],
   ["S2-478616453", "[Telugu] → <i>vait imdstris praivet limited</i>", "#3D, ANAND NAGAR KHAIRATABAD, HYDERABAD, Telangana"],
   ["S3-767728003", "M/s White  Industries Private Ltd", "Anand Nagar Khairatabad, Hyderabad, Andhra Pradesh, ##3D"]], [2.8, 5.5, 8.2])
P("The Telugu name transliterates to 'vait imdstris praivet limited': phonetically White Industries Private Limited, but sharing almost no "
  "letters with it. Also note the <b>wrong state</b> (Andhra Pradesh instead of Telangana) and the unit '3D' moved to the end.", "small")
H3("India: mixed scripts inside one name")
T([["Record", "Name", "Address"],
   ["<b>S1-416520778</b>", "Golden Gold Services", "Plot No 26 &amp; 26A Roz Ka Meo, Industrial Area, Phase-I, Sohna, Mewat, Haryana"],
   ["S2-68476347", "Golden Gold Services Company", "PLOT NO 26 &amp; 26A ROZ KA MEO, INDUSTRIAL AREA, PHASE-I, SOHNA, Haryana"],
   ["S3-283001366", "Golden [Devanagari] → <i>Golden gold srvisej</i>", "#26 &amp; 26A Roz Ka Meo, Mewat, Sohna, [Haryana in Devanagari]"],
   ["S3-367341170", "[Devanagari] → <i>goldn gold srvisej</i>", "Mewat, Door No 26 &amp; 26A Roz Ka Meo, Industrial Area, Phase-i, HR"]], [2.8, 5.5, 8.2])
H2("4.4 Twins versus typos (real training rows)")
P("Pairs with the <b>same name and the same street but a different house number</b>. Left: S1 · right: S2 record.")
T([["S1 record", "S2/S3 record", "Truth", "Why"],
   ["Adams Made, Inc. · 833 Gostlin Street, Hammond, IN", "Adams Made, Inc · 84 Gostlin Street, HAMMOND, IN", "<font color='#b91c1c'><b>different</b></font>", "twin business on the same street"],
   ["Oak Ridge Urgent Care Group · 372 East Drive, Oak Ridge, TN", "same name · 393 EAST DRIVE, OAK RIDGE, TN", "<font color='#b91c1c'><b>different</b></font>", "twin"],
   ["St. Zion · 2657 Winding Way Road, Culleoka, TN", "@stzion · 657 WINDING WAY ROAD, CULLEOKA, TN", "<font color='#15803d'><b>same</b></font>", "first digit dropped = noise"],
   ["Langston and Lopez, LLC · 2007 18th Street, Moorhead, MN", "langston and lopez, llc · 200 18TH ST, MOORHEAD, MN", "<font color='#15803d'><b>same</b></font>", "last digit dropped = noise"],
   ["447 Chatterton Drive, Virginia Beach City, VA", "Kogyo, LLC · 44 CHATTERTON DR, VIRGINIA BEACH CITY, VA", "<font color='#15803d'><b>same</b></font>", "truncated number"],
   ["Andrews and Lima · 1020 Grace Street, Lombard, IL", "Andrews and Lima · 1032 GRACE STREET, LOMBARD, IL", "<font color='#15803d'><b>same</b></font>", "hard: looks exactly like a twin"]],
  [5.1, 5.1, 1.7, 4.6])
P("Across training, among 80,117 US pairs with identical name and street but different house numbers, 97.7% are true matches: most "
  "number differences are noise. The rare exceptions (twins) are exactly where false merges come from. Section 22 shows which kinds of "
  "digit change mean 'twin'.", "small")
H2("4.5 Group sizes and structure (training)")
T([["Statistic", "Value"],
   ["True links (S1 → S2/S3)", "7,638,365 links, 7,638,365 distinct S2/S3 IDs → each record belongs to at most one S1"],
   ["Matches per S1", "0 to 11, mean 3.46; 3 is the most common size (530,841 entities)"],
   ["Singletons (S1 with no match)", "5.6%"],
   ["S2/S3 records that match nothing (distractors)", "S2 26.6%, S3 25.4% (about 26%)"],
   ["S2/S3 records per S1", "train 4.68 · test 5.75 (test has more distractors: estimated ~40%)"],
   ["Country inside a true match", "always identical (0.0% mismatch)"]], [6, 10.5])

H1("5. What exploring the data revealed")
T([["Finding", "Evidence", "Design consequence"],
   ["Each S2/S3 record belongs to at most one S1", "7,638,365 links, 7,638,365 distinct targets",
    "attach each record only to its best S1 (argmax); competition between candidates becomes a feature"],
   ["Country always agrees inside a match", "0.0% mismatch", "blocking keys are scoped by country, used as an opaque string"],
   ["About 26% of train S2/S3 records match nothing", "S2 73.4%, S3 74.6% matched", "the model must learn to say 'no'"],
   ["Groups of 0-11 records, mean 3.46", "5.6% singletons", "decide per entity, handle singletons explicitly"],
   ["Indian scripts", "5-11% of S2/S3 names non-Latin: Devanagari, Bengali, Gujarati, Odia, Tamil, Kannada, Malayalam, Telugu",
    "transliterate with anyascii, compare phonetic skeletons"],
   ["Empty addresses", "~3.4% of S2/S3 rows; <b>97.7% of them are true matches</b>", "a name-only matching path; emptiness is a feature"],
   ["Look-alike twins", "same name, nearby number (3460 vs 3462 Goodwin St)", "twin detection at group level (stage 2)"],
   ["Test is different", "adds France (15% of S1), more India (47% vs 40%), 5.75 vs 4.68 records per S1",
    "validation alone cannot be trusted; calibrate for the shift"]],
  [4.2, 5.8, 6.5])
H2("Noise catalogue: what was done to the records")
T([["Field", "Noise patterns observed"],
   ["Name", "word re-ordering (EE Co Legal) · legal-suffix swaps (Inc/LLC/Corp/Pvt Ltd/SARL/SASU) · injected generic words (Center, Partners, "
    "Downtown, Services) · random accents (QÚALITY, Àmicale) · character typos (Auxilgiaare) · domain forms (eelegal.com, wilfordhancock.com) · "
    "glued names (#nórthernosisko, @stzion) · OCR-style digits (m0unt 8aptist) · DBA/brand names · native-script transliterations "
    "(praivet limited = Private Limited) · junk prefixes ('--', '&lt;&lt;')"],
   ["Address", "abbreviations (Rd/St/Ave; R./Av./Imp in French) · '#' and '00' prefixes (001405 CLUB HOUSE DRIVE) · component re-ordering · "
    "missing house numbers · number typos (330 → 33) · state name ↔ code ↔ native script (Maharashtra ↔ MH ↔ Devanagari) · wrong state · "
    "city typos (APOTS for Aptos) · NULL tokens · landmarks (Near SBI ATM) · empty address"]], [2.4, 14.1])
CALL("<b>The central difficulty.</b> True matches and distractors are produced by overlapping noise. A genuine record can have a "
     "changed house number (a typo), and a distractor can be a near-perfect copy of a real business with the number changed. No single "
     "similarity score separates them; the evidence has to be combined, compared across competitors, and checked against the rest of the group.")

# ====================================================================================== PART II
PART("II", "What we built", "The pipeline stage by stage: what each stage does, how it works, why it exists, and the numbers it produces.")

H1("6. Architecture and the funnel of numbers")
P("The pipeline is a <b>cascade</b>: cheap, high-recall steps first, then increasingly expensive and precise models on fewer and fewer pairs.")
story.append(pipeline_diagram())
SP(8)
H2("The funnel (test set)")
T([["Step", "Pairs / items", "Per S1 entity", "What shrinks it"],
   ["All possible pairs", "1.73 M × 9.97 M ≈ 17.3 trillion", "5.75 M", "nothing: impossible to score"],
   ["Key blocking (top-10 per record)", "94.7 M pairs", "54.7", "shared hashed keys"],
   ["Expansion for weak records (v10b/v10c)", "+72.5 M (ranks 11-30) and India ranks 31-50, re-scored", "(temporary)", "only records with best p1 < 0.5"],
   ["Learned pruning = candidate_pairs.tsv", "<b>6,595,983 pairs</b>", "<b>3.81</b>", "best S1 per record, p1 ≥ 0.02"],
   ["Uncertain pairs → cross-encoders", "1.11 M (+49,075 changed by expansion)", "0.64", "0.02 ≤ p1 ≤ 0.98"],
   ["Final matches = matching_results.tsv", "<b>5,772,673</b>", "<b>3.33</b>", "stacker, veto, calibration, expected-F0.5"]],
  [5.1, 5.2, 2.2, 4])
H2("Stage summary")
T([["Stage", "Input → output", "Main algorithm", "Script"],
   ["0 Normalise", "raw text → canonical tokens", "vectorised regex (polars), anyascii transliteration", "normalize.py, prep.py"],
   ["1 Blocking", "records → candidate pairs", "hashed multi-key inverted index, IDF-weighted scoring", "blocking.py"],
   ["2 Features", "pair → 72 numbers", "rapidfuzz string metrics, sparse TF-IDF cosine, digit logic", "features.py"],
   ["3 Stage 1", "features → p1; prune", "LightGBM, 2-fold cross-fitting, argmax assignment", "model.py, run_pipeline.py"],
   ["4 Cross-encoders", "raw text of uncertain pairs → ce", "fine-tuned MiniLM-L6 transformers (4 models)", "ce_train.py, ce_score.py"],
   ["5 Stacker", "p1, margin, ce, ce7, 72 features → ps", "LightGBM stacker (76 inputs)", "ce_stack_train.py"],
   ["6 Stage-2 veto", "group context → p2", "LightGBM on 27 group / twin features", "stage2.py"],
   ["7 Calibration", "p → shift-corrected p", "Saerens-Latinne-Decaestecker EM", "model.py, ce_predict.py"],
   ["8 Decision", "p → final sets", "per-entity expected-F0.5 maximisation", "model.py"],
   ["Expansion", "weak records → deeper candidates", "same blocking with top-30/top-50, re-scoring, CE", "ext_*.py"]],
  [2.6, 4.3, 6, 3.6])

H1("7. Stage 0 · normalisation")
P("<b>Goal:</b> make two spellings of the same thing identical, so that simple comparisons (and hashed keys) work. It is fully "
  "vectorised in polars and takes about 3.5 minutes for 12.5 M records.")
T([["Step", "What it does", "Example"],
   ["Transliteration", "anyascii maps every non-Latin character to Latin letters (applied only to non-ASCII strings)", "Devanagari 'Private Limited' → praivet limited"],
   ["Lower-casing, accents", "case and accents removed (accents via transliteration)", "Établissements → etablissements, QÚALITY → quality"],
   ["Legal canonicalisation", "map legal-form variants to one form, then separate them from the core name", "pvt / praivet / praibhet / pra → private; ltd / limitet → limited; elelpi → llp"],
   ["Core name", "drop legal words and stop words (the, and, of, de, du, la ...)", "'The Geis Assets Inc.' → 'geis assets'"],
   ["Concatenated name", "glue the core tokens together", "eelegal.com → eelegal = 'ee legal' glued"],
   ["Phonetic skeleton", "merge aspirates (bh → b), drop vowels after the first letter, collapse repeats", "bharat green / bharat grin → brtgrn"],
   ["Address short forms", "street/saint → st, road → rd, rue → r, avenue → av ...", "same canonical form on both sides, any country"],
   ["State phrases", "state names, native-script names and codes → one code", "Maharashtra / mharastr / MH → mh"],
   ["Numbers", "digit runs extracted, leading zeros stripped, split from letters", "'#00540-B' → 540 and b; 001405 → 1405"],
   ["OCR repair (v3+)", "digits inside letter tokens → look-alike letters", "m0unt 8aptist → mount baptist"]],
  [3.2, 7.3, 6])
P("Real output of our normaliser for two records of the same business:")
CODE("input : 'Geis Assets Inc.'   | 'IN, 7857 Melbourne Road, Indianapolis'\n"
     "input : 'GEIS  ASSETS'       | '#7857 MELBOURNE RD, IN, INDIANAPOLIS'\n"
     "both  → core name 'geis assets' · concatenated 'geisassets' · skeleton 'gsts' · token skeletons [gs, asts]\n"
     "        numbers [7857] · rare address tokens [melbourne, indianapolis] · legal form 'inc' vs '' (kept as a feature)")
INTUITION("normalisation removes the differences that carry no information, so that the models can spend their capacity on the "
          "differences that do (a changed house number, a different legal form, an extra distinctive word).")

H1("8. Stage 1 · blocking (candidate generation)")
P("Comparing every S2/S3 record with every S1 would mean about 17 trillion pairs. <b>Blocking</b> proposes a short candidate list per "
  "record using <b>keys</b>: two records become candidates only if they share at least one key. It works like the index at the back of a "
  "book: instead of reading every page, you look up a word and jump to the pages that contain it.")
H2("The seven key families")
T([["Key", "Built from", "Catches", "Illustration for 'Geis Assets Inc., 7857 Melbourne Road'"],
   ["A", "pair of name tokens (phonetic skeletons)", "word re-ordering, extra injected words", "US | gs + asts"],
   ["B", "name token × address number", "noisy name but correct house number", "US | geis + 7857"],
   ["C", "address number × address token", "transliterated or brand (DBA) names", "US | 7857 + melbourne"],
   ["D", "first 8 characters of the concatenated core name", "domains, glued names", "US | geisasse"],
   ["E", "first 7 characters of the phonetic skeleton", "typos, transliteration", "US | gsts"],
   ["F", "name token × address token", "missing house numbers", "US | geis + melbourne"],
   ["G", "pair of address tokens", "address-only evidence without numbers", "US | melbourne + indianapolis"]], [1, 5, 5, 5.5])
B(["<b>Hashing.</b> Each key string is hashed to a 64-bit integer. Integers are cheap to store, sort and join; 48 M S1 key rows fit in memory.",
   "<b>Country prefix.</b> Every key starts with the country string, so a French record can only meet French S1 entities. The label is "
   "opaque, so the unseen 'France' works without any code change.",
   "<b>Rarest tokens.</b> Each record uses only its 4 rarest name tokens and 5 rarest address tokens (rarity = document frequency among S1 "
   "records of that country). Rare tokens are distinctive ('gostlin' says more than 'street'), and choosing by rarity makes the key "
   "choice stable when the words are re-ordered.",
   "<b>Block cap.</b> A key shared by more than 200 S1 records (such as 'US | main + st') is too generic and is dropped.",
   "<b>Scoring.</b> Each (record, S1) pair gets Σ over shared keys of 1/√(block size): a key shared by few S1 entities is strong evidence "
   "(an IDF-like weight). Each record keeps its <b>top 10</b>. The per-family hit counts become model features."])
H2("How blocking recall improved")
T([["Design", "recall@1", "recall@10", "recall@50", "Note"],
   ["v0: name + number keys only (cap 60)", "0.867", "~0.93", "0.944", "5.6% of true links unreachable"],
   ["+ address-token keys F, G and rarest-token selection (cap 60)", "0.946", "0.975", "0.982", "the main fix"],
   ["cap 200 (final)", "0.951", "0.978", "0.987", "used in all submitted versions"]], [6.2, 1.6, 1.7, 1.7, 5.3])
P("<b>Recall@k</b> = share of true links whose S1 is among the record's top-k candidates. On the validation fold, a perfect matcher "
  "working on our candidates would score macro F0.5 <b>0.9926</b>: this is the <b>blocking ceiling</b>. A learned ranker over key-hit "
  "counts was tried to rescue true matches ranked below 10th; it ranked worse than the simple IDF score (AUC 0.92 vs 0.99), so it was dropped.")

H1("9. Stage 2 · the 72 pair features")
P("For every candidate pair we compute 72 numbers that describe how similar the two records are, and how this candidate compares with "
  "the record's other candidates. Throughput: about 69,000 pairs per second on 12 cores.")
T([["Group", "Features", "Why"],
   ["Name strings", "rapidfuzz ratio, token-sort, token-set and partial ratio on core names; Jaro-Winkler and partial ratio on the "
    "concatenated name; ratio and partial on the phonetic skeleton", "robust to re-ordering, typos, extra words, domains, transliteration"],
   ["Name statistics", "word TF-IDF cosine, token Jaccard, common-token count, token counts, exact / prefix equality, IDF mass of each name, "
    "legal-form equality and conflict", "a distinctive name ('Geis Assets') deserves more trust than a generic one ('Family Specialists')"],
   ["Address", "TF-IDF cosine, ratio, token-set, partial, token Jaccard, token counts, IDF mass, empty-address flag", "street-level agreement"],
   ["House numbers", "first-number equality, fuzzy ratio, log |difference|, same length, prefix, suffix, containment, set Jaccard and "
    "coverage both ways", "the strongest identity signal in an address"],
   ["Digit position (v6)", "Hamming distance, position of the first differing digit from the left and from the right, 'last digit ±≤2' flag",
    "noise hits leading digits; twins differ in trailing digits"],
   ["Blocking evidence", "block score, block rank, hits per key family A-G", "how strongly the index linked them"],
   ["Competition", "for name / address cosine, token-set, block score and number Jaccard: gap to the best candidate and margin over the "
    "second best; number of candidates; how many records chose this S1 first", "relative evidence: 'is this candidate better than its rivals?'"],
   ["Other", "non-Latin-script flag, source (S2 vs S3). Deliberately <b>no country feature</b>", "must transfer to the unseen France"]],
  [2.7, 9.3, 4.5])
INTUITION("absolute similarity asks 'how alike are these two?'. Competition features ask 'is this the best explanation for this record?'. "
          "Because each record belongs to at most one S1, the second question is often more decisive: the top features by importance "
          "are all relative (margin over the second-best candidate, block rank, gap to the best).")

H1("10. Stage 3 · stage-1 LightGBM, assignment and learned pruning")
B(["<b>Model.</b> A LightGBM binary classifier: an ensemble of about 1,500-2,000 small decision trees (127 leaves each, learning rate "
   "0.05, early stopping) that outputs p1, the probability that the pair is a true match.",
   "<b>Training data.</b> 1.2 M sampled S2/S3 records per fold, about 11 M candidate pairs each, labelled from the answer key.",
   "<b>2-fold cross-fitting (v3+).</b> Records are hashed into two folds. Model A trains on fold 0, model B on fold 1. Every training pair "
   "is scored by the model that did <i>not</i> see it, which gives honest (out-of-fold) probabilities for all 98 M training pairs. "
   "Later stages train on these honest scores. Test pairs get the average of A and B.",
   "<b>Assignment.</b> Each S2/S3 record keeps only its highest-probability S1 (argmax), because a record belongs to at most one S1.",
   "<b>Learned pruning.</b> The argmax pairs with p1 ≥ 0.02 form the candidate set: 3.81 per S1 on test. On training data it still "
   "contains 96.97% of all true links, while the final pipeline recalls 95.9%, so the pruning costs almost nothing."])
T([["Top stage-1 features by gain", ""],
   ["1. block-score margin over the 2nd-best candidate", "5. number-set Jaccard"],
   ["2. block rank", "6. address token-set ratio (and its gap to the best)"],
   ["3. log house-number difference", "7. name skeleton ratio"],
   ["4. block-score gap to the best candidate", "8. legal-form equality · name IDF mass · name ratio"]], [8.2, 8.3])
INTUITION("a gradient-boosted tree ensemble is a committee of simple if-then rules, each one built to fix the mistakes of the ones before "
          "it. It excels at combining dozens of heterogeneous numbers (ratios, counts, flags) and needs no feature scaling.")

H1("11. Stage 4 · transformer cross-encoders")
P("By v6, hand-made features had plateaued (v5 and v6 each moved validation by about +0.0002). The remaining errors needed a model that "
  "<b>learns</b> the noise directly from the raw text. That model is a <b>cross-encoder</b>.")
H2("What a cross-encoder is")
B(["A <b>transformer</b> is a neural network that reads a sequence of tokens and lets every token look at every other token "
   "(<b>self-attention</b>). This is how it can notice that '657' on one side corresponds to '2657' on the other, or that 'vait imdstris' "
   "sounds like 'white industries'.",
   "A <b>cross-encoder</b> reads <b>both records together</b> as one sequence and outputs a single match score. Because the two texts attend "
   "to each other, it can compare them word by word. (A <i>bi-encoder</i> would encode each record separately into a vector; it is faster "
   "for search but weaker for fine-grained decisions.)",
   "Input format: " + M("[CLS] name | address of S1 [SEP] name | address of the S2/S3 record [SEP]") + ", cut to 80 tokens."])
H2("Our models")
T([["Setting", "Value"],
   ["Base model", "cross-encoder/ms-marco-MiniLM-L-6-v2: 6 transformer layers, hidden size 384, 22.7 M parameters, Apache-2.0. "
    "Pre-trained on MS MARCO (search-query / passage relevance), so it already knows how to compare two texts"],
   ["Fine-tuning data", "the <b>uncertain</b> training pairs (0.02 ≤ p1 ≤ 0.98): 887k pairs, 49% positive. Test: 1.11 M pairs"],
   ["v7 models", "2 models (one per cross-fitting fold), 90k sampled pairs each"],
   ["v8 models", "2 models, 220k sampled pairs each (more data → better)"],
   ["Training", "1 epoch, batch 32, AdamW (learning rate 3e-5, weight decay 0.01), one-cycle schedule with 6% warm-up, binary "
    "cross-entropy on the logit, CPU (hours per model)"],
   ["Cross-fitting", "the fold-0 model scores fold-1 pairs and vice versa, so every training score is out-of-fold; test uses the mean of both"],
   ["Why only uncertain pairs", "pairs with p1 > 0.98 or < 0.02 are already decided; nearly all remaining errors live in this uncertain band (~17% of candidate pairs)"]], [3.6, 12.9])
H2("How much the cross-encoders helped (AUC on held-out uncertain pairs)")
T([["Model", "AUC", "Reading"],
   ["stage-1 LightGBM alone", "0.913", "the hand-made features on the hardest pairs"],
   ["v7 cross-encoder alone (90k/fold)", "0.926", "the text model alone is already better"],
   ["stage 1 + v7 stacked", "0.969", "they make different mistakes, so combining helps a lot"],
   ["v8 cross-encoder alone (220k/fold)", "0.952", "more training pairs → a stronger text model"],
   ["stage 1 + v7 + v8 stacked (final)", "<b>0.981</b>", "errors on uncertain pairs cut by more than half"]], [6.5, 2, 8])
INTUITION("the tree model sees 72 summary numbers; the cross-encoder sees the actual characters. The tree model knows the <i>context</i> "
          "(competitors, block evidence); the cross-encoder knows the <i>text</i>. Stacking both is better than either.")

H1("12. Stage 5 · the stacker")
P("A <b>stacker</b> is a second-level model whose inputs include the outputs of other models. Ours (stacker3) is a LightGBM "
  "(31 leaves, learning rate 0.03) with <b>76 inputs</b>: logit(p1), the margin over the record's second-best candidate, logit(v8 "
  "cross-encoder), logit(v7 cross-encoder), plus all 72 pair features. It is trained on the out-of-fold scores of the uncertain training "
  "pairs and outputs <b>ps</b>. Pairs that were not uncertain keep ps = p1.")
B(["<b>Why logits?</b> logit(p) = ln(p / (1 − p)) stretches probabilities near 0 and 1 (0.99 vs 0.999 becomes 4.6 vs 6.9), which helps "
   "the trees split on confident scores.",
   "<b>Why the raw features again?</b> The stacker learns <i>when</i> to trust the cross-encoder. Example: when the text is nearly identical "
   "but the last house-number digit differs by 2, the cross-encoder tends to say 'match'; the digit features let the stacker overrule it."])

H1("13. Stage 6 · the group / twin veto")
P("After assignment every S1 has a <b>group</b> of attached records. Error analysis showed that most wrong merges are <b>twins</b>: another "
  "business with the same name at a nearby address, not in S1 itself, but with its own records in S2/S3. The key insight:")
CALL("A true record with a house-number typo is a <b>one-off</b>. A twin's records <b>agree with each other</b>. If two attached records "
     "both say '84 Gostlin Street' while S1 and the other members say '833', those two are probably a different business.")
P("For every attached record, stage 2 computes 27 group features, including:")
B(["<i>g_twin_num</i>: how many other members share my house number while it disagrees with S1's number",
   "how many members, and how much probability mass, support S1's own number",
   "exact name / address agreement counts; the best probability among records sharing my number",
   "group size, sum / max of the other members' probabilities, my rank and relative probability in the group"])
P("A LightGBM (63 leaves) trained on the out-of-fold stage-1 groups predicts <b>p2</b>. How p2 is used changed over the versions:")
T([["Rule", "Formula", "Where", "Why"],
   ["Promote allowed (v3)", "final = p2", "-", "raised validation, <b>lowered</b> the leaderboard (0.972 → 0.965): see section 20"],
   ["Full veto (v4a-v6)", "final = min(ps, p2)", "all countries", "the group model may lower a score, never raise it"],
   ["Soft veto (v9+)", "if p2 ≥ 0.1: final = ps, else min(ps, p2)", "US, India", "with the cross-encoders, only strong group evidence should overrule the pair score"],
   ["Full veto (v9+)", "final = min(ps, p2)", "France (and any unseen country)", "France probe (0.978) showed its borderline pairs are mostly wrong"]],
  [3, 4.1, 3.3, 6.1])

H1("14. Stage 7 · calibration under test shift")
P("A probability is <b>calibrated</b> when, among all pairs scored 0.8, about 80% are true. On training data our model is well calibrated "
  "(mean p of best attachments 0.950 vs true rate 0.951 for US; 0.929 vs 0.931 for India). But the test set has more look-alike records, "
  "so the same score means a lower true-match chance there. This is <b>label shift</b>: the share of positives changes between train and test.")
H2("The EM label-shift method (Saerens, Latinne and Decaestecker, 2002)")
P("It estimates the test positive rate π_test from the model's own test predictions, without any labels, by iterating:")
CODE("start: π_test = π_train\n"
     "repeat until stable:\n"
     "   adjust each p:  p' = (π_test/π_train)·p / [ (π_test/π_train)·p + ((1−π_test)/(1−π_train))·(1−p) ]\n"
     "   π_test = mean of p' over the country's best attachments\n"
     "then rescale every score:  odds_new = odds_old × [π_test/(1−π_test)] / [π_train/(1−π_train)]")
T([["Country", "Train positive rate", "Test (EM estimate)", "Odds factor"],
   ["US", "0.951", "0.914", "×0.55"], ["India", "0.931", "0.873", "×0.51"],
   ["France (unseen)", "pooled 0.943", "0.891", "×0.50, then an extra ×0.5 (v10a) → about ×0.25"]], [4, 3.5, 3.5, 5.5])
P("<b>Worked example (US):</b> a pair scored 0.80 has odds 0.80/0.20 = 4. Multiply by 0.55 → 2.2 → probability 2.2/3.2 = 0.69. "
  "Borderline pairs are pushed down; confident pairs (0.999 → 0.998) barely move.")
B(["<b>Estimated on stage-1 scores.</b> The EM prior must be estimated on the raw stage-1 scores. Estimating it on stacked scores was a "
   "bug that inflated India's factor to ×2.24 (section 21).",
   "<b>The extra France factor</b> (v10a) came from a leaderboard probe: making France stricter gained +0.0002, while making US or India "
   "stricter (×0.7) lost a little."])

H1("15. Stage 8 · the per-entity expected-F0.5 decision")
P("For each S1, sort its attached records by final probability p_1 ≥ p_2 ≥ ... and choose how many to keep (m) by maximising a plug-in "
  "estimate of the expected F0.5:")
CODE("S_m = p_1 + ... + p_m (expected true positives kept)     T = sum of all p (expected true matches in the group)\n"
     "E[F](m) ≈ 1.25·S_m / (1.25·S_m + 0.25·(T − S_m) + (m − S_m))      for m ≥ 1\n"
     "E[F](0) = (1 − p_1)·(1 − p_2)·...   (the chance that the entity is really a singleton)\n"
     "→ keep the m with the highest value")
H2("Worked example: four attached records with p = 0.99, 0.97, 0.60, 0.20 (T = 2.76)")
T([["Keep m", "S_m", "Expected F0.5", ""],
   ["0", "-", "0.0001", "the group is almost certainly not empty"],
   ["1", "0.99", "0.732", ""],
   ["<b>2</b>", "1.96", "<b>0.911</b>", "<b>best: keep the two sure records</b>"],
   ["3", "2.56", "0.867", "the 0.60 record is rejected even though p > 0.5"],
   ["4", "2.76", "0.736", ""]], [2, 2, 3, 9.5])
INTUITION("a single global threshold treats each pair alone. The expected-F rule looks at the whole group: a 60%-likely record is not "
          "worth risking next to two sure ones (a wrong addition costs 4× a miss), but in a group with nothing else it might be. It needs "
          "no tuned threshold, which matters for France, where there is no data to tune one.")

H1("16. Candidate expansion for weak records (v10b, v10c)")
P("The loss analysis (section 23) showed that <b>blocking misses</b> were the largest remaining error: true matches whose S1 was never in "
  "the record's top-10. Widening blocking for everyone would multiply the work and the candidate set. Instead we widened it only where "
  "it was needed.")
B(["<b>Who is 'weak'?</b> Records whose best stage-1 probability is below 0.5: their current best S1 is probably wrong or missing. "
   "Test: 4,059,676 such records; train: 2,892,045.",
   "<b>v10b.</b> For weak records, blocking is re-run with top-30 instead of top-10. This adds 72.5 M new pairs (ranks 11-30) on test. "
   "The expanded groups are re-scored with the same stage-1 models (competition features change when there are more rivals), and each "
   "record again keeps only its single best S1.",
   "<b>v10c.</b> India had the most blocking misses (transliteration). Weak India records (1,986,498 on test) get a second expansion to "
   "top-50 (ranks 31-50).",
   "<b>Only changed records are re-processed.</b> If a record's best S1 changed (262,946 on test in v10c) and the new pair is uncertain "
   "(49,075), it is sent through the cross-encoders and the stacker. Every other record keeps its exact previous scores.",
   "<b>The candidate set does not grow.</b> A record still contributes at most one candidate, so the set stays at 3.81 per S1."])
T([["Version", "Held-out F0.5", "India", "US", "Leaderboard"],
   ["v10a (no expansion)", "0.98528", "0.98064", "0.98838", "0.979396"],
   ["v10b (top-30 for weak records)", "0.98612", "0.98240", "0.98860", "0.980128"],
   ["v10c (+ India top-50)", "<b>0.98637</b>", "<b>0.98302</b>", "0.98860", "<b>0.980385</b>"]], [5.5, 2.8, 2.6, 2.6, 3])

H1("17. The life of one entity: a France example end to end")
P("Test entity <b>S1-596389295</b>, 'Établissements Bien SARL, 64 Rue Auguste Lamy, Lille, Hauts-de-France'. France never appears in "
  "training. These are the real records and the real scores from our pipeline.")
T([["Record", "Name · address", "Stage-1 p1", "Cross-encoder (v8 / v7)", "Final"],
   ["S2-600405584", "Établissements Bien SARL · Hauts-de-France, LILLE, 64 RUE AUGSUTE LAMY", "0.99988", "not needed (sure)", "<font color='#15803d'><b>match</b></font>"],
   ["S3-484894381", "Établissements Bien (SARL) · R Auguste Lamy, Lille", "0.99993", "not needed (sure)", "<font color='#15803d'><b>match</b></font>"],
   ["S2-560120046", "établissements Bien SASU · 66 RUE AUGUSTE LAMY, LILLE", "0.024", "0.94 / 0.96", "<font color='#b91c1c'><b>reject</b></font>"],
   ["S2-835829267", "établissements Bien SARL · <i>(empty address)</i>", "0.041 → 0.063 after expansion", "0.71 / 0.69", "<font color='#b91c1c'><b>reject</b></font>"]],
  [2.4, 6.3, 2.6, 2.7, 2.5])
B(["<b>Normalisation</b> removes accents and case (É → e), turns 'Rue' / 'R' into the same token, fixes nothing about 'AUGSUTE' (a typo), "
   "and extracts the house numbers 64 and 66.",
   "<b>Blocking</b> finds the S1 for the first three records through shared keys (name tokens, street tokens, the house number). "
   "The empty-address record only has its name; 'Établissements Bien' is shared by other S1 entities, so its best candidate was a different "
   "S1 (p1 = 0.041). Being weak, it was <b>expanded</b>: the deeper search found this S1 and it became its new best (p1 = 0.063).",
   "<b>Stage 1</b> is certain about the first two (same name, same number 64 despite the street typo). It gives the SASU record only 0.024: "
   "the house number differs in the last digit (64 vs 66, a twin pattern) and the legal form conflicts (SARL vs SASU).",
   "<b>The cross-encoders</b> read the nearly identical text and say 0.94-0.96 for the SASU record: they are fooled by the surface "
   "similarity. This is why the <b>stacker</b> also sees the digit and legal-form features and the stage-1 score, and brings it back down.",
   "<b>Stage 2</b> sees a group whose two confident members both say 64, while the candidate says 66. In France the veto is full.",
   "<b>Calibration</b> (France: about ×0.25 on the odds) and the <b>expected-F0.5 rule</b> then keep the two sure records and reject the two "
   "doubtful ones: adding a doubtful record to a group of two sure ones is a bad bet under F0.5.",
   "<b>Output</b>: matching_results.tsv lists S2-600405584, S3-484894381; candidate_pairs.tsv lists all four."])

H1("18. How we validated (and why validation alone was not enough)")
B(["<b>Hold out S1 entities, not pairs.</b> The metric is per S1 entity, so validation must contain whole groups, including every record "
   "that could be falsely merged into them.",
   "<b>v1-v2:</b> 2.5% of S1 held out (hash(id) % 40 == 0: 54,940 entities). Every S2/S3 record touching them (2.19 M) was scored with "
   "its full candidate list, and training used only records that never touched them.",
   "<b>v3+:</b> 2-fold cross-fitting gives out-of-fold scores for everything; the later models (stage 2, stacker) are evaluated on "
   "<b>441,785 held-out S1 entities</b> (hash seed 13, id % 5 == 0) that their own training never saw.",
   "<b>Label-free test profiling.</b> Because v3 showed that validation can disagree with the leaderboard, every test output was also "
   "profiled without labels before uploading: share of accepted links whose house number disagrees with S1, share of empty-address "
   "links, share of weak-name links, matches per S1 and singleton rate per country, all compared with the last good version.",
   "<b>Leaderboard probes.</b> A few uploads were used as deliberate experiments (France soft veto, India ×0.7, US ×0.7) to measure "
   "how the test distribution reacts to a single change."])
CALL("<b>Lesson.</b> Held-out validation measures the <i>training</i> distribution. A change that adds links in borderline cases must also "
     "be checked against the test distribution, or it can win on validation and lose on the leaderboard (v3).")

# ====================================================================================== PART III
PART("III", "The journey", "Every version, what it changed, why, what happened, and what we learned, including the failures.")

H1("19. Version history at a glance")
story.append(score_chart())
P("Both lines are macro F0.5 on the same scale. The gap between them is the train → test shift: test is harder (more look-alikes, an "
  "unseen country). The gap shrank from 0.018 (v3) to 0.006 (v10c), mostly thanks to the veto-only rule, the cross-encoders and calibration.", "small")
T([["Ver.", "What was added or changed", "Held-out", "Leaderboard", "Verdict"],
   ["v1", "normalise + 5-key blocking + 63 features + LightGBM + argmax + threshold", "0.9764*", "-", "baseline"],
   ["v2", "numeric house-number features, 1.2 M training records, stage-2 consensus, expected-F rule", "0.9812*", "0.972", "first upload"],
   ["v3", "OCR repair, 2-fold cross-fitting, twin features, stage 2 allowed to <b>promote</b>", "0.9833", "<font color='#b91c1c'><b>0.965</b></font>", "regression"],
   ["v4a", "stage 2 = <b>veto only</b>; learned pruning → 3.8 candidates / S1", "0.9813", "0.973", "fixed, compact"],
   ["v5a", "per-country EM label-shift calibration", "(train unchanged)", "0.973", "neutral"],
   ["v6", "digit-position house-number features", "0.9815", "0.973612", "small gain"],
   ["v7b", "cross-encoder (90k pairs/fold) + 75-input stacker", "0.9840", "0.977", "<b>big jump</b>"],
   ["v9", "+ v8 cross-encoder (220k/fold), ensemble; <b>soft veto</b> in US/India", "0.9853", "0.979189", "gain"],
   ["probe", "soft veto also for France", "-", "0.978", "worse → rejected"],
   ["v10a", "v9 + stricter France calibration (odds ×0.5)", "0.9853", "0.979396", "gain"],
   ["v10b", "candidate expansion: top-30 blocking for weak records", "0.9861", "0.980128", "gain"],
   ["probes", "v10b with India ×0.7 / US ×0.7 calibration", "-", "0.980022 / 0.980105", "worse → rejected"],
   ["<b>v10c</b>", "<b>+ India ranks 31-50 expansion (final)</b>", "<b>0.9864</b>", "<b>0.980385</b>", "<b>best, submitted</b>"]],
  [1.2, 7.8, 2.2, 2.6, 2.7])
P("* v1-v2 validation used the smaller 54,940-entity fold; from v3 on, 441,785 held-out entities.", "small")

H1("20. Every version: what, why, result, lesson")
versions = [
    ("v1 · the baseline",
     "Normalisation, 5 key families, 63 features, one LightGBM, argmax assignment and a global threshold.",
     "We needed an end-to-end, reproducible pipeline for 26 M records on a laptop before optimising anything.",
     "Validation 0.9764 against a blocking ceiling of 0.9926.",
     "Error analysis showed near-copy distractors (same name, house number altered) and genuine records with number typos; the number "
     "features were too weak to tell them apart."),
    ("v2 · numbers, more data, group consensus",
     "Log numeric difference, digit length, prefix / containment features; 50% more training records; a first stage-2 model that checks "
     "whether confident group members agree; the expected-F0.5 decision rule.",
     "House numbers are the strongest identity signal, and the metric is per entity.",
     "Validation 0.9792 (stage 1) and 0.9812 with stage 2; <b>leaderboard 0.972</b> (first upload).",
     "A 0.009 gap between validation and leaderboard appeared: the test set is harder than training."),
    ("v3 · cross-fitting, twin features and a painful lesson",
     "OCR repair, 2-fold cross-fitting (honest scores for all 2.2 M training groups), group-wide twin features, and a stage 2 free to raise "
     "or lower scores.",
     "Make stage 2 learn from all groups and let it rescue weak true records that agree with confident members.",
     "Best validation yet (0.9833) but <b>leaderboard 0.965</b>, a drop of 0.007 (two uploads, both 0.965).",
     "v3 added 128k links; 58.7k had stage-1 p below 0.3 and were <i>promoted</i> by group features ('my name matches confident members'). "
     "In training that pattern means an incomplete genuine record; in test, with far more look-alike clusters, it meant false merges. "
     "Accepted US links with a disagreeing house number rose from 10.8% to 12.2%. <b>Group evidence may veto, never promote.</b>"),
    ("v4a · veto only, and a small candidate set",
     "final = min(p1, p2): stage 2 can only lower scores. Learned pruning: the candidate file became the argmax pairs with p1 ≥ 0.02.",
     "Keep what stage 2 is good at (catching twins) and drop what did not transfer (promotion). Meet the new 'smaller candidate set' rule.",
     "Leaderboard <b>0.973</b>, above both v2 and v3, with 3.8 candidates per S1 instead of 54.7.",
     "Robustness to the shift matters more than validation gains."),
    ("v5a · calibration for the test shift",
     "EM label-shift correction per country.",
     "The model's scores are too optimistic on test (more look-alikes near the boundary).",
     "Leaderboard 0.973 again. The 53k borderline links it removed were about half right and half wrong.",
     "Threshold-style tuning is not where the remaining gap is; the scores themselves need to be better."),
    ("v6 · digit-position features",
     "Which digit of the house number changed: first vs last, Hamming distance, 'last digit ±≤2'.",
     "EDA: a changed first digit means noise (P(true) 0.98); a changed last digit means twin (P(true) 0.33-0.46).",
     "Leaderboard <b>0.973612</b>; validation +0.0002.",
     "The trees had already approximated the pattern. Hand-made features had plateaued: the next step needed a model that reads the text."),
    ("v7b · the cross-encoder",
     "Fine-tuned MiniLM cross-encoders (90k pairs per fold, cross-fitted) re-read the raw text of the uncertain pairs; a 75-input stacker "
     "combines them with stage 1. Proof of concept first: on 20k held-out uncertain pairs, AUC 0.917 → 0.950 with only 30k training pairs.",
     "Nearly all remaining errors lived in the uncertain pairs (~17% of candidate pairs), where the 72 features were ambiguous.",
     "Validation 0.9840 (+0.0025) and <b>leaderboard 0.977 (+0.0034)</b>: the gain grew on test.",
     "Learning from raw text transfers better to the shifted test set than hand-made features."),
    ("v9 · a stronger cross-encoder and a soft veto",
     "v8 cross-encoders trained on 220k pairs per fold, ensembled with v7 in stacker3 (76 inputs). Soft veto in US/India: stage 2 overrules "
     "the stacked score only when it is confident (p2 < 0.1).",
     "More data made the text model much stronger (AUC 0.926 → 0.952); with a better pair score, the veto should only fire on strong evidence.",
     "Validation 0.9853; <b>leaderboard 0.979189</b>.",
     "When the pair model improves, the guard-rails around it should loosen, but only where we have training data."),
    ("France probe · soft veto everywhere",
     "The same soft veto applied to France.", "Check whether France behaves like the training countries.",
     "Leaderboard 0.978 (−0.0012).", "France's borderline pairs are mostly wrong: the unseen country needs stricter rules."),
    ("v10a · stricter France",
     "Full veto for France plus an extra ×0.5 on its calibrated odds.", "Act on the France probe.",
     "Leaderboard <b>0.979396</b> (+0.0002).", "Unseen-domain data deserves more caution."),
    ("v10b · candidate expansion",
     "Weak records (best p1 < 0.5) get top-30 blocking, re-scoring, and cross-encoders for changed pairs.",
     "Blocking misses were the largest loss (0.0076).",
     "Validation 0.9861; <b>leaderboard 0.980128</b> (+0.0007).",
     "Spend compute only where it is needed: 4 M weak records, not 10 M."),
    ("India / US calibration probes",
     "v10b with India or US odds × 0.7.", "Test whether the other countries also need stricter calibration.",
     "0.980022 (India) and 0.980105 (US): both slightly worse.", "Calibration for US/India was already right; stop tuning it."),
    ("v10c · India deeper (final)",
     "Weak India records get a second expansion to ranks 31-50.", "India had the most blocking misses (transliteration).",
     "Validation <b>0.98637</b> (India 0.98240 → 0.98302); <b>leaderboard 0.980385</b>, our best. Submitted.",
     "The held-out gain again predicted the leaderboard gain."),
]
for title, what, why, res, lesson in versions:
    story.append(KeepTogether([Paragraph(title, S["h2"]),
                               Table([[Paragraph("<b>What</b>", S["cell"]), Paragraph(what, S["cell"])],
                                      [Paragraph("<b>Why</b>", S["cell"]), Paragraph(why, S["cell"])],
                                      [Paragraph("<b>Result</b>", S["cell"]), Paragraph(res, S["cell"])],
                                      [Paragraph("<b>Lesson</b>", S["cell"]), Paragraph(lesson, S["cell"])]],
                                     colWidths=[2 * cm, 14.5 * cm],
                                     style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
                                                       ("BACKGROUND", (0, 0), (0, -1), SOFT), ("TOPPADDING", (0, 0), (-1, -1), 3),
                                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))]))

H1("21. Failures, bugs and dead ends (and what each taught us)")
T([["What went wrong", "Symptom", "Fix", "Lesson"],
   ["Stage 2 allowed to promote (v3)", "validation up, leaderboard 0.972 → 0.965", "veto-only rule min(p1, p2)", "group evidence did not transfer to a shifted test set"],
   ["EM prior estimated on stacked scores", "India odds factor inflated to ×2.24 (far too lenient), caught by profiling", "estimate EM on stage-1 scores only",
    "calibration must be estimated on the scores it was designed for"],
   ["Learned blocking ranker", "AUC 0.92 vs 0.99 for the simple IDF score", "kept the IDF score", "simple, well-motivated scores can beat learned ones"],
   ["Soft veto for France (probe)", "leaderboard 0.978 (−0.0012)", "full veto + stricter France calibration", "treat the unseen domain more cautiously"],
   ["India / US ×0.7 probes", "0.980022 / 0.980105 (both below 0.980128)", "no change", "stop tuning what is already right"],
   ["Expansion re-scoring ran out of memory", "crash on 106 M pairs with 16 GB RAM", "rewrote ext_score.py to stream q_rid ranges into parquet parts, resumable, with loud failure markers",
    "design every heavy job to be chunked and restartable"],
   ["Cross-encoder proof-of-concept crash", "an off-by-one in the learning-rate scheduler's step count", "total_steps = steps + 1", "test the training loop on a tiny sample first"],
   ["Hand-made features plateaued (v5, v6)", "+0.0002 each", "switched to cross-encoders", "know when to change the model family, not the features"],
   ["No GPU", "AWS GPU quota requests (EC2 and SageMaker) stayed at 0 until the end", "everything designed for CPU: small model, uncertain pairs only",
    "plan for the compute you have, not the compute you asked for"],
   ["Cloud session lost", "a parallel cloud compute session died when the local network dropped", "work continued locally",
    "keep the source of truth local and checkpoint everything"],
   ["Deadline misread", "assumed 01:00 on the 28th", "corrected to 23:59 IST on the 27th", "re-read the rules before the final day"]],
  [3.4, 4.3, 4.4, 4.4])

H1("22. Diagnostics that drove the decisions")
H2("House-number change type → probability that the pair is a true match (US training pairs)")
T([["Change", "Example", "P(true match)", "Reading"],
   ["first digit dropped", "1941 → 941", "0.986", "noise on a true record"],
   ["first digit substituted", "3365 → 6365", "0.979", "noise"],
   ["last digit dropped", "330 → 33", "0.876", "mostly noise"],
   ["last digit ±1-2", "3460 → 3462", "0.431", "often a twin"],
   ["last digit changed by more than 2", "105 → 108", "0.328", "often a twin"],
   ["different number within 20", "1709 → 1720", "0.463", "often a twin"]], [4.6, 3.2, 2.8, 5.9])
H2("Train vs test: why the leaderboard is lower than validation")
T([["Measure", "Train", "Test"],
   ["S2/S3 records per S1", "4.68", "5.75"],
   ["Share of records matching nothing", "26%", "~40% (estimated)"],
   ["Records with best p1 in 0.02-0.5 (US / India)", "3.6% / 4.4%", "5.7% / 8.0%"],
   ["Twin-like attachments (same name, different number), US", "10.9%", "13.5%"],
   ["Positive rate among best attachments (US / India), EM", "0.951 / 0.931", "0.914 / 0.873"]], [9, 3.5, 4])
H2("Final test profile (v10c, label-free)")
T([["Country", "S1 entities", "Candidates / S1", "Matches / S1", "S1 with no match"],
   ["France (unseen)", "259,452", "3.91", "3.31", "5.6%"],
   ["India", "809,986", "3.85", "3.31", "5.9%"],
   ["US", "663,106", "3.72", "3.37", "5.8%"]], [3.6, 3, 3.3, 3.3, 3.3])
P("Training ground truth has 3.46 matches per S1 and 5.6% singletons. France behaves like the training countries, with no sign of "
  "over- or under-merging.", "small")
H2("Things checked and ruled out")
B(["ID leakage: IDs of matched records are uncorrelated with their S1 (correlation ≈ 0). Row order: uncorrelated.",
   "Brand aliases as a hidden key: only 27% of alias-bearing records share the alias with another record of the same entity.",
   "Test S1 built by removing one twin of each pair: twin pairs inside S1 are rare in both train (1.7%) and test (2.7%).",
   "Record-level noise differences between train and test: none (empty-address and script rates are similar)."])

H1("23. Where the remaining error lives, and what would close the gap")
T([["Cause (held-out, v10b pipeline)", "Macro F0.5 lost", "Explanation"],
   ["Blocking misses", "0.0076", "the true S1 never became a candidate: gibberish or DBA names, divergent transliterations (indiyan entarpraijes), "
    "empty or truncated addresses"],
   ["Empty-address records (overlaps with the row above)", "0.0065", "97.7% of them are true matches, but with only a name the S1 is often "
    "ambiguous: 49% of S1 entities share their exact core name with another S1 in the same country"],
   ["Rejected look-alikes", "0.0030", "true matches whose number was perturbed like a twin's ('52' vs '54'); indistinguishable pair by pair"],
   ["False merges", "0.0019", "precision is already about 99.7%"]], [5, 2.3, 9.2])
P("Validation F0.5 is 0.9864 but the leaderboard is 0.9804: the test set is harder in the same places (more twins, an unseen country).")
H2("What would close the gap to 0.99 (it needs a GPU)")
T([["Lever", "Targets", "Expected effect"],
   ["Dense retrieval: fine-tuned text embeddings + nearest-neighbour search as a second blocking channel", "blocking misses (0.0076)",
    "recall ceiling ≥ 99.5%, especially for transliterated and empty-address records"],
   ["Larger cross-encoder (e.g. DeBERTa-v3, MIT) on all 887k uncertain pairs, several epochs", "rejected look-alikes and false merges",
    "the largest matching gain; our CPU-only models saw at most 220k pairs for 1 epoch"],
   ["Cross-encoder over the whole group (S1 + all attached records in one sequence)", "twins vs typos", "learns consensus directly from text"],
   ["Training on test-like conditions (more distractors)", "train → test gap", "better-calibrated decisions on test"]], [6.8, 4.2, 5.5])

# ====================================================================================== PART IV
PART("IV", "Engineering and delivery", "How 26 million records were processed on a 16 GB laptop, the cloud setup, and the final package.")

H1("24. Engineering on a 16 GB laptop")
T([["Technique", "What it means", "Where we used it"],
   ["Columnar data frames (polars)", "vectorised operations over whole columns in Rust, multi-threaded", "all normalisation, joins, feature assembly"],
   ["Parquet caches", "compressed columnar files; each stage writes its artefact once and is skipped when it exists", "work/ (every stage's artefacts)"],
   ["Integer record IDs (rid)", "string IDs mapped to 32-bit integers once", "every join after prep"],
   ["Hashed keys (u64)", "keys stored as 64-bit hashes, not strings", "blocking index of 48 M key rows"],
   ["Chunking by record ID range", "process 150k records at a time, write a part file, free memory", "features, scoring, expansion"],
   ["Streaming + resumable jobs", "existing part files are skipped on restart; failures leave a marker in the log", "ext_score.py (106 M pairs)"],
   ["One heavy job at a time", "never run two memory-heavy stages together", "peak RAM ~9 GB"],
   ["Background jobs with logs", "long jobs run detached (nohup) and write progress lines", "10-12 h cross-encoder stage"]], [4, 7, 5.5])
T([["Runtime (i7-1355U, 12 threads)", "Time"],
   ["Normalisation", "~3.5 min"], ["Blocking", "~38 min per split"],
   ["Stage-1 training (features, 2 cross-fitted models, out-of-fold scoring of 98 M pairs, stage 2)", "~97 min"],
   ["Test scoring with the 2-model ensemble", "~107 min"], ["Cross-encoder stage (4 models trained + scored)", "~10-12 h"],
   ["Expansion v10b / v10c (blocking + re-scoring + cross-encoders)", "several hours each"]], [12.5, 4])

H1("25. Infrastructure: AWS, cost protection, GPU requests")
T([["Item", "State"],
   ["Account", "AWS with $100 credits, profile amazon-ml, region us-east-1"],
   ["Budget", "amazon-ml-90: $90, email alerts at 50 / 80 / 100% actual and 100% forecast"],
   ["Kill switch", "at $81 actual: SNS → Lambda stops every running EC2 instance (least-privilege role); self-test passed"],
   ["Anomaly detection", "AWS services monitor with a daily email for anomalies ≥ $5"],
   ["GPU quota", "EC2 G/VT instances and SageMaker ml.g5.2xlarge requested; both cases still open with quota 0 at the deadline"],
   ["Transfer bucket", "private S3 bucket with 7-day expiry; a scoped IAM user for long-lived download links"],
   ["Clean-up after the challenge", "delete the scoped IAM user, its access key and the transfer bucket"],
   ["Standing rule", "no instance is launched or resized without stating $/hour and $/day and getting explicit approval"]], [3.6, 12.9])
P("In the end no paid compute was used: the whole solution runs on the laptop CPU.", "small")

H1("26. The submission package and the compliance checks")
CODE("Vyrox_submission.zip  (87.8 MB, 28 files)\n"
     "├── output/\n"
     "│   ├── matching_results.tsv        5,772,673 matches, 1,732,544 rows (= the v10c leaderboard file)\n"
     "│   └── candidate_pairs.tsv         6,595,983 candidates = 3.81 per S1\n"
     "├── code/business_entity_resolution/\n"
     "│   ├── src/                        23 scripts (pipeline, cross-encoders, expansion)\n"
     "│   ├── README.md                   exact commands, data → blocking → matching → output\n"
     "│   └── requirements.txt            pinned versions (polars 1.44.2, lightgbm 4.7.0, torch 2.14.0, transformers 5.16.1 ...)\n"
     "└── Documentation_template.md       the filled-in methodology write-up")
T([["Check", "Result"],
   ["Organisers' validator with --check-ids on the files extracted from the zip", "PASS"],
   ["Every S1 present exactly once, same order in both files", "yes (1,732,544)"],
   ["Only existing S2/S3 IDs; no duplicates in any list", "yes"],
   ["Every match is also a candidate; no record assigned to two S1", "yes"],
   ["Outputs byte-identical to the v10c leaderboard upload", "yes"],
   ["Model licences and size", "LightGBM (MIT), MiniLM cross-encoder (Apache-2.0, 22.7 M parameters)"],
   ["No external data or APIs", "yes; only the base model weights are downloaded from Hugging Face"],
   ["Reproducible from the folder alone", "README + run_pipeline.py + run_ce_stage.sh (path-independent scripts)"]], [9, 7.5])
H2("Why our candidate set cannot shrink much further")
T([["Pruning threshold", "Candidates / S1", "Accepted matches deleted"],
   ["p1 ≥ 0.02 (submitted)", "<b>3.81</b>", "0"],
   ["p1 ≥ 0.05", "3.67", "1,100"], ["p1 ≥ 0.1", "3.59", "3,006"], ["p1 ≥ 0.2", "3.51", "7,798"], ["p1 ≥ 0.5", "3.40", "30,074"]], [5.5, 5, 6])
P("Training data has 3.46 true links per S1, so the theoretical floor is about 3.46 candidates per S1; ours is about 10% above it, and "
  "87.5% of our candidates end up as final matches. A tighter cut would delete real matches and lower F0.5.")

H1("27. Code map and how to reproduce")
T([["File (src/)", "Role"],
   ["run_pipeline.py", "part 1 entry point: --stage prep | block | train | predict"],
   ["config.py · normalize.py · prep.py", "paths and constants · transliteration and canonicalisation · normalised-record and answer-key caches"],
   ["blocking.py", "hashed multi-key blocking and candidate ranking"],
   ["features.py", "72 pair features incl. OCR repair and digit-position logic"],
   ["model.py", "LightGBM, argmax assignment, macro F0.5, expected-F rule, EM calibration"],
   ["stage2.py", "group / twin veto model"],
   ["ce_export.py · ce_prep_eval.py · ce_uncertain_features.py", "export uncertain pairs with raw text and features"],
   ["ce_train.py · ce_score.py", "fine-tune and score the cross-encoders"],
   ["ce_stack_train.py · ce_predict.py", "stacker3 training · v10a final prediction"],
   ["ext_expand_block.py · ext_score.py · ext_phase1.py · ext_phase2.sh · ext_phase3.py", "v10b expansion: deeper blocking, re-scoring, cross-encoders, evaluation and build"],
   ["ext_expand_block_in50.py · ext_merge50.py", "v10c: India ranks 31-50 and merge"],
   ["run_ce_stage.sh", "part 2: runs the cross-encoder stage and both expansions → final v10c outputs"]], [7, 9.5])
CODE("pip install -r requirements.txt\n"
     "cd code/business_entity_resolution\n"
     "python src/run_pipeline.py --data-dir <student_resource/dataset>     # part 1, ~5 h on a 12-thread laptop CPU\n"
     "sh src/run_ce_stage.sh                                               # part 2, cross-encoders + expansions → output/\n"
     "python utils/validate_submission.py --matching output/matching_results.tsv \\\n"
     "       --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids")

# ====================================================================================== PART V
PART("V", "Terminology", "Every term used in this project: what it means in general, and what it means here.")

H1("28. Glossary: every term, in depth")
gloss = [
    ("The problem", [
        ("Entity resolution (record linkage)", "Deciding which records, possibly from different sources, describe the same real-world thing (here: a business). "
         "Also called deduplication when done inside one source."),
        ("Entity / record", "An <i>entity</i> is the real business; a <i>record</i> is one row describing it. One entity can have many records."),
        ("Source (S1, S2, S3)", "A data feed. S1 is the deduplicated reference list; S2 and S3 are messy feeds with repeats and distractors."),
        ("Match / link", "An (S1, S2/S3) pair that describes the same business. The training answer key lists 7,638,365 of them."),
        ("Ground truth / answer key", "The correct links for training S1 entities (train_ground_truth.tsv). Test has no answer key."),
        ("Singleton", "An S1 entity with no matching S2/S3 records (5.6% in training). Predicting an empty list for it scores 1.0."),
        ("Distractor (decoy)", "An S2/S3 record that matches no S1 entity. About 26% in training and an estimated 40% in test."),
        ("Twin (look-alike)", "A different business with nearly the same name and a nearby address (84 vs 833 Gostlin Street). The main source of false merges."),
        ("Noise", "Random changes applied to records: typos, abbreviations, re-ordering, scripts, missing parts."),
        ("Transliteration", "Writing a word from one script in another script's letters (Telugu 'White Industries' → 'vait imdstris')."),
        ("DBA", "'Doing business as': a trade name different from the legal name. Hard to match by name."),
        ("Legal form", "The company-type suffix: Inc, LLC, Corp, Pvt Ltd, LLP, SARL, SASU. Differences can be noise or a real distinction."),
    ]),
    ("Metric and evaluation", [
        ("True / false positive (TP / FP)", "A predicted link that is correct / wrong. An FP is a wrong merge."),
        ("False negative (FN)", "A true link we did not predict: a missed match."),
        ("Precision", "TP / (TP + FP): of what we predicted, how much is right."),
        ("Recall", "TP / (TP + FN): of what is true, how much we found."),
        ("F-beta, F0.5", "A weighted harmonic mean of precision and recall. β = 0.5 weights precision more: an FP costs 4× an FN in the denominator."),
        ("Macro average", "Compute the score per S1 entity, then average, so every entity counts equally (as opposed to pooling all pairs = micro average)."),
        ("Held-out validation", "Scoring the pipeline on training entities that the models never saw, to estimate test performance. Ours: 441,785 S1 entities."),
        ("Public / private leaderboard", "The live leaderboard scores a subset of test (public); the final ranking may use the rest (private)."),
        ("Probe", "A leaderboard upload made to measure the effect of one change on the test distribution."),
        ("SCORED status / rejection", "An upload first passes a format check; if it fails (missing S1, unknown IDs, duplicates) it is rejected and gets no score."),
        ("Public subset", "The hidden part of the test S1 entities used for the live leaderboard. The final (private) evaluation can differ slightly."),
        ("Standard error, sampling noise", "How much a score would vary if a different random subset of entities were used: std / √N. "
         "Differences smaller than a few standard errors may be noise."),
        ("Paired difference", "Comparing two submissions on the same entities: only the entities where they differ create noise, so small "
         "differences between versions are measurable even when the absolute score is noisier."),
        ("Entity-equivalent", "A loss of 1.0 on one entity (e.g. one wrong merge into a singleton). +0.001 on 1.73 M entities ≈ 1,733 of them."),
        ("Label-free profiling", "Checking statistics of a test output that need no labels (number-disagreement rate, matches per S1) to catch problems before uploading."),
        ("AUC", "Area under the ROC curve: the probability that a random true pair gets a higher score than a random false pair. 0.5 = random, 1.0 = perfect."),
        ("Log-loss", "The average of −log(probability given to the correct answer). Rewards confident correct predictions, punishes confident wrong ones."),
    ]),
    ("Candidate generation", [
        ("Blocking", "Cheap candidate generation: only records that share a key are compared, which avoids the 17-trillion-pair cross product."),
        ("Key / key family", "A short string derived from a record (e.g. two rare name tokens). Seven families A-G, each catching a different kind of noise."),
        ("Inverted index", "A map from each key to the list of records containing it, like the index at the back of a book."),
        ("Hashing", "Turning a string into a fixed-size integer (64-bit here). Fast to compare and store; collisions are negligible at this scale."),
        ("Block / block size / block cap", "A block is the set of S1 records sharing a key. Blocks larger than 200 are dropped as too generic."),
        ("Document frequency, IDF", "How many records contain a token; inverse document frequency (IDF) ~ log(N / df) makes rare tokens count more."),
        ("Rarest-token selection", "Each record builds keys only from its 4 rarest name tokens and 5 rarest address tokens."),
        ("Block score", "Σ over shared keys of 1/√(block size): an IDF-like strength of the link between a record and an S1."),
        ("Top-k, recall@k", "Keep the k best-scoring candidates; recall@k is the share of true links whose S1 is among them."),
        ("Blocking ceiling (oracle F)", "The score a perfect matcher would reach on our candidates (0.9926): the limit set by blocking."),
        ("Candidate set", "The pairs the final model scores; ours is candidate_pairs.tsv, 3.81 per S1."),
        ("Learned pruning", "Using the stage-1 model to shrink the candidate set: best S1 per record with p1 ≥ 0.02."),
        ("Candidate expansion", "Re-running blocking deeper (top-30 / top-50) only for weak records (best p1 < 0.5), to recover blocking misses."),
        ("Dense retrieval", "Blocking by nearest neighbours in a learned embedding space (not used; needs a GPU)."),
    ]),
    ("Text processing and features", [
        ("Normalisation / canonicalisation", "Rewriting text into one standard form (lower case, no accents, 'road' → 'rd', 'pvt' → 'private')."),
        ("anyascii", "A library that maps any Unicode character to ASCII letters by table lookup (deterministic transliteration, ISC licence)."),
        ("Token", "A word-like unit after splitting text on spaces and punctuation. (For transformers: a sub-word piece from the tokenizer.)"),
        ("Stop words", "Very common words with little identity value (the, and, of, de, du, la) removed from core names."),
        ("Core name / concatenated name", "The name without legal and stop words / the same tokens glued together (catches domains like eelegal.com)."),
        ("Phonetic skeleton", "A sound-based code: aspirates merged, vowels after the first letter dropped, repeats collapsed. bharat green and bharat grin → brtgrn."),
        ("OCR repair", "Replacing digits inside words with look-alike letters (0→o, 1→l, 8→b): m0unt 8aptist → mount baptist."),
        ("Levenshtein / edit distance", "The minimum number of single-character insertions, deletions or substitutions to turn one string into another."),
        ("rapidfuzz ratio", "Similarity 0-100 derived from the edit distance of two whole strings."),
        ("Token-sort / token-set ratio", "Ratios computed after sorting the tokens / after comparing the shared and unshared token sets: robust to re-ordering and extra words."),
        ("Partial ratio", "Best ratio of the shorter string against any same-length window of the longer one: catches truncations."),
        ("Jaro-Winkler", "A string similarity that rewards matching characters in order and gives a bonus to a shared prefix; good for short names."),
        ("TF-IDF cosine", "Each text becomes a vector of token weights (term frequency × IDF); cosine similarity measures the angle between vectors (1 = same tokens)."),
        ("Jaccard similarity", "|A ∩ B| / |A ∪ B| for two sets (of tokens or numbers)."),
        ("Hamming distance", "Number of positions where two equal-length digit strings differ."),
        ("Competition features", "How a candidate compares with the record's other candidates: gap to the best, margin over the second best, number of candidates."),
        ("Feature importance (gain)", "How much a feature reduced the training loss across all tree splits that used it."),
    ]),
    ("Models and training", [
        ("Binary classifier", "A model that outputs the probability that an input belongs to the positive class (here: 'same business')."),
        ("Decision tree", "A flowchart of if-then splits on feature values ending in leaves that hold predictions."),
        ("Gradient boosting", "Building an ensemble of trees one after another, each fitted to the errors (gradients of the loss) of the ensemble so far."),
        ("LightGBM", "A fast gradient-boosting library (MIT) that grows trees leaf-wise using histograms of feature values."),
        ("Leaves, learning rate, early stopping", "Leaves = max terminal nodes per tree (127 in stage 1). Learning rate = how much each tree contributes (0.05). "
         "Early stopping = stop adding trees when the validation loss stops improving."),
        ("Overfitting", "Learning patterns specific to the training data that do not generalise. Honest validation detects it."),
        ("Fold, cross-validation", "Splitting data into parts (folds); training on some and evaluating on the rest."),
        ("Cross-fitting / out-of-fold (OOF)", "Every training item is scored by a model that never trained on it. Gives honest scores to train later stages on."),
        ("Data leakage", "When information about the answer sneaks into training or features, giving scores that are too optimistic."),
        ("Ensemble", "Averaging or combining several models; errors partly cancel. Test p1 = mean of the two fold models."),
        ("Stacking / stacker", "A second-level model trained on the outputs of first-level models (plus features). Ours: stacker3 with 76 inputs."),
        ("Logit, odds", "odds = p / (1 − p); logit = ln(odds). Logits turn probabilities into an unbounded scale that is additive for evidence."),
        ("Argmax assignment", "Attaching each S2/S3 record only to its single highest-probability S1."),
        ("Stage 1 / stage 2", "Stage 1 scores each pair on its own (p1). Stage 2 scores each attachment in the context of the whole S1 group (p2)."),
        ("Veto / soft veto", "Veto: final = min(score, p2), the group model can only lower a score. Soft veto: it lowers only when confident (p2 < 0.1)."),
    ]),
    ("Transformers and cross-encoders", [
        ("Neural network", "A function made of layers of weighted sums and non-linearities, whose weights are learned by gradient descent."),
        ("Transformer", "A neural architecture built from self-attention layers; the basis of modern language models."),
        ("Self-attention", "Each token computes a weighted mix of all other tokens, so it can relate distant parts of the text (e.g. two house numbers)."),
        ("Tokenizer", "Splits text into sub-word pieces from a fixed vocabulary; unknown words become several pieces."),
        ("[CLS] / [SEP]", "Special tokens: [CLS] starts the sequence and its final vector feeds the output score; [SEP] separates the two records."),
        ("Pre-training / fine-tuning", "Pre-training teaches a model general language on huge data; fine-tuning adapts it to our task with our labelled pairs."),
        ("Cross-encoder", "A transformer that reads both texts together and outputs one relevance / match score. Accurate but costs one pass per pair."),
        ("Bi-encoder", "Encodes each text separately into a vector; compares vectors. Fast for search (dense retrieval), weaker for fine decisions."),
        ("MiniLM-L6 (ms-marco)", "A small distilled transformer (6 layers, 22.7 M parameters, Apache-2.0) pre-trained for query-passage relevance on MS MARCO."),
        ("Epoch, batch", "Epoch = one pass over the training data. Batch = the number of examples processed per weight update (32)."),
        ("AdamW, weight decay", "A popular optimiser with adaptive step sizes; weight decay gently pulls weights towards zero to reduce overfitting."),
        ("One-cycle schedule, warm-up", "The learning rate rises for the first 6% of steps (warm-up), then decays: stable and fast training."),
        ("Binary cross-entropy", "The log-loss used to train a match / no-match model."),
        ("Uncertain pairs", "Best attachments with 0.02 ≤ p1 ≤ 0.98: where all remaining errors live, and the only pairs sent to the cross-encoders."),
        ("GPU vs CPU", "A GPU runs thousands of operations in parallel, making transformers ~20× faster to train; we had only a CPU."),
    ]),
    ("Calibration and decisions", [
        ("Calibration", "Probabilities are calibrated when a score of 0.8 is right about 80% of the time."),
        ("Label shift (prior shift)", "The share of positives differs between train and test while each class looks the same. Test has more look-alikes."),
        ("Prior (π)", "The base rate of positives, here among each record's best attachments (US train 0.951, test 0.914)."),
        ("EM algorithm (Saerens et al.)", "Expectation-maximisation: alternately re-weights predictions with a guessed prior and re-estimates the prior, until stable. Needs no labels."),
        ("Odds factor", "The multiplier applied to every score's odds after EM: [π_test/(1−π_test)] / [π_train/(1−π_train)] (US ×0.55)."),
        ("Threshold", "A fixed cut-off: predict 'match' when p > t. We replaced it with the expected-F rule."),
        ("Expected-F0.5 decision", "Per S1, keep the top-m records that maximise the estimated expected F0.5; m = 0 is allowed."),
        ("Plug-in estimate", "Replacing unknown quantities (true positives) by their expected values (sums of probabilities) inside a formula."),
    ]),
    ("Engineering", [
        ("TSV", "Tab-separated values: a text table with tabs between columns."),
        ("polars", "A fast data-frame library written in Rust, with lazy and streaming execution."),
        ("Parquet", "A compressed, columnar file format: fast to read only the columns you need."),
        ("Streaming / chunking", "Processing data in pieces that fit in memory instead of loading everything at once."),
        ("Resumable job", "A job that writes finished parts to disk and skips them on restart."),
        ("Hash seed / hash split", "Deterministic splitting of IDs into groups (folds, held-out set) by hashing them with a fixed seed (13)."),
        ("Validator", "The organisers' script that checks the output format and IDs (validate_submission.py --check-ids)."),
        ("Pinned dependencies", "Exact library versions in requirements.txt, so the environment can be recreated."),
        ("IAM, S3, SNS, Lambda", "AWS identity and permissions, object storage, notifications, and serverless functions (used for the budget kill switch)."),
    ]),
]
for section, items in gloss:
    H2(section)
    T([["Term", "Meaning"]] + [[f"<b>{a}</b>", b] for a, b in items], [4.4, 12.1])

doc = SimpleDocTemplate(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "Entity_Resolution_Project_Report.pdf"), pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Business Entity Resolution - Project Report", author="Team Vyrox")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("written")
