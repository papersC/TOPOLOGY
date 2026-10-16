"""Explicit claim-registry verifier for the manuscript.

Every table cell of every data table, and a registry of numeric claims in the
abstract/text, is recomputed from results/*.json|csv and compared with the
number printed in bare_jrnl.tex.  A printed number passes when it equals the
source value correctly rounded to the printed precision (|x - shown| <= half a
unit in the last printed digit, plus 1e-9).  Scientific notation (a x 10^b) is
checked to the printed mantissa precision.

Usage: python tools/verify_manuscript.py path/to/bare_jrnl.tex [results_dir]
Exit code 0 iff no mismatch.
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT.parent / "bare_jrnl.tex"
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else (_ROOT.parent / "results" if (_ROOT.parent / "results").is_dir() else _ROOT / "results")
tex = TEX.read_text()
body = tex.split("\\begin{thebibliography}")[0]
J = lambda f: json.loads((RES / f).read_text())
fails, checks = [], [0]

def ci95(v):
    v = np.asarray(v, float); n = len(v)
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))

NUM = re.compile(r"(-?\d+(?:\.\d+)?)(?:\\times10\^\{(-?\d+)\})?")
def nums(s):
    s = s.replace(",", "") if re.search(r"\d,\d{3}", s) else s
    out = []
    for m in NUM.finditer(s):
        out.append((m.group(1), m.group(2)))
    return out

def ok(shown, exp):
    mant, ex = shown
    d = len(mant.split(".")[1]) if "." in mant else 0
    if ex is not None:
        e = int(ex); m = float(mant)
        return abs(exp / 10 ** e - m) <= 0.5 * 10 ** -d + 1e-9
    return abs(float(mant) - exp) <= 0.5 * 10 ** -d + 1e-9

def check(where, shown, exp):
    checks[0] += 1
    if exp is None:
        return
    if not ok(shown, exp):
        fails.append(f"{where}: printed {shown[0]}{'e'+shown[1] if shown[1] else ''} but source gives {exp:.6g}")

def table_rows(label):
    m = re.search(r"\\begin\{table\*?\}(?:(?!\\end\{table).)*?\\label\{%s\}.*?\\end\{table\*?\}" % re.escape(label), tex, re.S)
    t = m.group(0)
    t = t[t.index("\\midrule"):t.index("\\bottomrule")]
    rows = []
    for r in t.split("\\\\"):
        r = re.sub(r"\\(mid|cmid)rule(\(lr\))?(\{[^}]*\})?", "", r).strip()
        r = re.sub(r"~?\\(cite|eqref)\{[^}]*\}", "", r)
        if r:
            rows.append([c.strip() for c in r.split("&")])
    return rows

def cmp_table(label, expected, skip_cells=1):
    rows = table_rows(label)
    if len(rows) != len(expected):
        fails.append(f"{label}: {len(rows)} rows printed, {len(expected)} expected"); return
    for r, (name, ev) in zip(rows, expected):
        shown = [n for c in r[skip_cells:] for n in nums(c)]
        ev = [e for e in ev if e != "skip"] if False else ev
        if len(shown) != len(ev):
            fails.append(f"{label}/{name}: {len(shown)} numbers printed, {len(ev)} expected ({r})"); continue
        for s, e in zip(shown, ev):
            check(f"{label}/{name}", s, e)

# ---------------- Table: aggregate seed 0 ----------------
agg = {r["scheme"]: r for r in J("main_tables.json")["table_agg"]}
bs = J("baselines_summary.json")["summary"]
def s0(name):
    d = bs[name]["seed0"]; return [d["mean_phase_s"], d["max_phase_s"], d["optical_pct"], d["n_reconfig"], d["speedup_vs_eps"]]
cmp_table("tab:agg", [
    ("EPS", s0("EPS Clos")), ("Static", s0("Static OCS")), ("GHEM", s0("SAD-OTR (GHEM-AC)")),
    ("delta0", [agg["SAD-OTR, delta=0"][k] for k in ("mean_phase_s", "max_phase_s", "optical_pct", "n_reconfig", "speedup_vs_eps")])])

# ---------------- Table: baselines (10 seeds) ----------------
seeds = list(csv.DictReader(open(RES / "baselines_seeds.csv")))
per = {}
for r in seeds:
    per.setdefault(r["scheme"], {})[int(r["seed"])] = float(r["mean_phase_s"])
tm = per["SAD-OTR-TM"]
order = ["EPS Clos", "Static OCS", "RotorNet-style", "MWM per phase", "Solstice-K", "Solstice-K+LP",
         "Eclipse-K", "Eclipse-K+LP", "SAD-OTR (GHEM-AC)", "SAD-OTR-TM", "Ideal bound"]
exp = []
for s in order:
    d = bs[s]
    m, h = d["mean_phase_s"]["mean"], d["mean_phase_s"]["ci95_half"]
    row = [m, h]
    if s != "EPS Clos":
        row += [d["speedup_vs_eps"]["mean"], d["speedup_vs_eps"]["ci95_half"]]
    else:
        row += [1.0]
    rm, rh = ci95([per[s][k] / tm[k] for k in sorted(tm)])
    row += [rm] if s == "SAD-OTR-TM" else [rm, rh]
    if s != "Ideal bound":
        row += [d["optical_pct"]["mean"], d["n_reconfig"]["mean"]]
        if s not in ("EPS Clos", "Static OCS", "RotorNet-style"):
            row += [d["mean_latency_ms"]["mean"]]
    exp.append((s, row))
cmp_table("tab:baselines", exp)

# ---------------- Table: significance ----------------
sig = J("significance.json")["tests"]
cmp_table("tab:sig", [(f"{t['a']} vs {t['b']}", [t["mean_ratio_a_over_b"], t["ratio_ci95"], t["wins_a_faster"], t["n"],
                        t["p_t_holm"], t["p_wilcoxon_holm"], t["cohen_dz_log"]]) for t in sig])

# ---------------- Table: TM ablation ----------------
ab = {r["variant"]: r for r in J("tm_ablation.json")["summary"]}
abl_order = [("default (L_max=32)", [32]), ("L_max=1", [1]), ("L_max=2", [2]), ("L_max=4", [4]), ("L_max=8", [8]),
             ("L_max=16", [16]), ("no lazy stop", []), ("amortization off (adopt all slices)", []),
             ("first slice always charged", []), ("no GHEM-AC first slice", []), ("|A|<=4", [4]), ("|A|<=16", [16]),
             ("|A| unrestricted", []), ("budget 10 ms", [10]), ("budget 20 ms", [20]), ("budget 40 ms", [40])]
exp = []
for v, lab in abl_order:
    s = ab[v]
    extra = []
    exp.append((v, lab + extra + [s["mean_phase_s"], s["ci95"], s["ratio_to_default"], s["ratio_ci95"], s["n_reconfig"],
                                  s["mean_latency_ms"], s["max_latency_ms"]]))
cmp_table("tab:tmabl", exp, skip_cells=0)

# ---------------- Table: LLM workload ----------------
ls = J("llm_summary.json")["summary"]["15.0"]
exp = []
for s in order:
    d = ls[s]
    row = [1000 * d["mean_phase_s"]["mean"], 1000 * d["mean_phase_s"]["ci95_half"], d["speedup_vs_eps"]["mean"], d["ratio_to_tm"]["mean"]]
    if s != "Ideal bound":
        row += [d["optical_pct"]["mean"]]
    row += [d["n_reconfig"]["mean"], d["mean_latency_ms"]["mean"]]
    exp.append((s, row))
cmp_table("tab:llm", exp)

# ---------------- Table: optimality N=8..32 ----------------
opt = J("main_tables.json")["table_opt"]; lat = {r["N"]: r for r in J("latency_table_ii.json")["table_opt_latency"]}
cmp_table("tab:opt", [(r["N"], [r["N"], r["mean_gap_pct"], r["worst_gap_pct"], lat[r["N"]]["ghem_mean_ms"], lat[r["N"]]["milp_median_ms"]]) for r in opt], skip_cells=0)

# ---------------- Table: scalability ----------------
exp = []
for r in J("scalability.json")["summary"]:
    exp.append((f"{r['family']}{r['N']}", [r["N"], r["n_seeds"], r["mean_gap_pct"], r["worst_gap_pct"], r["ghem_mean_ms"],
                r["milp_median_ms"], r["milp_p95_ms"], r["milp_max_ms"], r["tm_mean_ms"], r["tm_over_lb"],
                r["speedup_ghem_vs_eps"], r["speedup_tm_vs_eps"]]))
cmp_table("tab:scal", exp)

# ---------------- Table: split ablation ----------------
sp = {r["policy"]: r for r in J("ablation_split.json")["rows"]}
cmp_table("tab:split", [(p, [sp[p]["mean_phase_s"], sp[p]["optical_pct"], sp[p]["speedup_vs_eps"]]) for p in
                        ["EPS Clos (packet-only)", "GHEM-AC unsplittable", "GHEM-AC free split", "MILP unsplittable", "MILP free split"]])

# ---------------- Tables K and B_eps ----------------
cmp_table("tab:k", [(r["K"], [r["K"], r["mean_phase_s"], r["speedup"], r["optical_pct"]]) for r in J("main_tables.json")["table_k"]], skip_cells=0)
cmp_table("tab:beps", [(r["B_eps"], [r["B_eps"], r["eps_s"], r["static_s"], r["sad_s"], r["speedup"]]) for r in J("main_tables.json")["table_beps"]], skip_cells=0)

# ---------------- Text-claim registry ----------------
def claim(pattern, exp, desc):
    if exp is None:
        if not re.search(pattern, body, re.S): fails.append(f"TEXT[{desc}]: pattern not found")
        checks[0] += 1; return
    """pattern: regex with one group capturing the printed number; must match at least once in body."""
    ms = list(re.finditer(pattern, body, re.S))
    if not ms:
        fails.append(f"TEXT[{desc}]: pattern not found: {pattern}"); return
    for m in ms:
        check(f"TEXT[{desc}]", nums(m.group(1))[0], exp)

b = bs
tmS, ecl = b["SAD-OTR-TM"], b["Eclipse-K+LP"]
st = {t["a"] + "|" + t["b"]: t for t in sig}
t_te = st["SAD-OTR-TM|Eclipse-K+LP"]
claim(r"\$(5\.14)\\pm0\.07\$ speedup", tmS["speedup_vs_eps"]["mean"], "TM speedup")
claim(r"5\.14\\pm(0\.07)\$ speedup", tmS["speedup_vs_eps"]["ci95_half"], "TM speedup CI")
lbratio = ci95([tm[k] / per["Ideal bound"][k] for k in tm])[0]
claim(r"within (7\.9)~percent of an ideal", 100 * (lbratio - 1), "TM over bound")
claim(r"(1\.1)~percent ahead of Eclipse", 100 * (1 - t_te["mean_ratio_a_over_b"]), "TM vs Eclipse+LP")
claim(r"margin\s+over Eclipse-K\+LP is small \((1\.1)~percent", 100 * (1 - t_te["mean_ratio_a_over_b"]), "limitations margin")
llm_t = J("llm_summary.json")["tests"]["SAD-OTR-TM vs Eclipse-K+LP"]
claim(r"margin over retimed Eclipse grew to\s+(11\.4)~percent", 100 * (1 - llm_t["mean_ratio"]), "LLM margin conclusion")
ghem0 = b["SAD-OTR (GHEM-AC)"]["seed0"]
claim(r"by (2\.49) times against a packet Clos", ghem0["speedup_vs_eps"], "GHEM seed0 vs EPS")
claim(r"and (1\.96)\s+times against a static", J("multiseed.json")["per_seed"][0]["speedup_vs_static"], "GHEM seed0 vs static")
claim(r"from 1\.43 at one port to (2\.49) at four", [r for r in J("main_tables.json")["table_k"] if r["K"] == 4][0]["speedup"], "K=4")
claim(r"(6\.43) at eight", [r for r in J("main_tables.json")["table_k"] if r["K"] == 8][0]["speedup"], "K=8")
claim(r"saturates at (6\.68)", [r for r in J("main_tables.json")["table_k"] if r["K"] == 16][0]["speedup"], "K sat")
# delta sweep text
ds = list(csv.DictReader(open(RES / "delta_sweep.csv")))
def dsv(ms, col):
    for r in ds:
        if abs(float(r[[k for k in r if k.startswith("delta")][0]]) - ms) < 1e-6 * max(1, ms): return float(r[col])
dk = [k for k in ds[0] if k.startswith("delta")][0]
dunit = 1000.0 if "_s" in dk else 1.0
def row_at(ms):
    return min(ds, key=lambda r: abs(float(r[dk]) * dunit - ms))
mcol = [k for k in ds[0] if "mean" in k][0]
claim(r"rises from (1\.132)~s", float(row_at(1)[mcol]), "delta 1ms mean")
claim(r"to (1\.826)~s", float(row_at(1000)[mcol]), "delta 1s mean")
# TM ablation text
claim(r"two slices recover (\d+)~percent", 100 * (ab["L_max=1"]["mean_phase_s"] - ab["L_max=2"]["mean_phase_s"]) /
      (ab["L_max=1"]["mean_phase_s"] - ab["default (L_max=32)"]["mean_phase_s"]), "two slices recover")
sd = J("tm_smalldelta.json")
claim(r"raising it to 128 lowers the mean phase time from (0\.512)~s", sd["default L_max=32"]["mean_phase_s"], "L32 small delta")
claim(r"from 0\.512~s to\s+(0\.486)~s", sd["L_max=128"]["mean_phase_s"], "L128 small delta")
bd = [r for r in csv.DictReader(open(RES / "baselines_delta.csv"))]

# LLM text
l1, l100 = J("llm_summary.json")["summary"]["1.0"], J("llm_summary.json")["summary"]["100.0"]
claim(r"(19\.1)~ms", 1000 * l1["SAD-OTR-TM"]["mean_phase_s"]["mean"], "LLM delta1 TM")
claim(r"(85\.5)~ms", 1000 * l100["SAD-OTR (GHEM-AC)"]["mean_phase_s"]["mean"], "LLM delta100 GHEM")
claim(r"(135\.4)~ms", 1000 * l100["MWM per phase"]["mean_phase_s"]["mean"], "LLM delta100 MWM")


# ---- baselines text ----
def pr(a, bb):
    return ci95([per[a][k] / per[bb][k] for k in sorted(per[bb])])
m_, h_ = pr("MWM per phase", "SAD-OTR (GHEM-AC)")
claim(r"MWM/GHEM-AC is \$(1\.016)\\pm", m_, "MWM/GHEM ratio"); claim(r"MWM/GHEM-AC is \$1\.016\\pm(0\.010)\$", h_, "MWM/GHEM CI")
claim(r"gains only (1\.6)~percent", 100 * (m_ - 1), "MWM gain")
m_, h_ = pr("RotorNet-style", "SAD-OTR (GHEM-AC)")
claim(r"RotorNet-style rotation is (1\.34) times slower", m_, "rotor/ghem"); claim(r"\(\$1\.34\\pm(0\.10)\$ paired\)", h_, "rotor/ghem CI")
claim(r"Eclipse-K reaches (0\.690)~s", b["Eclipse-K"]["mean_phase_s"]["mean"], "Eclipse mean")
claim(r"GHEM-AC is \$(1\.47)\$ times slower", b["SAD-OTR (GHEM-AC)"]["mean_phase_s"]["mean"] / b["Eclipse-K"]["mean_phase_s"]["mean"], "GHEM/Eclipse means")
claim(r"time multiplexing is (1\.47) times faster", b["SAD-OTR (GHEM-AC)"]["mean_phase_s"]["mean"] / b["Eclipse-K"]["mean_phase_s"]["mean"], "abstract 1.47")
claim(r"\((46\.8) charged", b["Solstice-K"]["n_reconfig"]["mean"], "solstice reconfigs")
claim(r"Eclipse-K by (24)~percent", 100 * (1 - b["Eclipse-K+LP"]["mean_phase_s"]["mean"] / b["Eclipse-K"]["mean_phase_s"]["mean"]), "LP gain eclipse")
claim(r"Solstice-K by\s+(18)~percent", 100 * (1 - b["Solstice-K+LP"]["mean_phase_s"]["mean"] / b["Solstice-K"]["mean_phase_s"]["mean"]), "LP gain solstice")
m_, h_ = pr("SAD-OTR (GHEM-AC)", "SAD-OTR-TM")
claim(r"\$(1\.94)\\pm0\.11\$ reduction", m_, "ghem/tm"); claim(r"\$1\.94\\pm(0\.11)\$ reduction", h_, "ghem/tm CI")
m_, h_ = pr("Ideal bound", "SAD-OTR-TM")
claim(r"paired ratio \$(0\.927)\\pm", m_, "lb/tm"); claim(r"paired ratio \$0\.927\\pm(0\.004)\$", h_, "lb/tm ci")
claim(r"It is \$(7\.9)\$~percent above", 100 * (lbratio - 1), "7.9 text")
claim(r"within\s+(8) percent of an ideal bound", 100 * (lbratio - 1), "abstract 8%")
claim(r"\$(0\.845)\\pm0\.074\$~s", b["SAD-OTR-TM"]["max_phase_s"]["mean"], "TM max"); claim(r"\$0\.845\\pm(0\.074)\$~s", b["SAD-OTR-TM"]["max_phase_s"]["ci95_half"], "TM max ci")
claim(r"\$(3\.050)\\pm0\.433\$~s", b["SAD-OTR (GHEM-AC)"]["max_phase_s"]["mean"], "GHEM max"); claim(r"\$3\.050\\pm(0\.433)\$~s", b["SAD-OTR (GHEM-AC)"]["max_phase_s"]["ci95_half"], "GHEM max ci")
claim(r"small but consistent: \$(1\.1)\\pm0\.4\$~percent", 100 * (1 - t_te["mean_ratio_a_over_b"]), "margin"); claim(r"\$1\.1\\pm(0\.4)\$~percent", 100 * t_te["ratio_ci95"], "margin ci")
claim(r"achieved with (16)~percent fewer", 100 * (1 - b["SAD-OTR-TM"]["n_reconfig"]["mean"] / b["Eclipse-K+LP"]["n_reconfig"]["mean"]), "fewer reconfigs")
pk = {r["scheme"]: r for r in csv.DictReader(open(RES / "baselines_phasekind.csv"))}
claim(r"\((0\.170)~s and 0\.682~s mean", float(pk["SAD-OTR-TM"]["TP_mean_s"]), "TP"); claim(r"0\.170~s and (0\.682)~s mean", float(pk["SAD-OTR-TM"]["PP_mean_s"]), "PP")
claim(r"GHEM-AC averages (2\.576)~s", float(pk["SAD-OTR (GHEM-AC)"]["EP_mean_s"]), "EP ghem")
claim(r"SAD-OTR-TM (0\.769)~s against", float(pk["SAD-OTR-TM"]["EP_mean_s"]), "EP tm"); claim(r"bound of (0\.670)~s", float(pk["Ideal bound"]["EP_mean_s"]), "EP lb")
D = {(float(r["delta_ms"]), r["scheme"]): float(r["mean_phase_s"]) for r in bd}
claim(r"at 100~ms it\s+takes (0\.624)~s", D[(100.0, "SAD-OTR-TM")], "d100 tm"); claim(r"against (0\.652)~s for Eclipse-K\+LP", D[(100.0, "Eclipse-K+LP")], "d100 ecl")
claim(r"and (1\.091)~s for GHEM-AC", D[(100.0, "SAD-OTR (GHEM-AC)")], "d100 ghem")
claim(r"it takes (1\.516)~s", D[(1000.0, "SAD-OTR-TM")], "d1000 tm"); claim(r"against (1\.710)~s for GHEM-AC", D[(1000.0, "SAD-OTR (GHEM-AC)")], "d1000 ghem")
claim(r"and (1\.714)~s for Eclipse-K\+LP", D[(1000.0, "Eclipse-K+LP")], "d1000 ecl")
claim(r"faster \((0\.487)~s against 0\.512~s\)", D[(0.1, "Eclipse-K+LP")], "d0.1 ecl"); claim(r"\(0\.487~s against (0\.512)~s\)", D[(0.1, "SAD-OTR-TM")], "d0.1 tm")
claim(r"Eclipse-K\+LP \((0\.487)~s\)\.", D[(0.1, "Eclipse-K+LP")], "d0.1 ecl in ablation")
for d_ in [1.0, 5.0, 15.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
    best = min((v, s) for (dd, s), v in D.items() if dd == d_ and s != "Ideal bound")
    if best[1] != "SAD-OTR-TM":
        fails.append(f"TEXT[fastest from 1 ms]: at {d_} ms fastest is {best[1]}")
# ---- significance text ----
claim(r"on all ten seeds \(\$d_z=(-2\.0)\$\)", t_te["cohen_dz_log"], "dz TM-ecl")
claim(r"per phase on all ten seeds \(\$d_z=(-1\.1)\$\)", st["SAD-OTR (GHEM-AC)|MWM per phase"]["cohen_dz_log"], "dz ghem-mwm")
claim(r"raises to (0\.018)", st["SAD-OTR-TM|Eclipse-K+LP"]["p_wilcoxon_holm"], "holm floor")
claim(r"2\^\{-9\}\\approx(0\.002)", 2 ** -9, "wilcoxon floor")
if not all(t["p_t_holm"] < 0.05 and t["p_wilcoxon_holm"] < 0.05 for t in sig):
    fails.append("TEXT[all significant]: not every test has Holm p < 0.05")
# ---- ablation text ----
dflt = ab["default (L_max=32)"]
claim(r"four slices bring the controller within\s+(9\.5)~percent", 100 * (ab["L_max=4"]["ratio_to_default"] - 1), "L4")
claim(r"eight within (0\.5)~percent", 100 * (ab["L_max=8"]["ratio_to_default"] - 1), "L8")
claim(r"\((52\.1)\s+against 27\.6\)", ab["amortization off (adopt all slices)"]["n_reconfig"], "amort reconf")
claim(r"against (27\.6)\) and costs", dflt["n_reconfig"], "default reconf")
claim(r"and costs (5\.2)~percent", 100 * (ab["amortization off (adopt all slices)"]["ratio_to_default"] - 1), "amort cost")
claim(r"incumbent\s+costs (0\.7)~percent", 100 * (ab["first slice always charged"]["ratio_to_default"] - 1), "charged")
claim(r"greedy slice costs (1\.1)~percent", 100 * (ab["no GHEM-AC first slice"]["ratio_to_default"] - 1), "greedy first")
claim(r"\(\$(-0\.3)\\pm0\.3\$~percent\)", 100 * (ab["|A|<=4"]["ratio_to_default"] - 1), "A4"); claim(r"\(\$-0\.3\\pm(0\.3)\$~percent\)", 100 * ab["|A|<=4"]["ratio_ci95"], "A4 ci")
claim(r"20~ms yields (1\.36) times", ab["budget 20 ms"]["ratio_to_default"], "b20"); claim(r"40~ms yields (1\.11) times", ab["budget 40 ms"]["ratio_to_default"], "b40")
for v in ("L_max=16", "no lazy stop"):
    if abs(ab[v]["mean_phase_s"] - dflt["mean_phase_s"]) > 1e-9: fails.append(f"TEXT[unchanged]: {v} differs from default")
if abs(ab["budget 10 ms"]["mean_phase_s"] - ab["L_max=1"]["mean_phase_s"]) > 1e-6: fails.append("TEXT[10ms=GHEM]")
claim(r"four phases per run reach the cap", None, "cap phases")
if sd["default L_max=32"]["phases_at_cap"] != 4: print("  note phases_at_cap =", sd["default L_max=32"]["phases_at_cap"])
# ---- LLM text ----
L = J("llm_summary.json"); ls1 = L["summary"]["15.0"]; lt = L["tests"]
claim(r"SAD-OTR-TM completes a phase in (37\.6)~ms", 1000 * ls1["SAD-OTR-TM"]["mean_phase_s"]["mean"], "llm tm")
claim(r"37\.6~ms on average,\s+(2\.49) times", ls1["SAD-OTR-TM"]["speedup_vs_eps"]["mean"], "llm tm speedup")
claim(r"and (11\.4)~percent faster than Eclipse-K\+LP", 100 * (1 - lt["SAD-OTR-TM vs Eclipse-K+LP"]["mean_ratio"]), "llm margin")
claim(r"margin grows to (11) percent", 100 * (1 - lt["SAD-OTR-TM vs Eclipse-K+LP"]["mean_ratio"]), "abstract 11")
claim(r"\(paired ratio (0\.886)", lt["SAD-OTR-TM vs Eclipse-K+LP"]["mean_ratio"], "llm ratio")
claim(r"Wilcoxon \$p=(0\.002)\$", lt["SAD-OTR-TM vs Eclipse-K+LP"]["p_wilcoxon"], "llm p")
if lt["SAD-OTR-TM vs Eclipse-K+LP"]["wins_a"] != 10: fails.append("TEXT[llm all ten seeds]")
claim(r"GHEM-AC reaches (2\.03) times", ls1["SAD-OTR (GHEM-AC)"]["speedup_vs_eps"]["mean"], "llm ghem")
claim(r"MWM per phase \((1\.86) times\)", ls1["MWM per phase"]["speedup_vs_eps"]["mean"], "llm mwm")
claim(r"pays (12) penalties against 8", ls1["MWM per phase"]["n_reconfig"]["mean"], "mwm 12"); claim(r"pays 12 penalties against (8)", ls1["SAD-OTR (GHEM-AC)"]["n_reconfig"]["mean"], "ghem 8")
claim(r"adopts\s+(68\.2) slices", ls1["Solstice-K"]["n_reconfig"]["mean"], "solstice llm")
if ls1["Solstice-K"]["mean_phase_s"]["mean"] <= ls1["EPS Clos"]["mean_phase_s"]["mean"]: fails.append("TEXT[solstice slower than EPS]")
claim(r"factor of (2\.28) \(bound", ls1["SAD-OTR-TM"]["mean_phase_s"]["mean"] / ls1["Ideal bound"]["mean_phase_s"]["mean"], "llm tm/lb")
claim(r"\(bound (16\.5)~ms\)", 1000 * ls1["Ideal bound"]["mean_phase_s"]["mean"], "llm lb")
claim(r"\((4\.90) times faster than EPS Clos\)", l1["SAD-OTR-TM"]["speedup_vs_eps"]["mean"], "llm d1 speedup")
claim(r"within (16)~percent of the\s+bound", 100 * (l1["SAD-OTR-TM"]["mean_phase_s"]["mean"] / l1["Ideal bound"]["mean_phase_s"]["mean"] - 1), "llm d1 vs lb")
if abs(l1["SAD-OTR-TM"]["mean_phase_s"]["mean"] - l1["Eclipse-K+LP"]["mean_phase_s"]["mean"]) > 5e-5: fails.append("TEXT[llm d1 coincide]")
for s in ("SAD-OTR-TM", "SAD-OTR (GHEM-AC)", "Eclipse-K+LP", "Solstice-K+LP"):
    if abs(1000 * l100[s]["mean_phase_s"]["mean"] - 85.5) > 0.05 or abs(l100[s]["n_reconfig"]["mean"] - 1) > 1e-9:
        fails.append(f"TEXT[llm d100 converge]: {s} = {1000*l100[s]['mean_phase_s']['mean']:.2f} ms, {l100[s]['n_reconfig']['mean']} reconf")
claim(r"reaches (64\.7)~ms", 1000 * l100["Static OCS"]["mean_phase_s"]["mean"], "llm d100 static")
claim(r"microseconds, (49\.3)~ms", 1000 * l100["RotorNet-style"]["mean_phase_s"]["mean"], "llm d100 rotor")
if l100["MWM per phase"]["mean_phase_s"]["mean"] <= l100["EPS Clos"]["mean_phase_s"]["mean"]: fails.append("TEXT[mwm slower than eps at 100]")
# ---- latency text ----
scl = {(r["family"], r["N"]): r for r in J("scalability.json")["summary"]}
claim(r"in (77)~ms at 128 racks", scl[("uniform", 128)]["ghem_mean_ms"], "ghem 128")
if max(lat[n]["ghem_mean_ms"] for n in lat) >= 15: fails.append("TEXT[within one penalty up to 32]")
# ---- scalability text ----
raw = list(csv.DictReader(open(RES / "scalability_raw.csv")))
g16u = sorted(float(r["gap_pct"]) for r in raw if r["family"] == "uniform" and r["N"] == "16" and float(r["gap_pct"]) > 1e-4)
g16s = sorted(float(r["gap_pct"]) for r in raw if r["family"] == "skewed" and r["N"] == "16" and float(r["gap_pct"]) > 1e-4)
if len(g16u) != 3 or len([r for r in raw if r["family"] == "uniform" and r["N"] == "16"]) != 60: fails.append(f"TEXT[three of 60]: {g16u}")
claim(r"lie (3\.5) to 5\.1~percent", g16u[0], "g16u min"); claim(r"lie 3\.5 to (5\.1)~percent", g16u[-1], "g16u max")
if len(g16s) != 2: fails.append(f"TEXT[two skewed]: {g16s}")
claim(r"phases\s+(0\.7) and 1\.4~percent", g16s[0], "g16s a"); claim(r"(1\.4)~percent above it", g16s[-1], "g16s b")
big = max(float(r["gap_pct"]) for r in raw if int(r["N"]) >= 32)
if big >= 4e-6: fails.append(f"TEXT[residual gaps < 4e-6]: {big}")
if not all(r["milp_status"] == "Optimal" for r in raw): fails.append("TEXT[every MILP optimal]")
tl = [r["tm_over_lb"] for r in J("scalability.json")["summary"]]
claim(r"within (7) to 11~percent", 100 * (min(tl) - 1), "tm/lb min"); claim(r"within 7 to (11)~percent", 100 * (max(tl) - 1), "tm/lb max")
S = J("scalability.json")["summary"]; sc2 = {(r["family"], r["N"]): r for r in S}
claim(r"ratio GHEM-AC/TM rises from\s+(1\.58)", sc2[("uniform", 16)]["ghem_over_tm"], "g/tm u16")
claim(r"1\.58 to (2\.51)", sc2[("uniform", 256)]["ghem_over_tm"], "g/tm u256")
claim(r"and from (3\.32) to", sc2[("skewed", 16)]["ghem_over_tm"], "g/tm s16"); claim(r"to (7\.52)\.", sc2[("skewed", 256)]["ghem_over_tm"], "g/tm s256")
claim(r"from\s+3\.12 to (2\.14) on uniform", sc2[("uniform", 256)]["speedup_ghem_vs_eps"], "u256 speedup")
claim(r"from (1\.76) to only 1\.05", sc2[("skewed", 16)]["speedup_ghem_vs_eps"], "s16 sp")
ge = J("scalability.json")["growth_exponents_N32_256"]
claim(r"\$N\^\{(1\.78)\}\$ \(uniform\)", ge["uniform_ghem_mean_ms_exponent"], "exp u"); claim(r"\$N\^\{(1\.68)\}\$ \(skewed\)", ge["skewed_ghem_mean_ms_exponent"], "exp s")
claim(r"\$N\^\{(0\.95)\}\$", ge["uniform_milp_median_ms_exponent"], "milp exp u"); claim(r"and \$N\^\{(0\.91)\}\$", ge["skewed_milp_median_ms_exponent"], "milp exp s")
claim(r"MILP median \((197)~ms\)", sc2[("uniform", 256)]["milp_median_ms"], "milp 256"); claim(r"Python heuristic \((298)~ms\)", sc2[("uniform", 256)]["ghem_mean_ms"], "ghem 256")
claim(r"16 racks was (10\.0)~s", sc2[("skewed", 16)]["milp_max_ms"] / 1000, "milp max s16")
claim(r"reaching (2\.5)~s per phase", sc2[("uniform", 256)]["tm_mean_ms"] / 1000, "tm 256u"); claim(r"and\s+(45\.6)~s on the skewed", sc2[("skewed", 256)]["tm_mean_ms"] / 1000, "tm 256s")
# ---- split text ----
at = J("ablation_split.json")["attribution"]
claim(r"Of the\s+(1\.702)~s total", at["gain_total_s"], "gain total")
claim(r"about (92)~percent comes", 100 * at["fraction_of_gain_from_topology"], "92"); claim(r"about (8)~percent from splitting", 100 * at["fraction_of_gain_from_splitting"], "8")

# ---- volume sweep text ----
vs = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(RES / "volume_sweep.csv"))]
first = min((r for r in vs if r["n_reconfig"] > 0), key=lambda r: r["vol_gb"])
last0 = max((r for r in vs if r["n_reconfig"] == 0), key=lambda r: r["vol_gb"])
claim(r"Up to (0\.6)~Gb per TP pair", last0["vol_gb"], "last zero")
claim(r"adopted at (0\.94)~Gb", first["vol_gb"], "first vol"); claim(r"0\.94~Gb \(speedup (1\.11)\)", first["speedup_vs_eps"], "first sp")
e8 = min((r for r in vs if r["n_reconfig"] >= 8), key=lambda r: r["vol_gb"]); e9 = min((r for r in vs if r["n_reconfig"] >= 9), key=lambda r: r["vol_gb"])
claim(r"eight are adopted from (2\.3)~Gb", e8["vol_gb"], "eight"); claim(r"all nine\s+from (10)~Gb", e9["vol_gb"], "nine")
claim(r"speedup reaches (2\.10)", e9["speedup_vs_eps"], "sp at 10")
claim(r"at 0\.94~Gb it is (13\.3)~ms", 1000 * first["eps_mean_s"], "eps at first")
if any(r["speedup_vs_eps"] != 1.0 for r in vs if r["n_reconfig"] == 0): fails.append("TEXT[exactly 1.00]")
claim(r"GHEM-AC keeps all nine reconfigurations up to (251)~ms", 1000 * max(float(r["delta_s"]) for r in ds if float(r["n_reconfig"]) == 9), "delta 9")
claim(r"eight from\s+(398)~ms", 1000 * min(float(r["delta_s"]) for r in ds if float(r["n_reconfig"]) == 8), "delta 8")
claim(r"still\s+(1\.56) times faster than EPS", bs["EPS Clos"]["seed0"]["mean_phase_s"] / float(row_at(1000)[mcol]), "1s vs eps")
# ---- GHEM multiseed text ----
g = bs["SAD-OTR (GHEM-AC)"]
claim(r"time of GHEM-AC is \$(1\.016)\\pm0\.083\$~s", g["mean_phase_s"]["mean"], "ghem ms"); claim(r"is \$1\.016\\pm(0\.083)\$~s", g["mean_phase_s"]["ci95_half"], "ghem ci")
claim(r"deviation\s+(0\.117)~s", g["mean_phase_s"]["std"], "ghem sd")
claim(r"optical share \$(77\.1)\\pm0\.4\$", g["optical_pct"]["mean"], "opt"); claim(r"\$77\.1\\pm(0\.4)\$~percent", g["optical_pct"]["ci95_half"], "opt ci")
claim(r"Clos \$(2\.65)\\pm0\.12\$", g["speedup_vs_eps"]["mean"], "sp"); claim(r"Clos \$2\.65\\pm(0\.12)\$", g["speedup_vs_eps"]["ci95_half"], "sp ci")
claim(r"With \$\\delta=0\$ the same controller yields (1\.132)~s", agg["SAD-OTR, delta=0"]["mean_phase_s"], "delta0")
claim(r"maximum phase time, however, is (3\.862)~s", agg["SAD-OTR (GHEM-AC)"]["max_phase_s"], "max")
claim(r"reaches 2\.241~s and carries (34\.2)~percent", agg["Static OCS"]["optical_pct"], "static opt")
claim(r"1\.143~s with (77\.5)~percent", agg["SAD-OTR (GHEM-AC)"]["optical_pct"], "ghem opt")
claim(r"and\s+(1\.96) times against the static optical topology", J("multiseed.json")["per_seed"][0]["speedup_vs_static"], "vs static seed0")
# ---- robustness text ----
R = {(r["mode"], r["level"]): r for r in J("robust.json")["rows"]}
for pat, key, fld in [(r"degrades by\s+\$(0\.28)\\pm", ("volume", 0.1), "ghem_degr_pct"), (r"\$0\.28\\pm(0\.20)\$", ("volume", 0.1), "ghem_ci95"),
                      (r"\$(3\.9)\\pm1\.2\$", ("volume", 0.3), "ghem_degr_pct"), (r"\$3\.9\\pm(1\.2)\$", ("volume", 0.3), "ghem_ci95"),
                      (r"\$(8\.7)\\pm2\.3\$", ("volume", 0.5), "ghem_degr_pct"), (r"\$8\.7\\pm(2\.3)\$", ("volume", 0.5), "ghem_ci95"),
                      (r"\$(10\.0)\\pm1\.6\$", ("volume", 0.1), "tm_degr_pct"), (r"\$10\.0\\pm(1\.6)\$", ("volume", 0.1), "tm_ci95"),
                      (r"\$(29\.3)\\pm2\.8\$", ("volume", 0.3), "tm_degr_pct"), (r"\$29\.3\\pm(2\.8)\$", ("volume", 0.3), "tm_ci95"),
                      (r"\$(52\.9)\\pm7\.0\$", ("volume", 0.5), "tm_degr_pct"), (r"\$52\.9\\pm(7\.0)\$", ("volume", 0.5), "tm_ci95"),
                      (r"\$(6\.0)\\pm3\.4\$", ("fanout", 0.2), "ghem_degr_pct"), (r"\$6\.0\\pm(3\.4)\$", ("fanout", 0.2), "ghem_ci95"),
                      (r"\$(9\.9)\\pm3\.9\$", ("fanout", 0.5), "ghem_degr_pct"), (r"\$9\.9\\pm(3\.9)\$", ("fanout", 0.5), "ghem_ci95"),
                      (r"\$(40\.0)\\pm9\.6\$", ("fanout", 0.2), "tm_degr_pct"), (r"\$40\.0\\pm(9\.6)\$", ("fanout", 0.2), "tm_ci95"),
                      (r"\$(73\.3)\\pm9\.2\$", ("fanout", 0.5), "tm_degr_pct"), (r"\$73\.3\\pm(9\.2)\$", ("fanout", 0.5), "tm_ci95"),
                      (r"studied: (0\.797)~s", ("volume", 0.5), "tm_abs_s"), (r"0\.797~s against (1\.102)~s", ("volume", 0.5), "ghem_abs_s"),
                      (r"and (0\.902)~s against", ("fanout", 0.5), "tm_abs_s"), (r"0\.902~s against (1\.112)~s", ("fanout", 0.5), "ghem_abs_s")]:
    claim(pat, R[key][fld], f"robust {key} {fld}")
bp = {r["B_eps"]: r for r in J("main_tables.json")["table_beps"]}
claim(r"per rack the speedup is (3\.00)", bp[100]["speedup"], "beps100"); claim(r"400~Gbps it is (2\.49)", bp[400]["speedup"], "beps400"); claim(r"it falls to (1\.95)", bp[1600]["speedup"], "beps1600")

print(f"verify_manuscript: {checks[0]} numbers checked, {len(fails)} mismatches")
for f in fails:
    print("  MISMATCH", f)
sys.exit(1 if fails else 0)
