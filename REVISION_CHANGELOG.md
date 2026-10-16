# Revision changelog (final check, 25 Sep 2026)

New code
- `scripts/run_significance.py`: paired t-test (log times), exact Wilcoxon signed-rank, Holm adjustment, d_z, rank-biserial.
- `sad_otr/multislot.py::sad_otr_tm_variant` + `scripts/run_tm_ablation.py`: SAD-OTR-TM ablation (slice cap, amortization, first slice, candidate set, wall-clock budget). Default parameters reproduce `sad_otr_tm` exactly.
- `scripts/run_tm_smalldelta.py`: slice-cap explanation at delta = 0.1 ms.
- `sad_otr/llm_workload.py` + `scripts/run_llm_workload.py`: model-derived workload (DeepSeek-V3 published dimensions, arXiv:2412.19437).
- `sad_otr/inputs_io.py`, `scripts/export_inputs.py`, `scripts/reproduce_headline.py`: exported inputs and headline reproduction from them.
- `tools/make_tables.py`, `tools/verify_manuscript.py` (explicit claim registry), `tools/check_numbers.py` (generic scan).

Manuscript corrections found by the final check
- "GHEM-AC and MWM statistically indistinguishable" was wrong: GHEM-AC is faster on 10/10 seeds (Holm p_t = 0.013); text fixed.
- Speedup CIs in Table VI: MWM 0.12 -> 0.11 and Solstice-K 0.08 -> 0.07 (rounding of ci95_half 0.1150 and 0.0749).
- Scalability text: TM within "7 to 12 percent" of bound -> "7 to 11 percent" (max 11.47 %).
- Complexity proposition restated (O(N^2 log N + K N^2 Lambda)), bisection proposition uses rho(T) (clash with retiming R), TM dominance proof notes the tolerance; Eclipse DOI corrected to proceedings DOI.
- "Accounts for that margin" (greedy first slice) softened to "suggests ... most of that margin".
- fig_optimality and fig_delta removed (redundant with Table tab:opt and the delta text) to stay at 16 pages.
- Holm (1979) reference: the DOI 10.2307/4615733 does not resolve (doi.org 404, not in Crossref); replaced by the JSTOR stable URL. All other 42 DOIs resolve at doi.org (checked 25 Sep 2026).
