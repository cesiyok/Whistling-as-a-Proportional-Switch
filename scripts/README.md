# Scripts

The scripts in this folder read only the data package, with paths relative to
their own folder, and print their results; run them from this folder, e.g.
`python maintenance_tracking.py`. They need Python 3 with NumPy, SciPy,
pandas, statsmodels, Matplotlib and openpyxl. `score_trials.py` uses only the
standard library. Column meanings are given in `COLUMNS.md` in the parent
folder. The scripts are licensed under the MIT License (`LICENSE`).

| Script | Computes |
|---|---|
| `score_trials.py` | The per-trial files and `per-participant-summary_data.csv` from the per-tick files (it overwrites them; the scoring definitions of the paper) |
| `maintenance_tracking.py` | Sections V-A to V-C, Tables 1–4, Appendix F-B |
| `mixed_models.py` | Models for maintenance success and for practice effects: Appendix F-A and F-E, Table 12 |
| `subjective_and_range.py` | Subjective measures and calibration: Table 5, Sections V-A, V-D, V-E and V-G, Appendix A-H, and the recency rows of Table 13 |
| `range_associations.py` | Calibrated range and the channel differences, with confidence intervals, the comparison of dependent correlations, and amplitude variability in force points and in dB (shared normalization): Section V-G |
| `designated_tests.py` | The nine designated tests of H1–H4 with the Holm correction across them and the decision rule: Section IV-H, Appendix F-G, Table 14 |
| `checks_and_openloop.py` | Data-integrity, order, fatigue and recency checks, open-loop production noise, session log and pilot measurements: Sections V-A and V-D, Appendix F-F (Table 13), Appendix G |
| `near_break_analysis.py` | Trials closest to a break: Appendix F-C, Table 10 |
| `sensitivity_sweep.py` | All 33 scoring configurations: Appendix E, Table 9 |
| `target_extremes.py` | Low and high targets in intensity maintenance: Appendix F-D, Table 11 |
| `dsp_chain.py` | The intensity signal chain as a module, used by `frequency_dependence.py` and `target_extremes.py` |
| `frequency_dependence.py` | Frequency dependence of the intensity estimate: Appendix A-G, Table 7 (about 7 minutes) |
| `fig_freqdep.py` | Fig. 9 |
| `result_figures.py` | Figs. 5–8 |
