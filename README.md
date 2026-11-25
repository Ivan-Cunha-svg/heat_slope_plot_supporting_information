# Supporting Information: code and data

**A General Heat–Slope Plot for Heterogeneous Adsorbents: Classifying the Tail of the Adsorption Energy
Distribution from Isotherms and Calorimetric Curves** (I. E. Cunha)

This archive contains the code of the protocol, the test bench with its results, the data of Dunne et al.
with their analysis, the scripts of all figures, and the numerical checks quoted in the paper.
Everything is Python 3 (numpy, scipy, matplotlib; mpmath only for the polylogarithm check). Each script is run
from its own folder, e.g. `cd figs && python3 fig_pair.py`.

## `bench/` — protocol and test bench (Tables 2 and 3, and the numbers of the validation section)

| file | content |
|---|---|
| `protocol.py` | the protocol as stated in the paper, with its two routes (isosteric and calorimetric) |
| `systems.py` | the 18 synthetic systems and their exact isotherms and heats |
| `design.py` | design of the synthetic experiments and generation of the noisy data |
| `run_bench.py` → `results.csv.gz` | every system × configuration × replica, with fixed seeds (105 000 analyses) |
| `summarize.py` → `bench_summary.txt` | the numbers of Tables 2 and 3 and of the text |
| `systems2.py`, `run_second.py` → `second_results.csv.gz`, `summarize_second.py` → `second_summary.txt` | calibration of the threshold of the position index and its test on a second set of sixteen systems |
| `crossover_band.py` → `crossover_band_output.txt` | reference range of the scale ratio quoted in the protocol and in Table 4 |
| `check_scale_ratio.py` → `check_scale_ratio_output.txt` | exact check of the identities of the scale ratio and of the position index on the 18 systems |
| `README.md` | conditions of the simulation, rules of the protocol, and when each rule was fixed |

## `figs/` — figures, data of Dunne et al., and numerical checks

| file | content |
|---|---|
| `model.py` | exact forward model (adsorption integral, Langmuir kernel) for the archetypal tails |
| `ward.py` | York regression with errors in both variables, parametric bootstrap |
| `style.py` | common style of the figures |
| `fig_pair.py`, `fig_collapse.py`, `fig_ward.py`, `fig_dunne_ward.py`, `fig_toc.py` | Figures 1–4 and the TOC graphic (each writes a PDF and a PNG) |
| `dunne_data.py` | data of Dunne et al. (Langmuir 1996, 12, 5896), Tables 4–23, transcribed |
| `apply_dunne.py` → `table_dunne.txt` | the calorimetric route of `bench/protocol.py` applied to the 20 data sets (Table 5) |
| `dunne_checks.py` → `dunne_checks.txt` | the checks quoted in the section on those data |
| `check_identities.py` → `check_identities_output.txt` | numerical checks of the identities and numbers quoted in the text |
| `check_gaussian.py` → `check_gaussian_output.txt` | Henry heat of a Gaussian, thermal index of a single site and of a shifted stretched exponential |
| `check_inequalities.py` → `check_inequalities_output.txt` | the two inequalities of Appendix D, and two cases outside their hypotheses |

## Reproducing

```
cd bench
python3 run_bench.py 500 && python3 summarize.py          # about 7 minutes on two cores
python3 run_second.py && python3 summarize_second.py      # about 2 minutes
python3 crossover_band.py && python3 check_scale_ratio.py
cd ../figs
python3 apply_dunne.py && python3 dunne_checks.py
python3 check_identities.py && python3 check_gaussian.py && python3 check_inequalities.py
for f in fig_pair fig_collapse fig_ward fig_dunne_ward fig_toc; do python3 $f.py; done
```

The seeds are fixed (`SEED` in `run_bench.py`, `SEED2` in `run_second.py`). The result files `results.csv.gz` and
`second_results.csv.gz` are included, so that the summaries can be regenerated without rerunning the simulations.
