#!/usr/bin/env python3
"""The protocol (calorimetric route, ../bench/protocol.py) applied to the data of Dunne et al. (1996), one temperature per set.

Each row of their tables is a dose; its heat is attributed to the mean loading of the dose (from the loading of the preceding
row), where ln P and the local slope of the isotherm are interpolated. sigma_Q = 0.5 kJ/mol, as declared by the authors.
Window: n <= 0.3 Ns (CO2/NaX: n <= 0.5 Ns, exploratory). The position index Lambda is one minus the slope of ln a_rec
against ln Q_st over the points of the window with a_rec > 1.2 RT; the scale ratio tau_n = a_cal/a_rec is given with the range
of the same statistic on the exact isotherms of the descending tails of the bench in the same window of a_rec/RT
(../bench/crossover_band.py). Intervals: 4000 parametric bootstrap samples.
Writes table_dunne.txt.  local_slope is kept for the other scripts of this folder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import numpy as np
from dunne_data import DATA, nsat
from protocol import analyse as protocol, R
from crossover_band import band
SQ = 0.5

def local_slope(P, n, half=2):
    """slope of ln n against ln P in a moving window of 2*half+1 points, truncated at the ends of the data; returns m, se(m)."""
    x, y = np.log(P), np.log(n); N = len(x); m = np.empty(N); se = np.empty(N)
    for i in range(N):
        a, b = max(0, i - half), min(N, i + half + 1)
        xx, yy = x[a:b], y[a:b]; A = np.c_[xx, np.ones_like(xx)]
        coef = np.linalg.lstsq(A, yy, rcond=None)[0]; r = yy - A@coef
        m[i] = coef[0]; se[i] = np.sqrt((r@r)/max(len(xx) - 2, 1)/((xx - xx.mean())**2).sum())
    return m, se

def analyse(key, n_max=None, end_of_dose=False, nboot=4000):
    gas, sol, tC = key; T = tC + 273.15; P, n, Q = DATA[key].T; fin = np.isfinite(Q)
    n_start = np.r_[0.0, n[:-1]]
    if end_of_dose: n_start = n.copy()               # sensitivity: the heat attributed to the loading at the end of the dose
    if n_max is None: n_max = 0.5 if (gas, sol) == ("CO2", "NaX") else 0.3
    o = protocol({T: (P, n)}, Ns=nsat(gas, sol), rng=np.random.default_rng(0), nboot=nboot,
                 heat=(n_start[fin], n[fin], Q[fin], np.full(fin.sum(), SQ)), n_max=n_max)
    o.update(key=key, T=T, RT=R*T, n_max=n_max*nsat(gas, sol))
    return o

LABEL = {"end point": "end point (effective edge)", "stretched": "Gumbel tail: stretched, rho = Lambda", "exponential": "Gumbel tail: exponential",
         "growing scale": "Gumbel tail: growing scale", "power law": "power-law tail", "scales inconsistent": "scales inconsistent (tau_n < 0.7)",
         "homogeneous": "homogeneous", "Henry": "Henry regime: tail not resolved", "lateral": "heat rises with loading: lateral interactions",
         "abstain": "abstention (fewer than 5 points)", "xi < 0, not decidable": "xi < 0, existence not decidable"}

def line(o):
    gas, sol, tC = o["key"]; p = o["points"]; ok = p["ok"]
    s = f"{gas+'/'+sol:14s} {tC:5.1f}C  pts={o['npts']:2d}  m_max={o['m_max']:.2f}"
    if o["npts"] >= 5:
        s += f"  a/RT={o['a_lo']:.2f}-{o['a_hi']:.2f}  Q={p['Q'][ok].max():.1f}-{p['Q'][ok].min():.1f}  tau_n={o['tau_n']:5.2f}+-{o['tau_n_se']:.2f}"
        bd = list(band(o["T"], o["a_lo"], o["a_hi"]).values())
        if bd and o["a_hi"] > 1.2: s += f" (bench {min(bd):.2f}-{max(bd):.2f})"
        s += f"  tau={o['tau']:5.2f}+-{o['tau_id_se']:.2f}"
        if np.isfinite(o["Lam"]) and np.isfinite(o["xi"]): s += f"  Lambda={o['Lam']:.2f} [{o['Lam_lo']:.2f};{o['Lam_hi']:.2f}]"
    if np.isfinite(o["xi"]):
        Qw = p["Q"][ok]; s += (f"  xi={o['xi']:+.2f} [{o['xi_lo']:+.2f};{o['xi_hi']:+.2f}]  Eiso={o['Eiso']:.1f} [{o['E_lo']:.1f};{o['E_hi']:.1f}]  k={o['k']:.1f}"
                               f"  extrap={(o['Eiso'] - Qw.max())/(Qw.max() - Qw.min()):.1f} widths")
    elif o["verdict"] in ("Henry", "homogeneous"): s += f"  mean Q={o['Eiso']:.1f}  range={o['spread']:.1f}"
    if np.isfinite(o["dQdn"]): s += f"  dQ/dn={o['dQdn']:+.2f}"
    return s + f"  -> {LABEL[o['verdict']]}"

if __name__ == "__main__":
    lines = [line(analyse(key)) for key in DATA]
    lines.append("")
    for key in (("CO2", "NaX", 32.8), ("CO2", "NaX", 31.4)):
        o = analyse(key, end_of_dose=True); lines.append("heat attributed to the end of the dose: " + line(o))
        o = analyse(key, n_max=0.3); lines.append("formal window n <= 0.3 Ns: " + line(o))
    print("\n".join(lines)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_dunne.txt"), "w").write("\n".join(lines) + "\n")
