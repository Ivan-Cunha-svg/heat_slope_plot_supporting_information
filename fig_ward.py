#!/usr/bin/env python3
"""Figure 3: Ward plots (a_rec = RT/m against the isosteric heat at the same loading) for four archetypal site-energy
distributions: exact curves at three temperatures (Langmuir kernel, P0 constant) and two synthetic experiments with the
design and the realistic noise of the test bench (20 points per isotherm, 1.5 % in n, 1 % in P), one analysed by the
isosteric route (three isotherms) and one by the calorimetric route (one isotherm and the heat of each dose, 0.5 kJ/mol).
The points are those of the protocol (../bench/protocol.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from style import *
from systems import bench_systems
from design import exact_design, exact_doses, noisy, noisy_cal, LEVELS, SIGMA_Q, TEMPS
from protocol import analyse
R = 8.314462618e-3; T0 = 300.0; RT0 = R*T0
BENCH = {s.key: s for s in bench_systems()}
panels = [("power-law edge (Weibull), k = 2", "W2"), ("stretched exponential (Gumbel), ρ = 2", "G3"),
          ("power law (Fréchet), ξ = 1/2", "F1"), ("single site (35 kJ/mol)", "H1")]
sn, sP, npts = LEVELS["realistic"]; rng = np.random.default_rng(7)
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
for ax, (title, key), lab in zip(axs.ravel(), panels, "abcd"):
    s = BENCH[key]; ex = exact_design(s, npts)
    for T, col, ls in zip(TEMPS, (C[0], C[1], C[2]), ("-", "--", ":")):
        lnP = np.linspace(ex[T][0][0] - 2.0, s.lnP_at(0.3, T), 80)
        ax.plot(s.heat(lnP, T), R*T/s.slope(lnP, T)/RT0, ls, color=col, lw=1.2, label=f"exact, {int(T)} K")
    o = analyse(noisy(ex, sn, sP, rng), rng=rng, nboot=20); p = o["points"]; ok = p["ok"] & np.isfinite(p["Q"])
    ax.errorbar(p["Q"][ok], p["a"][ok]/RT0, xerr=p["sQ"][ok], yerr=p["sa"][ok]/RT0, fmt="o", ms=3.2, color="0.2", mfc="white", mew=0.8,
                elinewidth=0.5, capsize=0, label="isosteric route", zorder=4)
    data, heat = noisy_cal(ex, exact_doses(s, ex, T0), T0, sn, sP, SIGMA_Q["realistic"], rng)
    o = analyse(data, rng=rng, heat=heat, nboot=20); p = o["points"]; ok = p["ok"] & np.isfinite(p["Q"])
    ax.errorbar(p["Q"][ok], p["a"][ok]/RT0, xerr=p["sQ"][ok], yerr=p["sa"][ok]/RT0, fmt="s", ms=3.0, color=C[3], mfc=C[3], mew=0.0,
                elinewidth=0.5, capsize=0, label="calorimetric route", zorder=5)
    ax.axhline(1.0, color="0.6", lw=0.6, ls=(0, (1, 2))); ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"$Q_{\rm st}$ (kJ/mol)"); ax.set_ylabel(r"$a_{\rm rec}/RT_0$"); panel(ax, lab)
axs[0, 0].plot([50.0], [0], "*", color="0.2", ms=8, clip_on=False, zorder=6); axs[0, 0].set_ylim(0, 5); axs[0, 0].set_xlim(30, 51)
axs[0, 0].annotate(r"$E_{\max}^{\rm iso}$", (50, 0.05), (47.2, 0.9), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
axs[0, 0].plot([50 - 2*2.6*RT0, 50], [2.6, 0], color="0.5", lw=0.6, ls=(0, (4, 2)), zorder=1)
axs[0, 1].set_ylim(0, 5); axs[1, 0].set_ylim(0, 12); axs[1, 1].set_ylim(0.6, 2.0); axs[1, 1].set_xlim(32, 38)
axs[0, 0].legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("fig_ward.pdf"); fig.savefig("fig_ward.png", dpi=200); print("ok")
