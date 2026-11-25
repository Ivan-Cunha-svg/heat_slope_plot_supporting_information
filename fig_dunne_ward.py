#!/usr/bin/env python3
"""Figure 4: Ward plots of the calorimetric data of Dunne et al. (1996): a_rec = RT/m against the calorimetric isosteric heat
of each dose, at the mean loading of the dose (calorimetric route of the protocol, apply_dunne.py).
Filled symbols: admissible window (n <= 0.3 Ns; CO2/NaX: n <= 0.5 Ns); open symbols: points beyond it."""
import numpy as np
from style import *
from apply_dunne import analyse, SQ
panels = [("CO$_2$/NaX", [("CO2", "NaX", 32.8), ("CO2", "NaX", 31.4)]), ("CO$_2$/Na-ZSM-5", [("CO2", "Na-ZSM-5", 24.1)]),
          ("CO$_2$/H-ZSM-5", [("CO2", "H-ZSM-5", 23.9), ("CO2", "H-ZSM-5", 22.5)]), ("N$_2$/Na-ZSM-5", [("N2", "Na-ZSM-5", 21.9)]),
          ("SF$_6$/NaX", [("SF6", "NaX", 31.6)]), ("CH$_4$/NaX", [("CH4", "NaX", 31.3)])]
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.7))
for ax, (title, keys), lab in zip(axs.ravel(), panels, "abcdef"):
    for key, col, mk in zip(keys, (C[0], C[1]), ("o", "s")):
        o = analyse(key, nboot=200); RT = o["RT"]; p = o["points"]; ok = p["ok"]; out = np.isfinite(p["Q"]) & ~ok
        ax.errorbar(p["Q"][ok], p["a"][ok]/RT, xerr=SQ, yerr=p["sa"][ok]/RT, fmt=mk, ms=4, color=col, mec=col, elinewidth=0.5, label=f"{key[2]:.1f} °C", zorder=4)
        ax.plot(p["Q"][out], p["a"][out]/RT, mk, ms=4, mfc="white", mec=col, mew=0.9, zorder=3)
        if np.isfinite(o["xi"]) and key[:2] == ("CO2", "NaX"):
            qq = np.array([p["Q"][ok].min() - 1.0, o["Eiso"]]); ax.plot(qq, o["xi"]*(qq - o["Eiso"])/RT, ":", color=col, lw=1.0)
    ax.axhline(1.0, color="0.6", lw=0.6, ls=(0, (1, 2))); ax.set_title(title, fontsize=8); panel(ax, lab)
    ax.set_xlabel(r"$Q_{\rm st}$ (kJ/mol)"); ax.set_ylabel(r"$a_{\rm rec}/RT$")
    if len(keys) > 1: ax.legend(frameon=False, loc="upper right")
axs[0, 0].set_ylim(0, 9); axs[0, 0].set_xlim(34, 54); axs[0, 1].set_ylim(0, 9); axs[0, 2].set_ylim(0, 5)
axs[1, 0].set_ylim(0, 3); axs[1, 0].set_xlim(18.5, 24.5); axs[1, 1].set_ylim(0, 9); axs[1, 2].set_ylim(0, 3); axs[1, 2].set_xlim(17.5, 21)
fig.tight_layout(); fig.savefig("fig_dunne_ward.pdf"); fig.savefig("fig_dunne_ward.png", dpi=200); print("ok")
