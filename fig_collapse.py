#!/usr/bin/env python3
"""Figure 2: thermal collapse and the exact linear law for a power-law edge with a Langmuir kernel (Emax = 50, W = 40 kJ/mol)."""
import numpy as np
from math import gamma
from style import *
from model import Edge, slope_m, isosteric_heat, R
Emax, W = 50.0, 40.0
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
xs = np.linspace(-7, 12, 200)
for k, col in zip((1, 2, 3), C):
    sg = np.linspace(0, 80, 160001)[1:]                                          # Psi_k(x) = k int_0^inf s^(k-1)/(1+e^(s-x)) ds
    Psi = [k*np.trapezoid(sg**(k - 1)*0.5*(1 - np.tanh(0.5*(sg - x))), sg) for x in xs]   # = Gamma(k+1) F_{k-1}(x) = -Gamma(k+1) Li_k(-e^x)
    a.plot(xs, Psi, color=col, label=f"k = {k}")
    for T, mk in ((270.0, "o"), (300.0, "s"), (330.0, "^")):
        x = np.linspace(-6.5, 11.5, 10) + {270.0: 0.0, 300.0: 0.6, 330.0: 1.2}[T]; RT = R*T
        th = Edge(k, Emax, W).theta(x - Emax/RT, T)
        a.plot(x, th*(W/RT)**k, mk, ms=3.6, mfc="white", mec=col, mew=0.9)
a.set_yscale("log"); a.set_xlabel(r"$x=(E_{\max}-E_c)/RT$"); a.set_ylabel(r"$\theta\,(W/RT)^k$"); a.set_ylim(5e-4, 3e3)
a.axvline(0, color="0.6", lw=0.6, ls=":"); a.text(-0.3, 1.2e3, "Henry", ha="right", fontsize=7, color="0.35"); a.text(0.3, 1.2e3, "tail", ha="left", fontsize=7, color="0.35")
h = [plt.Line2D([], [], color=C[i], label=f"k = {i+1}") for i in range(3)] + [plt.Line2D([], [], ls="", marker=m, mfc="white", mec="0.3", ms=4, label=f"{T} K") for m, T in (("o", 270), ("s", 300), ("^", 330))]
a.legend(handles=h, frameon=False, ncol=2, loc="lower right"); panel(a, "a")
k = 2; s = Edge(k, Emax, W)
aa = np.linspace(0, 11.5, 10); b.plot(aa, Emax - k*aa, "-", color="0.25", lw=1.0, label=r"$Q_z=E_{\max}-k\,a_{\rm rec}$")
for T, mk, col in ((270.0, "o", C[0]), (300.0, "s", C[1]), (330.0, "^", C[2])):
    RT = R*T; lnz = np.linspace(-Emax/RT - 6, -Emax/RT + 8.5, 14); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T)
    b.plot(RT/m, Q, mk, ms=4, mfc="white", mec=col, mew=1.0, label=f"{int(T)} K")
    b.plot([RT], [Emax - k*RT], "*", color=col, ms=8, zorder=5)
b.set_xlim(0, 11.5); b.set_ylim(26, 51); b.set_xlabel(r"$a_{\rm rec}=RT/m$ (kJ/mol)"); b.set_ylabel(r"$Q_z$ (kJ/mol)"); b.legend(frameon=False, title="k = 2"); panel(b, "b")
fig.tight_layout(); fig.savefig("fig_collapse.pdf"); fig.savefig("fig_collapse.png", dpi=200); print("ok")
