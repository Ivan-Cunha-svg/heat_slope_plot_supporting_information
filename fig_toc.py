#!/usr/bin/env python3
"""Table-of-contents graphic (3.25 in x 1.75 in): sites of different depths, the tail of their distribution,
and the heat-slope plot whose straight line ends at the strongest site. The plot uses the exact model (edge k = 2)."""
import numpy as np
from style import *
from model import Edge, slope_m, isosteric_heat, R
INK, MUTED, SOLID, EDGE = "#1f2530", "#5b6472", "#e3e6eb", "#3d4654"; BLUE, ORANGE = C[0], C[1]
fig = plt.figure(figsize=(3.25, 1.75))

# --- left: energy landscape of the surface --------------------------------------------------------------------------
aL = fig.add_axes([0.035, 0.07, 0.345, 0.80]); aL.set_xlim(0, 1); aL.set_ylim(-1.22, 0.50); aL.axis("off")
xi = np.linspace(0.09, 0.91, 7); di = np.array([0.34, 0.58, 0.24, 1.00, 0.42, 0.72, 0.30]); w = 0.040
x = np.linspace(0, 1, 1500); y = -(di[:, None]*np.exp(-0.5*((x[None, :] - xi[:, None])/w)**2)).sum(0)
aL.fill_between(x, y, -1.22, color=SOLID, lw=0); aL.plot(x, y, color=EDGE, lw=0.9, solid_capstyle="round")
for i in (3, 5): aL.plot(xi[i], -di[i] + 0.085, "o", ms=5.2, mfc=BLUE, mec="white", mew=0.5, zorder=4)       # adsorbed, deepest sites
for gx, gy in ((0.13, 0.26), (xi[3], 0.40), (0.87, 0.20)): aL.plot(gx, gy, "o", ms=5.2, mfc="white", mec=BLUE, mew=0.9)   # gas
aL.annotate("", xy=(xi[3], -0.50), xytext=(xi[3], 0.30), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6))
fig.text(0.035 + (0.345 + 0.13)/2, 0.905, "dilute gas, uneven sites", ha="center", va="bottom", fontsize=6.6, color=INK, fontweight="bold")

# --- middle: distribution of site depths, turned sideways on the same depth axis --------------------------------------
aD = fig.add_axes([0.385, 0.07, 0.125, 0.80]); aD.set_xlim(0, 1.12); aD.set_ylim(-1.22, 0.50); aD.axis("off")
u = np.linspace(0, 1, 400); g = u**1.2*(1 - u)**2; g /= g.max(); cut = 0.62
aD.fill_betweenx(-u, 0, g, color=SOLID, lw=0); aD.fill_betweenx(-u[u >= cut], 0, g[u >= cut], color=BLUE, lw=0, alpha=0.85)
aD.plot(g, -u, color=EDGE, lw=0.8); aD.plot([0, 0], [0, -1], color=EDGE, lw=0.6)
aD.plot([0], [-1], marker="*", ms=8.5, color=ORANGE, mec="white", mew=0.4, clip_on=False, zorder=5)
aD.text(0.47, -0.86, "tail", fontsize=6.6, color=INK, va="center"); aD.text(0.17, -1.03, r"$E_{\max}$", fontsize=7.2, color=INK, va="center")
fig.lines.append(plt.Line2D([0.035 + 0.345*(xi[3] + 0.05), 0.385], [0.07 + 0.80*(0.22/1.72)]*2, transform=fig.transFigure, color=ORANGE, lw=0.7, ls=(0, (2, 2))))

# --- right: the heat-slope plot ---------------------------------------------------------------------------------------
T = 300.0; RT = R*T; s = Edge(2, 50.0, 30.0); lz = np.linspace(-50/RT - 6, -50/RT + 8, 400)
lnz = np.interp(np.linspace(1.08, 3.85, 9), 1/slope_m(s, lz, T), lz)                   # points evenly spaced in RT/m
m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T); rng = np.random.default_rng(3)
aP = fig.add_axes([0.635, 0.215, 0.335, 0.60])
aP.plot([50 - 2*4.3*RT, 50], [4.3, 0], color=EDGE, lw=1.0, zorder=2)
aP.plot(Q + 0.25*rng.standard_normal(len(Q)), 1/m*(1 + 0.03*rng.standard_normal(len(Q))), "o", ms=3.6, mfc="white", mec=BLUE, mew=1.1, zorder=3)
aP.plot([50], [0], marker="*", ms=10, color=ORANGE, mec="white", mew=0.4, clip_on=False, zorder=5)
aP.set_xlim(28.5, 51.3); aP.set_ylim(0, 4.7); aP.set_xticks([]); aP.set_yticks([])
aP.text(28.9, -0.34, "isosteric heat", fontsize=6.8, color=INK, va="top", clip_on=False); aP.set_ylabel("1 / isotherm slope", fontsize=6.8, color=INK, labelpad=3)
aP.text(50.0, -0.34, r"$E_{\max}$", ha="center", va="top", fontsize=7.2, color=INK, clip_on=False)
aP.text(51.2, 3.95, r"$Q_{\rm st}+k\,\dfrac{RT}{m}=E_{\max}$", fontsize=6.3, color=INK, ha="right", va="center")
aP.axhline(1, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1); aP.text(29.0, 0.90, "Henry's law", fontsize=6.2, color=MUTED, va="top")
fig.text(0.635 + 0.335/2, 0.905, "heat–slope plot", ha="center", va="bottom", fontsize=6.6, color=INK, fontweight="bold")
fig.patches.append(matplotlib.patches.FancyArrowPatch((0.522, 0.50), (0.572, 0.50), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.2))
fig.savefig("fig_toc.pdf"); fig.savefig("fig_toc.png", dpi=600); print("ok")
