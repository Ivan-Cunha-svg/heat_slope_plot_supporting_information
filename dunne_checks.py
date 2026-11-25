#!/usr/bin/env python3
"""Checks quoted in the section on the data of Dunne et al. (1996), in addition to table_dunne.txt. Writes dunne_checks.txt.

 A. Reference state: the position index with Q_st - kappa RT in the place of Q_st (kappa = 1 and 2.5).
 B. Local equivalence of the two readings of a descending plot (eq indist of the paper), on the windows of the data: a power-law
    edge with the fitted (E_iso, k) (width 80 kJ/mol, nearly complete) and a stretched exponential with rho = Lambda, its
    scale chosen so that the heats agree at the centre of the window; exact isotherms, Langmuir kernel, P0 constant. For both,
    in the window of a_rec/RT of the data: range of heats, scale ratio, position index, dln m/dln n, and the quadratic
    coefficient of Q_st in ln n.
 C. Shape of the heat curve: weighted quadratic fit of Q_st in ln n over the admissible points (sigma_Q = 0.5 kJ/mol).
 D. Heats that rise with the loading: slope of Q_st against n over the first half of the doses that carry a heat.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import numpy as np
from scipy.optimize import brentq
from apply_dunne import analyse, DATA, SQ
from protocol import york, R, HENRY_A
from systems import edge, stretched
out = []
def say(s=""): print(s); out.append(s)
SETS = [("CO2", "NaX", 32.8), ("CO2", "NaX", 31.4), ("CO2", "Na-ZSM-5", 24.1), ("CO2", "H-ZSM-5", 23.9), ("CO2", "H-ZSM-5", 22.5), ("N2", "Na-ZSM-5", 21.9)]
def name(key): return f"{key[0]}/{key[1]} {key[2]:.1f}C"
def quad(x, y, sd):
    X = x - x.mean(); A = np.c_[np.ones_like(X), X, X*X]; c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c[2], sd*np.sqrt(np.linalg.inv(A.T@A)[2, 2])

say("A. Position index and the reference state: Lambda with Q_st, and with Q_st - kappa RT for kappa = 1 and 2.5")
res = {}
for key in SETS:
    o = analyse(key, nboot=400); RT = o["RT"]; p = o["points"]; sel = p["ok"] & (p["a"] > HENRY_A*RT); res[key] = o
    L = [1 - york(np.log(p["Q"][sel] - kp*RT), np.log(p["a"][sel]), p["sQ"][sel]/(p["Q"][sel] - kp*RT), p["sa"][sel]/p["a"][sel])[1] for kp in (0, 1, 2.5)]
    say(f"   {name(key):22s} {L[0]:.2f}; {L[1]:.2f}; {L[2]:.2f}")

def profile(s, T):
    th = np.logspace(-7, np.log10(0.3), 400); lp = np.array([s.lnP_at(t, T) for t in th])
    return th, R*T/s.slope(lp, T), s.heat(lp, T)
def window(s, T, lo, hi):
    th, a, Q = profile(s, T); w = (a/(R*T) >= lo) & (a/(R*T) <= hi); x = np.log(th[w])
    return dict(Qhi=Q[w].max(), Qlo=Q[w].min(), tn=-np.polyfit(x, Q[w], 1)[0]/a[w].mean(), Lam=1 - np.polyfit(np.log(Q[w]), np.log(a[w]), 1)[0],
                s=np.polyfit(x, np.log(R*T/a[w]), 1)[0], c2=np.polyfit(x, Q[w], 2)[0])
say("\nB. An edge with the fitted (E_iso, k) and a stretched exponential with rho = Lambda, in the window of the data")
say("   (heats in kJ/mol; tau_n; Lambda; dln m/dln n; quadratic coefficient of Q_st in ln n)")
for key in (("CO2", "NaX", 32.8), ("CO2", "Na-ZSM-5", 24.1), ("CO2", "H-ZSM-5", 22.5), ("N2", "Na-ZSM-5", 21.9)):
    o = res[key]; T = o["T"]; RT = o["RT"]; p = o["points"]; ok = p["ok"]; lo, hi = o["a_lo"], o["a_hi"]; x = np.log(p["n"][ok])
    c2, se2 = quad(x, p["Q"][ok], SQ); j = np.argsort(p["a"][ok]); Qmid = np.interp(0.5*(lo + hi)*RT, p["a"][ok][j], p["Q"][ok][j])
    say(f"   {name(key)}: a_rec/RT {lo:.2f}-{hi:.2f}")
    say(f"      {'data':34s} Q {p['Q'][ok].max():.1f}-{p['Q'][ok].min():.1f}; tau_n {o['tau_n']:.2f} +- {o['tau_n_se']:.2f}; Lambda {o['Lam']:.2f}; dln m/dln n {np.polyfit(x, np.log(p['m'][ok]), 1)[0]:+.3f}; curvature {c2:+.2f} +- {se2:.2f}")
    def mid(E0):
        th, a, Q = profile(stretched("s", "", o["Lam"], E0), T); return Q[np.argmin(abs(a/RT - 0.5*(lo + hi)))] - Qmid
    E0 = brentq(mid, 5.0, 200.0, xtol=1e-3)
    for lab, s in ((f"edge, E_iso {o['Eiso']:.1f}, k {o['k']:.1f}", edge("e", "", o["k"], o["Eiso"], 80.0)), (f"stretched, rho {o['Lam']:.2f}, E_0 {E0:.1f}", stretched("s", "", o["Lam"], E0))):
        w = window(s, T, lo, hi)
        say(f"      {lab:34s} Q {w['Qhi']:.1f}-{w['Qlo']:.1f}; tau_n {w['tn']:.2f}; Lambda {w['Lam']:.2f}; dln m/dln n {w['s']:+.3f}; curvature {w['c2']:+.2f}")

say("\nC. Quadratic coefficient of Q_st in ln n over the admissible points (kJ/mol; negative: the fall accelerates with the loading)")
for key in SETS:
    o = res[key]; p = o["points"]; ok = p["ok"]; c2, se2 = quad(np.log(p["n"][ok]), p["Q"][ok], SQ)
    say(f"   {name(key):22s} {c2:+.2f} +- {se2:.2f}  ({ok.sum()} points)")

say("\nD. Slope of Q_st against n over the first half of the doses with a heat (kJ/mol per mmol/g)")
for key in (("SF6", "NaX", 31.6), ("C2H6", "NaX", 32.4)):
    P, n, Q = DATA[key].T; nm = 0.5*(np.r_[0.0, n[:-1]] + n); f = np.isfinite(Q); x, y = nm[f], Q[f]; h = len(x)//2; x, y = x[:h], y[:h]
    b = np.polyfit(x, y, 1)[0]; se = SQ/np.sqrt(((x - x.mean())**2).sum())
    say(f"   {name(key):22s} {b:+.2f} +- {se:.2f} ({h} doses, n <= {x.max():.2f} mmol/g, Q_st from {y[0]:.1f} to {y[-1]:.1f} kJ/mol)")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dunne_checks.txt"), "w").write("\n".join(out) + "\n")
