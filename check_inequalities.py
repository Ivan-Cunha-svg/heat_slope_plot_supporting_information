#!/usr/bin/env python3
"""Numerical check of the two inequalities of Appendix D (independent sites, energy-only heterogeneity) at 300 K, with the
Langmuir kernel, the Toth kernel F = y/(1+y^eta)^(1/eta) (eta = 0.3, 0.5, 2) and the Jovanovic kernel F = 1 - exp(-y), on
300 random discrete spectra of up to five types of site (energies uniform in 10-60 kJ/mol, Dirichlet weights) and on the
sixteen continuous distributions of the test bench (their quadrature nodes):
 (0) phi = dln F~/dln y of the appendix against a numerical derivative, and eq D2 against a finite difference of the heat;
 (i)  the slope m = <F~>/<F> does not exceed one;
 (ii) the heat Q = <E>_{F~} does not rise with the pressure.
Outside the hypotheses: a prefactor that grows as the depth falls (entropic inversion), and a kernel whose F~ is not
log-concave in ln y (two subsites of the same depth with prefactors 1 and 400).
Writes check_inequalities_output.txt."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import numpy as np
from scipy.special import expit
from systems import bench_systems, extra_systems
R = 8.314462618e-3; RT = R*300.0; out = []
def say(s): print(s); out.append(s)
def kern(name, u):
    """F, F~ = y F'(y) and phi = dln F~/dln y at u = ln y (arrays)"""
    if name == "Langmuir": F = expit(u); return F, F*expit(-u), expit(-u) - expit(u)
    if name == "Jovanovic":
        y = np.exp(np.minimum(u, 700.0)); return -np.expm1(-y), y*np.exp(-y), 1.0 - y
    eta = float(name.split()[1]); l = np.logaddexp(0.0, eta*u); F = np.exp(u - l/eta); p = expit(eta*u)       # p = y^eta/(1+y^eta)
    return F, F*np.exp(-l), 1.0 - (1.0 + eta)*p
KERNELS = ("Langmuir", "Toth 0.3", "Toth 0.5", "Toth 2", "Jovanovic")
rng = np.random.default_rng(5)
spectra = [(rng.uniform(10, 60, k), rng.dirichlet(np.ones(k))) for k in rng.integers(1, 6, 300)]
spectra += [(s.E, s.w) for s in bench_systems() + extra_systems() if s.cls != "discrete"]
lp = np.linspace(-45.0, 8.0, 1200)
say("(0) phi of the appendix against a numerical derivative of ln F~, and eq D2 against a finite difference (largest differences)")
for name in KERNELS:
    u = np.linspace(-8, 3, 400); h = 1e-5; num = (np.log(kern(name, u + h)[1]) - np.log(kern(name, u - h)[1]))/(2*h)
    E, w = spectra[7]; d = 0.0
    for x in np.linspace(-22, -6, 9):
        def Q(xx): g = w*kern(name, xx + E/RT)[1]; return (g*E).sum()/g.sum()
        g = w*kern(name, x + E/RT)[1]; g /= g.sum(); ph = kern(name, x + E/RT)[2]
        d = max(d, abs((Q(x + 1e-4) - Q(x - 1e-4))/2e-4 - ((g*E*ph).sum() - (g*E).sum()*(g*ph).sum())))
    say(f"    {name:10s} phi: {np.abs(num - kern(name, u)[2]).max():.1e};  eq D2: {d:.1e} kJ/mol")
say("(i), (ii) over 300 random discrete spectra and the 16 continuous distributions of the bench: largest slope m minus one; largest increase of Q between neighbouring pressures (kJ/mol)")
for name in KERNELS:
    mx = -1.0; up = -1e9
    for E, w in spectra:
        F, Ft, _ = kern(name, lp[:, None] + E[None, :]/RT); den = (w*Ft).sum(1); ok = den > 1e-280
        mx = max(mx, np.max(den[ok]/(w*F).sum(1)[ok]) - 1.0); up = max(up, np.max(np.diff(((w*Ft*E).sum(1)[ok]/den[ok]))))
    say(f"    {name:10s} max(m) - 1 = {mx:+.1e};  largest rise of Q = {up:+.1e}")
say("outside the hypotheses (Langmuir local isotherm):")
E = np.array([45.0, 30.0]); w = np.array([0.5, 0.5]); g = w*kern("Langmuir", lp[:, None] + np.array([-12.0, 0.0]) + E/RT)[1]; Q = (g*E).sum(1)/g.sum(1)
say(f"    prefactor e^-12 on the deeper of two types of site: Q rises from {Q[0]:.1f} to {Q.max():.1f} kJ/mol")
def two(u): return 0.5*(kern("Langmuir", u)[1] + kern("Langmuir", u + np.log(400.0))[1])        # F~ of a site with two subsites
u = np.linspace(-12, 6, 2000); c2 = np.diff(np.log(two(u)), 2).max()/(u[1] - u[0])**2
E2 = np.array([45.0, 40.0]); g = w*two(lp[:, None] + E2/RT); Q = (g*E2).sum(1)/g.sum(1); j = np.argmax(Q - np.minimum.accumulate(Q)); i = np.argmin(Q[:j + 1])
say(f"    kernel with two subsites of the same depth (prefactors 1 and 400), sites of 45 and 40 kJ/mol: largest d2 ln F~/d(ln y)2 = {c2:+.2f} (not log-concave); Q falls to {Q[i]:.2f} and rises again to {Q[j]:.2f} kJ/mol")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_inequalities_output.txt"), "w").write("\n".join(out) + "\n")
