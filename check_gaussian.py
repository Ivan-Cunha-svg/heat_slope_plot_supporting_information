#!/usr/bin/env python3
"""Gaussian distributions of site energies in exact arithmetic: the three statements quoted in the text.
(1) Henry heat  q_H = Ebar + s^2/RT  and the range of the heat over the window  theta <= 0.3;
(2) thermal index of a single type of site,  tau = (E/RT) theta  (eq 30 of the text);
(3) thermal index of a stretched exponential about an origin E1:  tau = 1 + (rho-1) Ec/(Ec-E1)  (condensation approximation).
The behavior of Gaussians under the protocol, with noise, is in the test bench (../bench).
Writes check_gaussian_output.txt.  P0 is constant (Delta H_0 = 0), Langmuir kernel, energies in kJ/mol."""
import numpy as np
from scipy.optimize import brentq
from model import R, _F, _xg, _wg, slope_m, isosteric_heat, SingleSite

class Gaussian:
    """-dG = exp[-(E-Ebar)^2/(2 s^2)] dE / (s sqrt(2 pi))"""
    def __init__(self, Ebar, s): self.Ebar, self.s = Ebar, s
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); out = np.zeros(len(lnz)); ed = np.linspace(-9, 9, 13)
        for a, b in zip(ed[:-1], ed[1:]):
            t = 0.5*(b - a)*_xg + 0.5*(a + b); w = 0.5*(b - a)*_wg; E = self.Ebar + self.s*t
            out += (_F(lnz[:, None] + E[None, :]/(R*T))*(np.exp(-t*t/2)/np.sqrt(2*np.pi)*w)[None, :]).sum(axis=1)
        return out

class ShiftedStretched:
    """G = exp{-[(E-E1)/E0]^rho}, E >= E1"""
    def __init__(self, rho, E0, E1): self.rho, self.E0, self.E1 = rho, E0, E1
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); out = np.zeros(len(lnz)); ed = np.linspace(0, 60, 13)
        for a, b in zip(ed[:-1], ed[1:]):
            t = 0.5*(b - a)*_xg + 0.5*(a + b); w = 0.5*(b - a)*_wg; E = self.E1 + self.E0*t**(1/self.rho)
            out += (_F(lnz[:, None] + E[None, :]/(R*T))*(np.exp(-t)*w)[None, :]).sum(axis=1)
        return out

def tau_z(s, lnz, T, d=0.5):
    """thermal index at fixed z (exact slopes)"""
    return (np.log(slope_m(s, lnz, T + d)) - np.log(slope_m(s, lnz, T - d)))/(np.log(T + d) - np.log(T - d))

T0 = 300.0; RT = R*T0; Ts = (282.0, 300.0, 318.0); out = []
def say(x): print(x); out.append(x)
def lnz_at_theta(s, th): return brentq(lambda x: s.theta(np.array([x]), T0)[0] - th, -40, 10)

say(f"RT = {RT:.3f} kJ/mol at {T0:.0f} K")
say("\n(1) Gaussian, Ebar = 35: Henry heat and range of the heat between theta = 1e-4 and 0.3")
for sd in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
    g = Gaussian(35.0, sd); l = np.array([lnz_at_theta(g, 1e-4), lnz_at_theta(g, 0.3)])
    Q = isosteric_heat(g, l, T0); m = slope_m(g, l, T0)
    say(f"  s = {sd:3.1f}: q_H = {Q[0]:.2f} (Ebar + s^2/RT = {35 + sd**2/RT:.2f}); Q(0.3) = {Q[1]:.2f}; range = {Q[0]-Q[1]:.2f} = {(Q[0]-Q[1])/RT:.2f} RT; m(0.3) = {m[1]:.3f} (single site: 0.700)")
sd_RT = brentq(lambda sd: np.subtract(*isosteric_heat(Gaussian(35.0, sd), np.array([lnz_at_theta(Gaussian(35.0, sd), 1e-4), lnz_at_theta(Gaussian(35.0, sd), 0.3)]), T0)) - RT, 2.0, 4.0)
say(f"  the range equals RT at s = {sd_RT:.2f} kJ/mol = {sd_RT/RT:.2f} RT")

say("\n(2) single site, E = 35: tau at fixed z against (E/RT) theta")
s1 = SingleSite(35.0)
for th in (0.01, 0.1, 0.2, 0.3):
    l = np.array([lnz_at_theta(s1, th)]); say(f"  theta = {th:4.2f}: tau = {tau_z(s1, l, T0)[0]:.3f}; (E/RT) theta = {35/RT*th:.3f}")

say("\n(3) stretched exponential about E1 = 20 (rho = 2, E0 = 40): tau at fixed z against 1 + (rho-1) Ec/(Ec-E1)")
ss = ShiftedStretched(2.0, 40.0, 20.0)
for Ec in (60.0, 80.0, 100.0):
    l = np.array([-Ec/RT]); say(f"  Ec = {Ec:.0f}: a/RT = {1/slope_m(ss, l, T0)[0]:.2f}; tau = {tau_z(ss, l, T0)[0]:.2f}; CA = {1 + Ec/(Ec - 20):.2f}")

open("check_gaussian_output.txt", "w").write("\n".join(out) + "\n")
