#!/usr/bin/env python3
"""Exact forward model used by the figures: the adsorption integral with a Langmuir kernel,
theta(z,T) = int F(z e^{E/RT}) (-dG(E)),  F(y) = y/(1+y),  z = P/P0 (P0 constant, so Delta H_0 = 0),
for the archetypal site-energy tails, by Gauss-Legendre quadrature. Energies in kJ/mol."""
import numpy as np
R = 8.314462618e-3   # kJ/mol/K
_xg, _wg = np.polynomial.legendre.leggauss(400)

def _F(lny):                       # Langmuir kernel in ln y, overflow-safe
    return 0.5*(1.0 + np.tanh(0.5*lny))

class Edge:
    """Pure power-law (Weibull) edge: -G'(E) = k (Emax-E)^(k-1) / W^k on [Emax-W, Emax]."""
    def __init__(self, k, Emax, W): self.k, self.Emax, self.W = k, Emax, W; self.label = f"edge k={k}"
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); k, W = self.k, self.W
        # substitution v = (Emax-E) = W u^(1/k): -dG = du, u in (0,1)
        u = 0.5*(_xg + 1.0); w = 0.5*_wg
        E = self.Emax - W*u**(1.0/k)
        return (_F(lnz[:, None] + E[None, :]/(R*T))*w[None, :]).sum(axis=1)

class Stretched:
    """Stretched-exponential (Gumbel) tail: G(E) = exp[-(E/E0)^rho]; rho = 2 is Dubinin-Radushkevich."""
    def __init__(self, rho, E0): self.rho, self.E0 = rho, E0; self.label = f"stretched rho={rho}"
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz)
        # substitution u = G(E) in (0,1): -dG = du, E = E0 (-ln u)^(1/rho); split to resolve the far tail
        out = np.zeros(len(lnz))
        edges = [0.0, 1e-12, 1e-8, 1e-5, 1e-3, 3e-2, 0.3, 1.0]
        tmax = 60.0                                   # s = -ln u in (0, tmax): -dG = e^{-s} ds
        s_edges = np.linspace(0.0, tmax, 13)
        for a, b in zip(s_edges[:-1], s_edges[1:]):
            s = 0.5*(b - a)*_xg + 0.5*(a + b); w = 0.5*(b - a)*_wg
            E = self.E0*s**(1.0/self.rho)
            out += (_F(lnz[:, None] + E[None, :]/(R*T))*(np.exp(-s)*w)[None, :]).sum(axis=1)
        return out

class PowerLaw:
    """Power-law (Frechet) tail with lower cut: G(E) = (E/E1)^(-1/xi), E >= E1."""
    def __init__(self, xi, E1): self.xi, self.E1 = xi, E1; self.label = f"power law xi={xi}"
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); out = np.zeros(len(lnz))
        # u = G in (0,1): E = E1 u^(-xi); integrate in s = -ln u
        s_edges = np.linspace(0.0, 60.0, 13)
        for a, b in zip(s_edges[:-1], s_edges[1:]):
            s = 0.5*(b - a)*_xg + 0.5*(a + b); w = 0.5*(b - a)*_wg
            E = self.E1*np.exp(self.xi*s)
            out += (_F(lnz[:, None] + np.minimum(E, 5e3)[None, :]/(R*T))*(np.exp(-s)*w)[None, :]).sum(axis=1)
        return out

class SingleSite:
    def __init__(self, E): self.E = E; self.label = "single site"
    def theta(self, lnz, T): return _F(np.atleast_1d(lnz) + self.E/(R*T))

def slope_m(sysm, lnz, T, h=1e-4):
    """local slope m = d ln theta / d ln z at fixed T (central difference)."""
    return (np.log(sysm.theta(lnz + h, T)) - np.log(sysm.theta(lnz - h, T)))/(2*h)

def isosteric_heat(sysm, lnz, T, dT=0.5):
    """Q_z = -R d ln z/d(1/T) at fixed theta, by solving theta(lnz', T+-dT) = theta(lnz, T)."""
    from scipy.optimize import brentq
    lnz = np.atleast_1d(lnz); th = sysm.theta(lnz, T); out = np.empty(len(lnz))
    for i, (l, t) in enumerate(zip(lnz, th)):
        f = lambda x, TT: np.log(sysm.theta(np.array([x]), TT)[0]) - np.log(t)
        lp = brentq(f, l - 3, l + 3, args=(T + dT,), xtol=1e-12); lm = brentq(f, l - 3, l + 3, args=(T - dT,), xtol=1e-12)
        out[i] = -R*(lp - lm)/(1.0/(T + dT) - 1.0/(T - dT))
    return out

if __name__ == "__main__":
    T = 300.0; RT = R*T
    # checks quoted in the text: Q_z + k a_rec - Emax for pure edges, Henry heat, exponential shift
    for k, Em, W in ((1, 45, 20), (2, 50, 22), (3, 55, 25)):
        s = Edge(k, Em, W); lnz = np.linspace(-Em/RT - 8, -Em/RT + 12, 9)
        th = s.theta(lnz, T); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T)
        print(f"edge k={k}: theta {th[0]:.1e}..{th[-1]:.2f}; max |Q + k RT/m - Emax| for theta<0.3: {np.max(np.abs(Q + k*RT/m - Em)[th<0.3]):.3f} kJ/mol; Q_Henry-(Emax-kRT) = {Q[0]-(Em-k*RT):+.4f}")
    s = PowerLaw(0.5, 12.0); lnz = np.linspace(-60/RT, -25/RT, 6); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T)
    print("power law xi=0.5: Q - a_rec/xi =", np.round(Q - (RT/m)/0.5, 3))
    s = Stretched(1.0, 6.0); lnz = np.linspace(-50/RT, -25/RT, 4); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T); Ec = -RT*lnz
    b = RT/6.0; print("exponential a=6: m =", np.round(m, 4), "beta =", round(b, 4), "; Q - Ec =", np.round(Q - Ec, 3), "; predicted -RT(1/b - pi cot(pi b)) =", round(-RT*(1/b - np.pi/np.tan(np.pi*b)), 3))
