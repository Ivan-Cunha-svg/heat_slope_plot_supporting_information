#!/usr/bin/env python3
"""York regression (errors in both variables) of a_rec = RT/m on Q_st, with a parametric bootstrap for the intervals.
The local slope m is computed in apply_dunne.local_slope."""
import numpy as np
R = 8.314462618e-3

def york(x, y, sx, sy, tol=1e-12, itmax=200):
    """York et al. (2004) best straight line y = a + b x with uncorrelated errors sx, sy. Returns a, b, sa, sb."""
    x, y = np.asarray(x, float), np.asarray(y, float); wx, wy = 1.0/np.asarray(sx, float)**2, 1.0/np.asarray(sy, float)**2
    b = np.polyfit(x, y, 1)[0]
    for _ in range(itmax):
        W = wx*wy/(wx + b*b*wy); xb, yb = (W*x).sum()/W.sum(), (W*y).sum()/W.sum(); U, V = x - xb, y - yb
        beta = W*(U/wy + b*V/wx); bn = (W*beta*V).sum()/(W*beta*U).sum()
        if abs(bn - b) < tol: b = bn; break
        b = bn
    W = wx*wy/(wx + b*b*wy); xb, yb = (W*x).sum()/W.sum(), (W*y).sum()/W.sum(); a = yb - b*xb
    beta = W*((x - xb)/wy + b*(y - yb)/wx); xa = xb + beta; xab = (W*xa).sum()/W.sum(); u = xa - xab
    sb = np.sqrt(1.0/(W*u*u).sum()); sa = np.sqrt(1.0/W.sum() + xab**2*sb**2)
    return a, b, sa, sb

def ward_fit(Q, a, sQ, sa, nboot=4000, seed=0):
    """York fit of a_rec = xi (Q - E_iso). Returns xi, E_iso, k and percentile bootstrap intervals (parametric, in both variables)."""
    a0, b0, *_ = york(Q, a, sQ, sa); xi, E = b0, -a0/b0
    rng = np.random.default_rng(seed); xs = []; Es = []
    for _ in range(nboot):
        try:
            aa, bb, *_ = york(Q + sQ*rng.standard_normal(len(Q)), a + sa*rng.standard_normal(len(a)), sQ, sa)
            if np.isfinite(bb) and bb != 0: xs.append(bb); Es.append(-aa/bb)
        except Exception: pass
    return dict(xi=xi, Eiso=E, k=-1.0/xi, xi_ci=np.percentile(xs, [2.5, 97.5]), E_ci=np.percentile(Es, [2.5, 97.5]))
