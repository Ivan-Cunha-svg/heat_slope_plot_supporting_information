#!/usr/bin/env python3
"""Numerical checks of the identities and numbers quoted in the text. Each line prints OK/FAIL.
Exact adsorption integral with a Langmuir kernel (model.py); P0 constant unless stated, so that Q_z = Q_st."""
import numpy as np
from scipy.special import gammainc, gamma as Gamma
from scipy.optimize import brentq
from model import Edge, Stretched, PowerLaw, SingleSite, slope_m, isosteric_heat, R, _F, _xg, _wg
ok_all = True
def check(name, cond, info=""):
    global ok_all; ok_all &= bool(cond); print(f"  {'OK  ' if cond else 'FAIL'} {name}  {info}")
T = 300.0; RT = R*T
EDGES = ((1, 45.0, 20.0), (2, 50.0, 22.0), (3, 55.0, 25.0))

print("1. Covariance (eq 17): theta[G_{lambda,c}](Ec;T) = theta[G]((Ec-c)/lambda; T/lambda)")
for lam, c in ((1.3, 0.0), (1.0, 7.0), (0.8, -5.0)):
    s0 = Edge(2, 50.0, 22.0); s1 = Edge(2, lam*50.0 + c, lam*22.0); Ec = np.linspace(30, 60, 7)
    lhs = s1.theta(-Ec/RT, T); rhs = s0.theta(-((Ec - c)/lam)/(R*T/lam), T/lam)
    check(f"lambda={lam}, c={c}", np.allclose(lhs, rhs, rtol=1e-10), f"max rel dev {np.max(abs(lhs/rhs-1)):.1e}")

print("2. Linear law (eq 22/24) for pure edges, residual Q_z + k RT/m - Emax (kJ/mol)")
for k, Em, W in EDGES:
    s = Edge(k, Em, W); lnz = np.linspace(-Em/RT - 8, -Em/RT + 9, 35); th = s.theta(lnz, T); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T)
    res = Q + k*RT/m - Em; r1 = res[th <= 1e-3].max(); r2 = res[th <= 0.25].max()
    pred = RT*(W/RT)**k*np.exp(-W/RT)/(Gamma(k)*gammainc(k, W/RT))          # Henry-regime residual from eq 26: kRT[1 - P(k+1,w)/P(k,w)]
    check(f"k={k}: residual for theta<=1e-3 = {r1:+.3f} (eq 26 predicts {pred:+.3f}); max for theta<=0.25 = {r2:+.3f}", abs(r1 - pred) < 3e-3 and r2 < (0.04, 0.4, 0.9)[k-1])

print("3. Henry heat (eq 26): q_H = Emax - kRT P(k+1,w)/P(k,w)")
for k, Em, W in EDGES:
    s = Edge(k, Em, W); Q = isosteric_heat(s, np.array([-Em/RT - 12.0]), T)[0]; w = W/RT
    pred = Em - k*RT*gammainc(k + 1, w)/gammainc(k, w)
    check(f"k={k}: numeric {Q:.4f}, formula {pred:.4f}, Emax-kRT = {Em-k*RT:.4f}", abs(Q - pred) < 2e-3)

print("4. Polylogarithm (eq 27) and thermal collapse")
import mpmath as mp
for k in (1, 2, 3):
    sg = np.linspace(0, 90, 360001)
    for x in (-5.0, 0.0, 6.0):
        fd = k*np.trapezoid(sg**(k - 1)/(1 + np.exp(np.clip(sg - x, -700, 700))), sg) if k > 1 else np.log1p(np.exp(x))
        li = float(-Gamma(k + 1)*mp.polylog(k, -mp.e**x))
        check(f"k={k}, x={x}: k int s^(k-1)/(1+e^(s-x)) ds = {fd:.6f}, -Gamma(k+1) Li_k(-e^x) = {li:.6f}", abs(fd/li - 1) < 1e-5)
errs = []
for k, Em, W in EDGES:
    s = Edge(k, Em, W)
    for x in (-6.0, -2.0, 0.0, 2.0, 4.0):
        th = s.theta(np.array([x - Em/RT]), T)[0]; cf = float(-Gamma(k + 1)*(RT/W)**k*mp.polylog(k, -mp.e**x)); errs.append((k, x, abs(cf/th - 1)))
e_h = max(e for k, x, e in errs if x <= -2.0); e_w = max(e for k, x, e in errs)
print("     relative error of eq 27 against the exact integral with finite W, by k and x:", [(k, x, f"{e:.1e}") for k, x, e in errs])
check("eq 27 within 3e-3 of the finite-W integral in the Henry regime and within 1e-2 for x <= 4", e_h < 3e-3 and e_w < 1e-2, f"max {e_h:.1e} (Henry), {e_w:.1e} (x <= 4)")
s = Edge(2, 50.0, 40.0); x = np.linspace(-6, 10, 9); col = [s.theta(x - 50.0/(R*TT), TT)*(40.0/(R*TT))**2 for TT in (270.0, 300.0, 330.0)]
check("collapse of theta (W/RT)^k at 270/300/330 K (k=2, W=40)", np.max(np.abs(np.array(col)/col[1] - 1)) < 1e-2, f"max rel spread {np.max(np.abs(np.array(col)/col[1]-1)):.1e}")

print("5. Translation identity (eqs 18-19): exponential tail, a = 6 kJ/mol")
s = Stretched(1.0, 6.0); lnz = np.array([-50/RT, -40/RT]); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T); b = RT/6.0
check(f"m = beta: {m[0]:.4f} vs {b:.4f}", abs(m[0] - b) < 1e-3)
check(f"Q_z - Ec = {Q[0]+RT*lnz[0]:.3f}, -RT(1/beta - pi cot(pi beta)) = {-RT*(1/b-np.pi/np.tan(np.pi*b)):.3f}", abs(Q[0] + RT*lnz[0] + RT*(1/b - np.pi/np.tan(np.pi*b))) < 5e-3)
th = s.theta(lnz, T); check(f"theta/G(Ec) = {th[0]/np.exp(lnz[0]*b):.4f} = pi beta/sin(pi beta) = {np.pi*b/np.sin(np.pi*b):.4f}", abs(th[0]/np.exp(lnz[0]*b)/(np.pi*b/np.sin(np.pi*b)) - 1) < 1e-3)

print("6. Dilation identity for a power-law tail (Frechet), G ~ E^-2 above 12 kJ/mol: Q_z - a_rec/xi")
s = PowerLaw(0.5, 12.0); Ec = np.linspace(35, 60, 6); m = slope_m(s, -Ec/RT, T); Q = isosteric_heat(s, -Ec/RT, T); d = Q - (RT/m)/0.5
check(f"|Q_z - a_rec/xi| for Ec >= 35: max {np.abs(d).max():.3f} kJ/mol", np.abs(d).max() < 0.04)
check(f"CA asymptotically exact: theta/G(Ec) at Ec=60: {s.theta(np.array([-60/RT]), T)[0]/(60/12.0)**-2:.4f}; a_rec/a(Ec) = {(RT/m[-1])/(0.5*60):.4f}", abs((RT/m[-1])/(0.5*60) - 1) < 0.02)

print("7. Thermal index (eqs 28-30) from the exact isotherms, tau = dln m/dln T at fixed z")
def tau(s, lnz, T, dT=1.0): return (np.log(slope_m(s, lnz, T + dT)) - np.log(slope_m(s, lnz, T - dT)))/(np.log(T + dT) - np.log(T - dT))
def dlnm_dlnz(s, lnz, T, h=1e-3): return (np.log(slope_m(s, lnz + h, T)) - np.log(slope_m(s, lnz - h, T)))/(2*h)
for k, Em in ((1, 45.0), (2, 50.0), (3, 55.0)):
    s = Edge(k, Em, 40.0); Ec = np.array([Em + 8, Em, Em - 6, Em - 12, Em - 16]); t = tau(s, -Ec/RT, T); ex = -(Em/RT)*dlnm_dlnz(s, -Ec/RT, T)
    check(f"edge k={k}: tau = {np.round(t, 2)} at Ec - Emax = +8, 0, -6, -12, -16; exact relation tau = -(Emax/RT) dln m/dln z: max dev {np.abs(t-ex).max():.1e}; CA value at -12: {Em/12:.2f}", np.abs(t - ex).max() < 5e-3)
for rho, E0 in ((1.5, 16.0), (2.0, 22.0)):
    s = Stretched(rho, E0); t = tau(s, -np.array([35.0, 45.0])/RT, T); check(f"stretched rho={rho}: tau = {np.round(t, 2)} (CA value {rho})", np.all(abs(t - rho) < 0.45))
s = PowerLaw(0.5, 12.0); t = tau(s, -np.array([45.0, 55.0])/RT, T); check(f"power law: tau = {np.round(t, 3)} (CA value 0)", np.all(abs(t) < 0.1))
s = Stretched(1.0, 6.0); t = tau(s, -np.array([40.0, 50.0])/RT, T); check(f"exponential: tau = {np.round(t, 3)} (value 1)", np.all(abs(t - 1) < 0.02))

print("8. Mean-field lateral interactions (eq 31): Q_z + k RT/m = Emax + (1-k) omega theta")
class Lateral:
    def __init__(self, base, omega): self.b, self.om = base, omega
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); return np.array([brentq(lambda th: self.b.theta(np.array([l + self.om*th/(R*T)]), T)[0] - th, 0.0, 1.0, xtol=1e-15) for l in lnz])
for k, Em, W in ((2, 50.0, 22.0), (1, 45.0, 20.0), (3, 55.0, 25.0)):
    s = Lateral(Edge(k, Em, W), 4.0); lnz = np.linspace(-Em/RT - 3, -Em/RT + 5.5, 6); th = s.theta(lnz, T); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T)
    res = Q + k*RT/m - Em - (1 - k)*4.0*th; base = Edge(k, Em, W); lz0 = lnz + 4.0*th/RT
    res0 = isosteric_heat(base, lz0, T) + k*RT/slope_m(base, lz0, T) - Em          # residual of the non-interacting edge at the same effective activity (finite W)
    check(f"k={k}, omega=4: theta up to {th.max():.3f}; residual minus that of the edge without interaction: max {np.abs(res-res0).max():.1e} kJ/mol", np.abs(res - res0).max() < 5e-3)

print("9. Linear enthalpy-entropy compensation: K(E) ~ exp(E/RT + g E); Q_z = Emax - k RT'/m, RT' = RT/(1 + g RT)")
class Comp:
    def __init__(self, k, Emax, W, g): self.k, self.Emax, self.W, self.g = k, Emax, W, g
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); u = 0.5*(_xg + 1.0); w = 0.5*_wg; E = self.Emax - self.W*u**(1.0/self.k)
        return (_F(lnz[:, None] + E[None, :]*(1/(R*T) + self.g))*w[None, :]).sum(axis=1)
g = 0.05; s = Comp(2, 50.0, 40.0, g); RTp = RT/(1 + g*RT); lnz = np.linspace(-50/RTp - 4, -50/RTp + 4, 5); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T)
check(f"g = {g}/kJ: Q_z + k RT'/m - Emax: max {np.abs(Q + 2*RTp/m - 50).max():.3f} kJ/mol; apparent exponent k/(1+gRT) = {2/(1+g*RT):.3f}", np.abs(Q + 2*RTp/m - 50).max() < 0.01)

print("10. Temperature-dependent end point and width: Q_z = [Ed - T dEd/dT] - k (1 - varpi) RT/m")
class Thermal:
    def __init__(self, k, E0, dEdT, W0, varpi): self.k, self.E0, self.dEdT, self.W0, self.vp = k, E0, dEdT, W0, varpi
    def theta(self, lnz, T):
        Em = self.E0 + self.dEdT*(T - 300.0); W = self.W0*(T/300.0)**self.vp; return Edge(self.k, Em, W).theta(lnz, T)
s = Thermal(2, 50.0, -0.02, 40.0, 0.3); lnz = np.linspace(-50/RT - 4, -50/RT + 4, 5); m = slope_m(s, lnz, T); Q = isosteric_heat(s, lnz, T, dT=0.25)
pred = (50.0 + 0.02*300.0) - 2*(1 - 0.3)*RT/m
check(f"dEd/dT = -0.02 kJ/mol/K, varpi = 0.3: max |Q_z - prediction| = {np.abs(Q - pred).max():.3f} kJ/mol", np.abs(Q - pred).max() < 0.01)

print("11. Asymptotically self-similar tails (Appendix C)")
class GammaTail:           # -G'(E) = E^(p-1) e^(-E/a)/(Gamma(p) a^p): exponential tail times a slowly varying factor
    def __init__(self, p, a): self.p, self.a = p, a
    def theta(self, lnz, T):
        lnz = np.atleast_1d(lnz); out = np.zeros(len(lnz)); ed = np.linspace(0, 80*self.a, 41)
        for lo, hi in zip(ed[:-1], ed[1:]):
            E = 0.5*(hi - lo)*_xg + 0.5*(hi + lo); w = 0.5*(hi - lo)*_wg
            out += (_F(lnz[:, None] + E[None, :]/(R*T))*(E**(self.p - 1)*np.exp(-E/self.a)/(Gamma(self.p)*self.a**self.p)*w)[None, :]).sum(axis=1)
        return out
    def G(self, E): from scipy.special import gammaincc; return gammaincc(self.p, E/self.a)
s = GammaTail(3.0, 6.0); b = RT/6.0; KF = np.pi*b/np.sin(np.pi*b)
dev = []
for Ec in (60.0, 120.0, 240.0):
    r = s.theta(np.array([-Ec/RT]), T)[0]/s.G(Ec); mm = slope_m(s, np.array([-Ec/RT]), T)[0]; dev.append((1 - r/KF, 1 - mm/b))
    print(f"     gamma tail (p=3, a=6): Ec={Ec:5.0f}: theta/[K_F G(Ec)] = {r/KF:.4f}, m/beta = {mm/b:.4f}")
dev = np.array(dev)
check("case (i): theta/[K_F(beta) G(Ec)] -> 1 and m -> beta, with deviations that halve when Ec doubles (slowly varying factor E^2)", np.all(dev[-1] < 0.06) and np.all(dev[:-1]/dev[1:] > 1.7))
s = PowerLaw(0.5, 12.0); r = [s.theta(np.array([-Ec/RT]), T)[0]/(Ec/12.0)**-2 for Ec in (40.0, 80.0, 160.0)]
check(f"case (ii): theta/G(Ec) -> 1 for the power law: {np.round(r, 4)}", abs(r[-1] - 1) < 5e-3 and abs(r[-1] - 1) < abs(r[0] - 1))
print("\nALL OK" if ok_all else "\nSOME CHECKS FAILED")
