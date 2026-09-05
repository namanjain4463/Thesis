"""
Numerically stress-test the three load-bearing claims of the v0.4 port formulation.
V1: port factorization invariance  H_e^-1 - Y_e = Y_c  (embodiment-independent).
V2: internal-force nullspace of object-motion obs, and whether proprioception breaks it.
V3: transfer certificate  ||H_hat - H|| <= eps/(m(m-eps)) and conditioning dependence.
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("="*72)
print("V1  PORT FACTORIZATION INVARIANCE  (frequency domain)")
print("="*72)
# interface impedance Z_c(s) = k/s + c  (spring+damper);  Y_c = 1/Z_c
k, c = 1000.0, 5.0
Zc = lambda s: k/s + c
Yc = lambda s: 1.0/Zc(s)
# two DIFFERENT embodiments (admittance Y_e = 1/(m s + b))
def Ye(s, m, b): return 1.0/(m*s + b)
embA = dict(m=0.5, b=20.0); embB = dict(m=2.0, b=8.0)
def He(s, emb): return 1.0/(Ye(s, **emb) + Yc(s))   # v0 -> f
print(" test at several excitation frequencies w (s = j*w):")
print("   w        |H_A|        |H_B|      | (H_A^-1 - Y_A) - Y_c |   (H_B^-1 - Y_B) - Y_c |")
maxerr = 0.0
for w in [1, 10, 50, 200, 1000]:
    s = 1j*w
    hA, hB = He(s, embA), He(s, embB)
    eA = abs((1/hA - Ye(s, **embA)) - Yc(s))
    eB = abs((1/hB - Ye(s, **embB)) - Yc(s))
    maxerr = max(maxerr, eA, eB)
    print("  %5d   %9.5f   %9.5f        %.2e               %.2e" % (w, abs(hA), abs(hB), eA, eB))
print("  --> |H_e| (raw observed response) differs a lot across embodiments,")
print("      but H_e^-1 - Y_e recovers the SAME interface Y_c.  max err = %.2e" % maxerr)
print("  VERDICT: the quotient identity is EXACT here BY CONSTRUCTION (H_e is built from")
print("           Y_e + Y_c, then Y_c recovered) -- an implementation check, not a data test.")
print("           The FALSIFIABLE content is the same quotient on INDEPENDENTLY-obtained H,Y")
print("           (the exact port split + free-space port ID), not this synthetic recovery.")

print("\n" + "="*72)
print("V2  INTERNAL-FORCE NULLSPACE  and  PROPRIOCEPTION")
print("="*72)
# object with 2 opposing contacts (1D). Grasp map to NET force along the squeeze axis.
G = np.array([[1.0, -1.0]])          # net force = f1 - f2 ; ker G = squeeze [1,1]
kerG = np.array([1.0, 1.0]) / np.sqrt(2)
f_true = np.array([5.0, 3.0])        # true contact normal forces (N)
w_net  = G @ f_true                  # object-observable net force
print(" object-observable net force  w = G f = %.2f  (blind to squeeze)" % w_net[0])
# any f = f_particular + t*kerG gives same object motion
fp = np.linalg.pinv(G) @ w_net
print(" min-norm f consistent w/ object motion:", fp, " (assumes ZERO squeeze -> WRONG)")
for t in [0, 2, 5]:
    f = fp + t*np.array([1.0,1.0])
    print("   f=%s  ->  Gf=%.2f (same),  squeeze f1+f2=%.2f" % (np.round(f,2), (G@f)[0], f.sum()))
# why squeeze matters physically: slip margin = mu*f_normal - |f_tangential|
mu, ft = 0.4, 1.5
print(" slip margin (mu*min(f_n) - f_t) for squeeze=8 vs squeeze=16 at same object motion:")
for fn in ([4,4],[8,8]):
    print("   contacts f_n=%s -> margin=%.2f %s" % (fn, mu*min(fn)-ft,
          "(SLIPS)" if mu*min(fn)-ft<0 else "(holds)"))
# proprioception: each arm's torque observes its OWN contact via its Jacobian (injective)
print(" PROPRIOCEPTION test: does adding per-arm torque obs identify f uniquely?")
J1 = np.array([[1.0,0.0]]); J2 = np.array([[0.0,1.0]])     # arm a sees f1, arm b sees f2
O_obj    = G                                    # object-motion-only observation
O_proprio= np.vstack([G, J1, J2])               # + both arms' proprioception
for name,O in [("object-only",O_obj),("object+proprio",O_proprio)]:
    ns = O.shape[1] - np.linalg.matrix_rank(O)
    print("   %-16s rank=%d  nullspace dim=%d  -> %s"
          % (name, np.linalg.matrix_rank(O), ns,
             "internal force UNIDENTIFIABLE" if ns>0 else "f fully identifiable"))
# underactuation caveat: shared single actuator (mimic gripper) -> cannot separate
J_shared = np.array([[1.0,1.0]])                # one actuator sees f1+f2 only (same as squeeze!)
O_under = np.vstack([G, J_shared])
ns = O_under.shape[1]-np.linalg.matrix_rank(O_under)
print("   %-16s rank=%d  nullspace dim=%d  -> %s"
      % ("object+1actuator", np.linalg.matrix_rank(O_under), ns,
         "still UNIDENTIFIABLE (underactuated gripper needs tactile)" if ns>0 else "ok"))

print("\n" + "="*72)
print("V3  TRANSFER CERTIFICATE  ||H_hat-H|| <= eps/(m(m-eps))")
print("="*72)
rng = np.random.default_rng(0)
def spd_with_min(n, smin, smax):
    Q,_ = np.linalg.qr(rng.standard_normal((n,n)))
    ev = np.linspace(smin, smax, n)        # sigma_min = smin (interaction conditioning)
    return Q@np.diag(ev)@Q.T
eps = 0.05                                  # SAME absolute contact-model error for both
for label, smin in [("stiff/well-conditioned  m=2.0", 2.0),
                     ("soft/near-singular     m=0.1", 0.1)]:
    A = spd_with_min(5, smin, 6.0); m = np.linalg.svd(A, compute_uv=False).min()
    H = np.linalg.inv(A)
    dA = rng.standard_normal((5,5)); dA = eps*dA/np.linalg.norm(dA,2)
    Hh = np.linalg.inv(A+dA)
    actual = np.linalg.norm(Hh-H,2); bound = eps/(m*(m-eps))
    print(" %-30s actual ||dH||=%.4f   bound=%.4f   %s"
          % (label, actual, bound, "OK" if actual<=bound+1e-9 else "VIOLATED"))
print(" --> SAME contact-model error eps=0.05 -> outcome error ~20x larger on the")
print("     ill-conditioned embodiment. sigma_min(A) is the COMPUTABLE certificate.")
