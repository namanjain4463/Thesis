"""
FREE-SPACE PORT IDENTIFICATION + the SEPARATION PRINCIPLE, and folding eps_Y into
the transfer certificate.

Motivation (the hardware gap): the certificate so far treated Y_G as known. On real
hardware Y_G is the CLOSED-LOOP endpoint admittance (arm inertia + controller +
latency) and must be IDENTIFIED, with its own error eps_Y. But identifying Y from
CONTACT data is entangled with the contact law C (the composed response
H=(Y+C^-1)^-1 has a gauge freedom: many (Y,C) give the same H).

Separation principle: in FREE space the contact force df ≡ 0, so the endpoint port
response is Y_robot ALONE — no C. Excite in free space -> Y_robot uncontaminated ->
then C is identified from contact data given Y. This makes the deconvolution well-posed.

Part A: free-space endpoint mobility ID recovers the TRUE endpoint port — including
        structural couplings (the finger tendon) that the naive rigid-body J M⁻¹Jᵀ misses.
Part B: from contact H alone, (Y,C) is a gauge family; free-space Y makes C unique.
Part C: eps_Y = free-space port-ID error vs sensor noise (the identification budget).
Part D: two-source certificate ||ΔH|| <= (eps_Y + eps_C)/(m(m-(eps_Y+eps_C))).
"""
import numpy as np, mujoco
import contact_probe as cp
import m2_floating_gripper_grasp as Mfg
import panda_embodiment as P
np.set_printoptions(precision=5, suppress=True)


def point_jacobian(m, d, body_id, point):
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, point, body_id)
    return jp


def freespace_mobility(m, d, body_id, point, noise=0.0, rng=None):
    """Identify the endpoint mobility Y_e (3x3): apply a unit force at `point` along
    each axis (in FREE space) and read the endpoint acceleration response
    a_p = Y_e f. Bias (gravity) removed by differencing against the zero-force state."""
    d.qvel[:] = 0.0; d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    Jp = point_jacobian(m, d, body_id, point)
    a0 = Jp @ np.array(d.qacc)
    Ye = np.zeros((3, 3))
    for i in range(3):
        f = np.zeros(3); f[i] = 1.0
        d.qfrc_applied[:] = Jp.T @ f
        mujoco.mj_forward(m, d)
        ai = Jp @ np.array(d.qacc)
        col = ai - a0
        if noise > 0 and rng is not None:
            col = col + noise*np.linalg.norm(col)*rng.standard_normal(3)   # measurement noise
        Ye[:, i] = col
        d.qfrc_applied[:] = 0.0
    return 0.5*(Ye+Ye.T)                                     # symmetrize (mobility is SPD)


def analytical_mobility(m, d, body_id, point):
    """Naive rigid-body endpoint mobility J M^-1 J^T (IGNORES active couplings)."""
    Jp = point_jacobian(m, d, body_id, point)
    Minv = np.linalg.inv(cp.full_M(m, d))
    return Jp @ Minv @ Jp.T

def constrained_mobility(m, d, body_id, point):
    """True endpoint mobility under the robot's ACTIVE structural constraints
    (e.g. the finger tendon): project M^-1 with the constraint Jacobian efc_J.
    This is what the contact actually feels, and what free-space ID must match."""
    Jp = point_jacobian(m, d, body_id, point)
    Minv = np.linalg.inv(cp.full_M(m, d))
    nefc = int(d.nefc)
    if nefc == 0:
        return Jp @ Minv @ Jp.T
    Jc = np.array(d.efc_J).reshape(nefc, m.nv)
    S = Jc @ Minv @ Jc.T
    Mc = Minv - Minv @ Jc.T @ np.linalg.solve(S + 1e-12*np.eye(nefc), Jc @ Minv)  # constrained inv-inertia
    return Jp @ Mc @ Jp.T


# ---------------- floating gripper, free space (at home, fingers far from object) ----------------
def floating_freespace():
    m = mujoco.MjModel.from_xml_string(Mfg.scene_xml(0.05,0.08,0.05,1.0,20.0))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)     # home: palm high, no contact
    fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lfinger")
    pt = np.array(d.xpos[fb]) + np.array([0, 0, -0.03])    # a fingertip point
    return m, d, fb, pt

def panda_freespace():
    # bare arm, NO object/pedestal -> the endpoint is genuinely free (no active
    # constraints). This is the physical "calibrate the arm in open space" step.
    m = mujoco.MjModel.from_xml_path(P.PANDA_XML)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    for nm in ("finger_joint1", "finger_joint2"):          # move fingers OFF their 0.04 limit
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        d.qpos[m.jnt_qposadr[jid]] = 0.02
    d.ctrl[:] = 0.0; mujoco.mj_forward(m, d)
    fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    pt = np.array(d.xpos[fb]) + np.array([0, 0, 0.02])
    return m, d, fb, pt


print("="*70); print("PART A  FREE-SPACE endpoint mobility ID  vs the TRUE (constrained) port"); print("="*70)
for name, setup in [("FLOATING", floating_freespace), ("PANDA", panda_freespace)]:
    m, d, fb, pt = setup()
    Ye_id = freespace_mobility(m, d, fb, pt)
    Ye_true = constrained_mobility(m, d, fb, pt)             # true port (accounts for tendon etc.)
    Ye_naive = analytical_mobility(m, d, fb, pt)             # rigid-body J M^-1 J^T (ignores couplings)
    rel_true  = np.linalg.norm(Ye_id-Ye_true)/max(np.linalg.norm(Ye_true),1e-12)
    rel_naive = np.linalg.norm(Ye_true-Ye_naive)/max(np.linalg.norm(Ye_true),1e-12)
    print(" %-9s nefc=%d  ID vs TRUE port=%.2e   |  naive rigid-body J M⁻¹Jᵀ misses port by %.1f%%"
          % (name, int(d.nefc), rel_true, 100*rel_naive))
print(" => free-space excitation recovers the TRUE endpoint port (incl. structural couplings")
print("    like the finger tendon) that the naive analytical model can miss. You IDENTIFY Y, not assume it.\n")

# ======================================================================
# PART B  SEPARATION PRINCIPLE: from contact data alone (Y,C) is a GAUGE family;
#          free-space Y breaks the gauge -> unique C.
# ======================================================================
print("="*70); print("PART B  Y/C non-identifiability from contact H; free-space Y resolves it"); print("="*70)
rng = np.random.default_rng(0)
def spd(n, lo, hi):
    Q,_ = np.linalg.qr(rng.standard_normal((n,n))); return Q@np.diag(np.linspace(lo,hi,n))@Q.T
Y_true = spd(3, 5.0, 30.0)                     # true robot port (what free space would give)
Cinv_true = spd(3, 200.0, 800.0)               # true contact compliance^-1 (stiff)
H = np.linalg.inv(Y_true + Cinv_true)          # composed interface response (the only contact-observable)
print(" observed H (force->rel.vel) fixes only  Y + C⁻¹ = H⁻¹  (3 eqns, 6 unknown matrices).")
print("  gauge family reproducing the SAME H (contact data cannot tell these apart):")
for scale in [0.5, 1.0, 1.5]:
    Yp = scale*Y_true                          # a WRONG guess of the robot port
    Cinv_p = np.linalg.inv(H) - Yp             # the C that makes H come out identical
    Hp = np.linalg.inv(Yp + Cinv_p)
    print("   Y'=%.1f·Y_true -> ||H'−H||=%.1e   (implied C differs, but H identical => C UNIDENTIFIABLE)"
          % (scale, np.linalg.norm(Hp-H)))
Cinv_from_freespace = np.linalg.inv(H) - Y_true
print(" free-space Y=Y_true -> C⁻¹ = H⁻¹ − Y : recovered ||C⁻¹−C⁻¹_true||=%.1e  (UNIQUE)."
      % np.linalg.norm(Cinv_from_freespace - Cinv_true))
print(" => without free-space, embodiment vs contact-law is a gauge freedom; free-space fixes it.\n")

# ======================================================================
# PART C  eps_Y: free-space identification error under measurement noise
# ======================================================================
print("="*70); print("PART C  eps_Y = free-space port-ID error vs measurement noise"); print("="*70)
m, d, fb, pt = floating_freespace()
Ye_true = constrained_mobility(m, d, fb, pt)
print("  noise    eps_Y=||Ŷ−Y||/||Y||   (median of 20 trials)")
eps_by_noise = {}
for noise in [0.0, 0.01, 0.03, 0.1]:
    errs = []
    for t in range(20):
        Ye_hat = freespace_mobility(m, d, fb, pt, noise=noise, rng=np.random.default_rng(t))
        errs.append(np.linalg.norm(Ye_hat - Ye_true)/np.linalg.norm(Ye_true))
    eps_by_noise[noise] = float(np.median(errs))
    print("  %5.0f%%   %.3e" % (100*noise, eps_by_noise[noise]))
print("  => eps_Y ~ linear in sensor noise; this is the identification budget the certificate must carry.\n")

# ======================================================================
# PART D  TWO-SOURCE certificate:  ||ΔH|| <= (eps_Y + eps_C)/(m(m-(eps_Y+eps_C)))
# ======================================================================
print("="*70); print("PART D  two-source transfer certificate (embodiment eps_Y + contact-law eps_C)"); print("="*70)
A0 = Y_true + Cinv_true; H0 = np.linalg.inv(A0); mmin = np.linalg.svd(A0, compute_uv=False).min()
print("  m = σ_min(Y+C⁻¹) = %.1f" % mmin)
print("  eps_Y  eps_C |  actual ||ΔH||   bound=(eps_Y+eps_C)/(m(m−Σ))   holds")
for eY, eC in [(0.0,2.0),(2.0,0.0),(2.0,2.0),(5.0,3.0)]:
    dY = spd(3,-1,1); dY = eY*dY/np.linalg.norm(dY,2)          # ||dY||=eps_Y
    dC = spd(3,-1,1); dC = eC*dC/np.linalg.norm(dC,2)          # ||dC||=eps_C
    Hh = np.linalg.inv(A0 + dY + dC)
    actual = np.linalg.norm(Hh-H0,2); tot = eY+eC
    bound = tot/(mmin*(mmin-tot)) if mmin>tot else float('inf')
    print("  %4.1f  %4.1f  |  %.4e     %.4e            %s"
          % (eY, eC, actual, bound, "OK" if actual<=bound+1e-9 else "VIOLATED"))
print("  => the two error sources ADD inside one certificate. On hardware eps_Y (port ID) and")
print("     eps_C (sim-to-real contact law) are BOTH budgeted; a stiff/well-conditioned interface")
print("     (large m) tolerates both, an ill-conditioned one amplifies both. This is the honest cert.")

# ---------------- figure ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ns = sorted(eps_by_noise); ax[0].plot([100*n for n in ns], [eps_by_noise[n] for n in ns], "o-", color="#06c")
ax[0].set_xlabel("endpoint acceleration sensor noise [%]"); ax[0].set_ylabel("ε_Y = ‖Ŷ−Y‖/‖Y‖")
ax[0].set_title("Part C: free-space port-ID error ε_Y\n(the embodiment identification budget)")
# certificate: actual vs bound as total error grows, for two conditionings
for mm2, lab, c in [(215.9,"stiff  m=216","#2a8"), (40.0,"soft  m=40","#c40")]:
    tots=np.linspace(0.1, mm2*0.6, 30); bnd=tots/(mm2*(mm2-tots))
    ax[1].plot(tots, bnd, "-", color=c, label="bound  "+lab)
ax[1].set_xlabel("total error  ε_Y + ε_C"); ax[1].set_ylabel("transfer error bound  ‖ΔH‖")
ax[1].set_title("Part D: two-source certificate\nε_Y and ε_C add; conditioning m sets amplification")
ax[1].legend(fontsize=8); ax[1].set_yscale("log")
plt.tight_layout(); plt.savefig("port_identification.png", dpi=95); print("\nwrote port_identification.png")
