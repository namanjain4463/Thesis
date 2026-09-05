"""Confirm the frictionless convex-contact KKT  (W+R) f = aref - a_u  (a_u = J*qacc_smooth)
is machine-exact for a single FRICTIONLESS box, at a SETTLED and a TRANSIENT state. If so,
the earlier ~1.3 residual was purely the friction-cone BOUNDARY term (contacts at the cone
edge), not a modeling error. NOTE: the settled state alone cannot validate the sign — the
wrong form (R-W)f = a_u+aref has residual -2*a_c, which vanishes at rest (that trap masked a
sign error in an earlier P2)."""
import numpy as np, mujoco, contact_probe as cp
np.set_printoptions(precision=6, suppress=True, linewidth=140)

XML = """
<mujoco>
  <option gravity="0 0 -9.81" cone="elliptic" jacobian="dense" solver="Newton"
          iterations="500" tolerance="1e-14" ls_iterations="100"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" friction="0 0 0"/>
    <body name="obj" pos="0 0 0.05">
      <freejoint/>
      <geom name="box" type="box" size="0.05 0.05 0.05" mass="0.2" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)

def kkt(d):
    """Frictionless active-constraint KKT residual for (W+R)f = aref - a_u, plus the
    OLD wrong-sign residual (R-W)f - (a_u+aref) [= -2*a_c] for contrast. Returns
    (correct_rel, old_rel, ||a_c||, f)."""
    mujoco.mj_forward(m, d)
    nefc, nv = int(d.nefc), m.nv
    J = np.array(d.efc_J).reshape(nefc, nv)
    W = J @ cp.Minv_apply(m, d, J).T
    R = np.array(d.efc_R[:nefc]); aref = np.array(d.efc_aref[:nefc]); f = np.array(d.efc_force[:nefc])
    a_u = J @ np.array(d.qacc_smooth); a_c = J @ np.array(d.qacc)
    act = f > 1e-9
    rhs = aref - a_u
    correct = np.linalg.norm(((W + np.diag(R)) @ f - rhs)[act]) / max(np.linalg.norm(rhs[act]), 1e-12)
    old = np.linalg.norm(((np.diag(R) - W) @ f - (a_u + aref))[act]) / max(np.linalg.norm((a_u + aref)[act]), 1e-12)
    return correct, old, float(np.linalg.norm(a_c[act])), f

d = mujoco.MjData(m)
for _ in range(2000): mujoco.mj_step(m, d)          # settle to rest (a_c ~ 0)
res_s, old_s, ac_s, f = kkt(d)
print("nefc=%d  ncon=%d" % (int(d.nefc), int(d.ncon)))
print("SETTLED  (||a_c||=%.2e):  correct (W+R)f=aref-a_u rel=%.2e   OLD (R-W)f=a_u+aref rel=%.2e" % (ac_s, res_s, old_s))
print("  sum normal force = %.5f N   weight = %.5f N" % (f.sum(), 0.2*9.81))
# TRANSIENT: push the box into the plane so the contact ACCELERATES (a_c != 0) -> this is
# where the OLD sign FAILS (residual -2*a_c) while the correct (W+R) form stays machine-zero.
d.qfrc_applied[:] = 0.0; d.qfrc_applied[2] = -30.0
res_t, old_t, ac_t, _ = kkt(d)
d.qfrc_applied[:] = 0.0
print("TRANSIENT (||a_c||=%.2e):  correct (W+R)f=aref-a_u rel=%.2e   OLD (R-W)f=a_u+aref rel=%.2e" % (ac_t, res_t, old_t))
ok = res_s < 1e-6 and res_t < 1e-6
print("\n VERDICT: %s" % ("PASS — the frictionless KKT (W+R)f = aref - a_u is EXACT at BOTH a settled "
      "and a transient state; friction contacts add only the cone-boundary term (handled by the "
      "SOLVE). The OLD (R-W) form passes ONLY at rest (residual -2*a_c), so a settled-only check "
      "cannot see the sign error." if ok else "CHECK — KKT residual not machine-zero."))
