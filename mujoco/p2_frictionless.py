"""Confirm the convex-contact KKT is exactly  jar + R*f = 0  in the case with no
friction cone to sit on: a single FRICTIONLESS box resting on a plane. If this is
~0, the earlier ~1.3 residual was purely the friction-cone BOUNDARY term (contacts
sliding/at the cone edge), i.e. expected, not a modeling error."""
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
d = mujoco.MjData(m)
for _ in range(2000): mujoco.mj_step(m, d)      # settle to rest
mujoco.mj_forward(m, d)
nefc, nv = int(d.nefc), m.nv
J = np.array(d.efc_J).reshape(nefc, nv)
W = J @ cp.Minv_apply(m, d, J).T
R = np.array(d.efc_R[:nefc]); aref = np.array(d.efc_aref[:nefc]); f = np.array(d.efc_force[:nefc])
qacc = np.array(d.qacc); qacc_s = np.array(d.qacc_smooth)
jar = J @ qacc + aref
q_s = J @ qacc_s + aref
print("nefc=%d (frictionless: normal rows only)  ncon=%d" % (nefc, int(d.ncon)))
print(" f      :", f)
print(" jar    :", jar)
print(" R*f    :", R*f)
print(" jar - R*f:", jar - R*f, "   rel=%.3e" % (np.linalg.norm(jar - R*f)/max(np.linalg.norm(jar),1e-12)))
print(" P1  J(qacc-qacc_s) - W f  rel=%.3e" % (np.linalg.norm(J@(qacc-qacc_s)-W@f)/max(np.linalg.norm(W@f),1e-12)))
print(" sum normal force = %.5f N   weight = %.5f N" % (f.sum(), 0.2*9.81))
# --- identify the exact PD contact system on the ACTIVE rows ---
act = f > 1e-9
Wa = W[np.ix_(act,act)]; Ra = np.diag(R[act]); fa = f[act]; qa = q_s[act]
print("\n active rows:", int(act.sum()))
print(" diag(W_active):", np.diag(Wa), "   R_active:", R[act])
for name, lhs in [("(W+R) f + q_s", (Wa+Ra)@fa + qa),
                  ("(W+R) f - q_s", (Wa+Ra)@fa - qa),
                  ("(R-W) f - q_s", (Ra-Wa)@fa - qa),
                  ("(R-W) f + q_s", (Ra-Wa)@fa + qa)]:
    print("   %-16s ||.||=%.3e" % (name, np.linalg.norm(lhs)))
ok = np.linalg.norm(jar - R*f)/max(np.linalg.norm(jar),1e-12) < 1e-6
print("\n VERDICT: %s" % ("PASS — convex-KKT  jar = R·f  EXACT with no friction cone. "
      "The line above with ||.||~0 is MuJoCo's exact PD contact system; friction "
      "contacts add only the cone-boundary term (handled by the SOLVE)." if ok else "CHECK"))
