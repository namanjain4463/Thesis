"""
panda_variants.py — Phase-2 SUBSTRATE CHECK for the factorized-predictor experiment.

Codex's decisive next experiment needs ARTICULATED robots whose *dynamics/port* genuinely differ,
so that a factorization (shared contact law composed with each body's ANALYTICAL port Y_G) has
something to transfer. Before building the full factorized-vs-unstructured consequence predictor,
this script VALIDATES the substrate:

  1. grasp+lift still succeeds on a real articulated arm (Panda) across arm-parameter VARIANTS;
  2. the endpoint DELASSUS PORT Y_G = J M⁻¹ Jᵀ (object row) genuinely VARIES across variants
     (a payload / damping / gain change must move the port — else the factorization is moot);
  3. arm/controller changes are SEPARABLE from gripper changes.

Variants are made by post-load edits (payload at the hand, joint damping, arm gain) — i.e. the
SAME gripper on a changed arm/controller, per "separate arm/controller changes from gripper changes."

Run:  MENAGERIE_DIR=/tmp/menagerie python panda_variants.py
"""
import os, numpy as np, mujoco
import panda_embodiment as pe
import contact_probe as cp
np.set_printoptions(precision=4, suppress=True)


def arm_dof_ids(m):
    return list(range(7))            # first 7 dofs are the arm joints


def variant_edit(payload=0.0, damp_scale=1.0, gain_scale=1.0):
    """Return a model_edit(m) that makes a Panda arm VARIANT: add a wrist payload, scale the arm
    joint damping, and scale the arm position-actuator gain (controller stiffness)."""
    def edit(m):
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if payload > 0:
            m.body_mass[hb] += payload
            m.body_inertia[hb] *= (1.0 + payload / max(m.body_mass[hb], 1e-6))
        for j in range(7):
            m.dof_damping[j] *= damp_scale
        # Panda arm actuators are the first 7; scale their position gain + matching bias.
        for a in range(min(7, m.nu)):
            m.actuator_gainprm[a, 0] *= gain_scale
            m.actuator_biasprm[a, 1] *= gain_scale
    return edit


VARIANTS = {
    "V0_base":     dict(payload=0.0,  damp_scale=1.0, gain_scale=1.0),
    "V1_payload":  dict(payload=0.6,  damp_scale=1.0, gain_scale=1.0),
    "V2_damped":   dict(payload=0.0,  damp_scale=3.0, gain_scale=1.0),
    "V3_soft":     dict(payload=0.0,  damp_scale=1.0, gain_scale=0.5),
    "V4_heavy_soft": dict(payload=0.4, damp_scale=2.0, gain_scale=0.6),
}


def port_probe(params, edit):
    """Run a grasp+lift; during the HOLD phase capture the object-directed Delassus port block
    (the 3x3 endpoint mobility seen through the arm) and the object planar slip + lift."""
    caught = {"W_hold": [], "obj0": None, "obj_last": None}

    def on_step(m, d, W, obj_bid, obj_gid, phase):
        op = np.array(d.xpos[obj_bid]).copy()
        if caught["obj0"] is None: caught["obj0"] = op
        caught["obj_last"] = op
        if phase == "hold" and W.size > 0:
            # summarize the port by its spectral scale (mean eigenvalue) — a scalar that must move
            caught["W_hold"].append(float(np.trace(W) / max(W.shape[0], 1)))

    r = pe.run_grasp(params, on_step=on_step, model_edit=edit)
    wmean = float(np.mean(caught["W_hold"])) if caught["W_hold"] else float("nan")
    slip = 0.0
    if caught["obj0"] is not None:
        d0, dl = caught["obj0"], caught["obj_last"]
        slip = float(np.hypot(dl[0] - d0[0], dl[1] - d0[1]))
    return dict(lift=r["lift"], port_scale=wmean, slip=slip)


def main():
    if not os.environ.get("MENAGERIE_DIR"):
        os.environ["MENAGERIE_DIR"] = "/tmp/menagerie"
    params = dict(pe.BASE)
    print("=" * 84)
    print("PHASE-2 SUBSTRATE CHECK — do Panda arm VARIANTS grasp+lift, and does the port VARY?")
    print("=" * 84)
    print("  %-14s %8s %14s %10s" % ("variant", "lift[m]", "port_scale", "slip[mm]"))
    rows = {}
    for name, vp in VARIANTS.items():
        res = port_probe(params, variant_edit(**vp))
        rows[name] = res
        ok = "GRASPED" if res["lift"] > 0.05 else "no-lift"
        print("  %-14s %8.3f %14.4g %10.2f   %s" % (name, res["lift"], res["port_scale"], res["slip"]*1000, ok))
    ports = np.array([rows[n]["port_scale"] for n in VARIANTS if np.isfinite(rows[n]["port_scale"])])
    lifts = np.array([rows[n]["lift"] for n in VARIANTS])
    print("\n  grasp+lift succeeded on %d/%d variants (lift>0.05m)" % (int((lifts > 0.05).sum()), len(lifts)))
    if len(ports) >= 2:
        spread = (ports.max() - ports.min()) / max(abs(ports.mean()), 1e-9)
        print("  port_scale across variants: min=%.4g max=%.4g  relative spread=%.1f%%" % (ports.min(), ports.max(), 100*spread))
        print("  -> the endpoint port %s across arm variants (%s for a factorization to matter)."
              % ("VARIES" if spread > 0.05 else "is ~CONSTANT", "required" if spread > 0.05 else "PROBLEM"))
    print("\n  VERDICT: substrate is %s for the factorized-predictor experiment."
          % ("READY" if (lifts > 0.05).sum() >= 3 and len(ports) >= 2 and (ports.max()-ports.min())/max(abs(ports.mean()),1e-9) > 0.05 else "NOT yet ready"))


if __name__ == "__main__":
    main()
