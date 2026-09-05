"""
m2_floating_gripper_grasp.py  —  M2 contact-transition data generator (v0).

FIRST embodiment: a FLOATING parallel gripper (3 Cartesian slide DOFs + 2
fingers) grasping a cylinder on a pedestal.  Rationale: this isolates the LOCAL
contact law (finger<->object) from arm kinematics — no IK, clean contact data —
which is exactly what the Factorized Interaction World Model learns first.
Arms (Panda / Lynxmotion) are added later as extra embodiments for the
cross-embodiment transfer study; the logging below is embodiment-agnostic.

Per trial it scripts open -> descend -> close -> lift and, every step of the
contact phase, logs the OBSERVABLE object motion (Δp, ΔR, v, ω) together with
the analytical global coupling W = J_c M^-1 J_c^T and per-contact
points/normals/forces, via the validated `contact_probe` module.

Outputs (under OUT_DIR):
  * trials.csv                 — one summary row per trial (harness-style)
  * traj_<tag>_<seed>.npz      — per-step arrays for that trial (the dataset)

Run:  python m2_floating_gripper_grasp.py
Backend: classic MuJoCo (CPU) — reference-exact W, ideal for validating the
pipeline. Swap to mujoco_warp for GPU scale once this is trusted (contact_probe
keeps the engine calls in one place).
"""

import os
import numpy as np
import mujoco
import contact_probe as cp

OUT_DIR = "m2_out"
CONTROL_HZ = 500                 # matches timestep 0.002
GATE_RISE = 0.015                # m, "lifted" threshold (same gate as the Isaac harness)
PALM_HOME = 0.6                  # world z of the palm at pz=0 (starts clear above the scene);
                                 # ctrl z-targets are WORLD z and get PALM_HOME subtracted.


# ----------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------
def scene_xml(diam, height, mass, mu, squeeze_force, solref_t=0.01):
    """Floating parallel gripper + cylinder on a pedestal. `squeeze_force` caps
    the finger actuator force (the slip knob). Object friction = mu. `solref_t` is
    the OBJECT-geom normal solref time-constant (material normal stiffness; smaller
    = stiffer). Default 0.01 reproduces the original scene exactly."""
    r = diam / 2.0
    ped_top = 0.20
    obj_z = ped_top + height / 2.0 + 0.001
    return f"""
<mujoco model="floating_gripper_grasp">
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <geom solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1" contype="1" conaffinity="1"/>
    <geom name="pedestal" type="box" pos="0 0 0.1" size="0.08 0.08 0.1"
          contype="1" conaffinity="1" friction="{mu} 0.01 0.001"/>

    <body name="palm" pos="0 0 {PALM_HOME}">
      <joint name="px" type="slide" axis="1 0 0"/>
      <joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.05 0.03 0.02" mass="0.3" contype="2" conaffinity="0"/>
      <body name="lfinger" pos="0.04 0 -0.05">
        <joint name="lf" type="slide" axis="-1 0 0" range="0 0.035"/>
        <geom name="lfinger" type="box" size="0.006 0.012 0.03" mass="0.03"
              condim="3" friction="{mu} 0.02 0.002" contype="1" conaffinity="1"/>
      </body>
      <body name="rfinger" pos="-0.04 0 -0.05">
        <joint name="rf" type="slide" axis="1 0 0" range="0 0.035"/>
        <geom name="rfinger" type="box" size="0.006 0.012 0.03" mass="0.03"
              condim="3" friction="{mu} 0.02 0.002" contype="1" conaffinity="1"/>
      </body>
    </body>

    <body name="object" pos="0 0 {obj_z}">
      <freejoint name="obj"/>
      <geom name="object" type="cylinder" size="{r} {height/2.0}" mass="{mass}"
            condim="3" friction="{mu} 0.005 0.0001" contype="1" conaffinity="1"
            solref="{solref_t} 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>

  <actuator>
    <position name="apx" joint="px" kp="800" kv="40"/>
    <position name="apy" joint="py" kp="800" kv="40"/>
    <position name="apz" joint="pz" kp="800" kv="40"/>
    <position name="alf" joint="lf" kp="300" kv="8" ctrlrange="0 0.035"
              forcerange="-{squeeze_force} {squeeze_force}"/>
    <position name="arf" joint="rf" kp="300" kv="8" ctrlrange="0 0.035"
              forcerange="-{squeeze_force} {squeeze_force}"/>
  </actuator>
</mujoco>
"""


# ----------------------------------------------------------------------
# one grasp trial
# ----------------------------------------------------------------------
def run_trial(params, seed=0, save_traj=True, on_step=None):
    d_ = params["d"]; h_ = params["h"]
    obj_z = 0.20 + h_ / 2.0 + 0.001
    m = mujoco.MjModel.from_xml_string(
        scene_xml(d_, h_, params["mass"], params["mu"], params["force"],
                  params.get("solref_t", 0.01)))
    cp.force_dense_jacobian(m)
    data = mujoco.MjData(m)

    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
         for n in ("apx", "apy", "apz", "alf", "arf")}
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
    fgeoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("lfinger", "rfinger")}

    OPEN, CLOSED = 0.0, 0.030
    gx = params.get("xoff", 0.0); gy = 0.0
    # grasp_z is the palm's WORLD z; +0.065 puts the finger tips just above the
    # pedestal top so they grip the object's body, not the pedestal.
    grasp_z = obj_z + params.get("zoff", 0.0) + 0.065
    hover_z = grasp_z + 0.15
    lift_z = grasp_z + 0.15

    def ctrl(px, py, pz, lf, rf):
        # pz is WORLD z; the slide joint is a displacement from PALM_HOME.
        data.ctrl[A["apx"]] = px; data.ctrl[A["apy"]] = py; data.ctrl[A["apz"]] = pz - PALM_HOME
        data.ctrl[A["alf"]] = lf; data.ctrl[A["arf"]] = rf

    def settle(px, py, pz, lf, rf, n):
        for _ in range(n):
            ctrl(px, py, pz, lf, rf); mujoco.mj_step(m, data)

    def ramp(p0, p1, lf, rf, n, log=None):
        for k in range(n):
            a = (k + 1) / n
            p = [p0[i] + a * (p1[i] - p0[i]) for i in range(3)]
            ctrl(p[0], p[1], p[2], lf, rf); mujoco.mj_step(m, data)
            if log is not None:
                log()

    # place gripper open, above object
    ctrl(gx, gy, hover_z, OPEN, OPEN)
    settle(gx, gy, hover_z, OPEN, OPEN, 60)
    mujoco.mj_forward(m, data)
    p0, q0 = cp.body_pose(data, obj_bid)
    obj_z0 = float(p0[2])

    # ---------- per-step logger (contact phase) ----------
    traj = {k: [] for k in ("t", "obj_p", "obj_q", "obj_v", "obj_w",
                            "dp", "drot", "n_obj_contacts", "Fn_sum",
                            "W_block_diag", "W_block_offdiag_max", "grip")}
    peak_z = [obj_z0]
    prev = {"p": p0.copy(), "q": q0.copy()}

    def log_step():
        mujoco.mj_forward(m, data)
        p, q = cp.body_pose(data, obj_bid)
        v, w = cp.body_vel(m, data, obj_bid)
        recs = cp.contact_records(m, data, involving_geoms={obj_gid})
        W, _ = assemble_or_empty(m, data)
        Wb, labels = cp.object_W_block(W, recs)
        peak_z[0] = max(peak_z[0], float(p[2]))
        traj["t"].append(len(traj["t"]))
        traj["obj_p"].append(p); traj["obj_q"].append(q)
        traj["obj_v"].append(v); traj["obj_w"].append(w)
        traj["dp"].append(p - prev["p"])
        traj["drot"].append(cp.quat_delta_rotvec(prev["q"], q))
        traj["n_obj_contacts"].append(len(recs))
        traj["Fn_sum"].append(sum(r["Fn"] for r in recs))
        if Wb.size:
            traj["W_block_diag"].append(float(np.mean(np.diag(Wb))))
            off = Wb - np.diag(np.diag(Wb))
            traj["W_block_offdiag_max"].append(float(np.abs(off).max()))
        else:
            traj["W_block_diag"].append(0.0); traj["W_block_offdiag_max"].append(0.0)
        traj["grip"].append(float(data.qpos[m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "lf")]]))
        if on_step is not None:
            on_step(m, data, W, obj_bid, obj_gid,
                    dict(dp=(p - prev["p"]), drot=cp.quat_delta_rotvec(prev["q"], q),
                         obj_v=v, obj_w=w, dt=float(m.opt.timestep)))
        prev["p"] = p.copy(); prev["q"] = q.copy()

    # descend (open) — no logging yet (no contact)
    ramp([gx, gy, hover_z], [gx, gy, grasp_z], OPEN, OPEN, 80)
    # close (log: contact forms here)
    ramp([gx, gy, grasp_z], [gx, gy, grasp_z], CLOSED, CLOSED, 60, log=log_step)
    settle_log(m, data, ctrl, gx, gy, grasp_z, CLOSED, 40, log_step)
    # lift (log: the informative transition)
    ramp([gx, gy, grasp_z], [gx, gy, lift_z], CLOSED, CLOSED, 90, log=log_step)
    settle_log(m, data, ctrl, gx, gy, lift_z, CLOSED, 60, log_step)

    # ---------- outcome ----------
    p_fin, _ = cp.body_pose(data, obj_bid)
    rise_final = float(p_fin[2]) - obj_z0
    rise_peak = float(peak_z[0]) - obj_z0
    grip_final = traj["grip"][-1] if traj["grip"] else 0.0
    n_contact_end = traj["n_obj_contacts"][-1] if traj["n_obj_contacts"] else 0
    if rise_final >= GATE_RISE:
        outcome = "stable_lift"
    elif rise_peak >= GATE_RISE and rise_final < 0.5 * GATE_RISE:
        outcome = "lift_then_drop"
    elif grip_final > 0.028 and n_contact_end == 0:
        outcome = "no_contact"
    elif n_contact_end > 0 and rise_peak < GATE_RISE:
        outcome = "grip_no_lift"
    else:
        outcome = "weak"

    wdiag = np.array(traj["W_block_diag"]) if traj["W_block_diag"] else np.array([0.0])
    woff = np.array(traj["W_block_offdiag_max"]) if traj["W_block_offdiag_max"] else np.array([0.0])
    fn = np.array(traj["Fn_sum"]) if traj["Fn_sum"] else np.array([0.0])
    dpmax = (max(np.linalg.norm(np.array(traj["dp"]), axis=1)) * 1000) if traj["dp"] else 0.0
    rec = dict(seed=seed, tag=params.get("tag", ""),
               d_mm=d_ * 1000, h_mm=h_ * 1000, mass=params["mass"], mu=params["mu"],
               force_N=params["force"], zoff_mm=params.get("zoff", 0.0) * 1000,
               xoff_mm=params.get("xoff", 0.0) * 1000,
               rise_peak_mm=rise_peak * 1000, rise_final_mm=rise_final * 1000,
               n_steps=len(traj["t"]), n_contact_end=n_contact_end,
               grip_final_mm=grip_final * 1000,
               W_diag_mean=float(wdiag.mean()), W_diag_max=float(wdiag.max()),
               W_offdiag_max=float(woff.max()), Fn_max=float(fn.max()),
               dp_step_max_mm=float(dpmax), outcome=outcome)

    if save_traj and len(traj["t"]):
        os.makedirs(OUT_DIR, exist_ok=True)
        arrs = {k: np.array(v) for k, v in traj.items()}
        arrs["meta"] = np.array([rec[k] for k in ("d_mm", "h_mm", "mass", "mu",
                                                   "force_N", "zoff_mm")])
        np.savez_compressed(os.path.join(OUT_DIR, "traj_%s_%d.npz" %
                                         (params.get("tag", "t"), seed)), **arrs)
    return rec


def assemble_or_empty(m, data):
    try:
        return cp.assemble_W(m, data, use_solve=True)
    except Exception:
        return np.zeros((0, 0)), None


def settle_log(m, data, ctrl, px, py, pz, grip, n, log):
    for _ in range(n):
        ctrl(px, py, pz, grip, grip); mujoco.mj_step(m, data); log()


# ----------------------------------------------------------------------
# a small spanning sweep (near the failure boundaries the Isaac harness found:
# squeeze force + vertical alignment matter; friction barely does)
# ----------------------------------------------------------------------
BASE = dict(tag="base", d=0.030, h=0.060, mass=0.10, mu=1.0, force=20.0, zoff=0.0, xoff=0.0)
SWEEP = [
    dict(tag="base"),
    dict(tag="strongF",  force=40.0),
    dict(tag="weakF_3",  force=3.0),
    dict(tag="weakF_1",  force=1.0),
    dict(tag="lowmu",    mu=0.15, force=20.0),
    dict(tag="highz",    zoff=+0.020),
    dict(tag="xoff_15",  xoff=0.015),
    dict(tag="small",    d=0.020),
    dict(tag="big",      d=0.044),
]


def random_params(rng):
    """Sample a trial from continuous ranges spanning easy->hard. Misalignment
    (xoff) and vertical offset (zoff) are the strong diversity knobs; friction
    and squeeze force are weaker for a caged grasp (confirmed empirically)."""
    return dict(
        tag="r",
        d=float(rng.uniform(0.018, 0.046)),
        h=float(rng.uniform(0.045, 0.075)),
        mass=float(rng.uniform(0.03, 0.20)),
        mu=float(rng.uniform(0.10, 1.40)),
        force=float(10 ** rng.uniform(np.log10(0.5), np.log10(40.0))),
        zoff=float(rng.uniform(-0.006, 0.022)),
        xoff=float(rng.uniform(-0.018, 0.018)),
    )


def generate_dataset(n, seed=0, save_every=25):
    """Randomized batch — the real data-generation entry point. Always writes a
    trials.csv summary (incl. per-trial W/force aggregates); saves the full
    per-step npz for every `save_every`-th trial (set 1 to keep them all)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    print("M2 randomized generation: %d trials (classic MuJoCo)" % n)
    print("=" * 66)
    for i in range(n):
        p = random_params(rng)
        r = run_trial(p, seed=i, save_traj=(i % save_every == 0))
        rows.append(r)
        if (i + 1) % max(1, n // 20) == 0 or i == n - 1:
            print("  [%d/%d] d=%.0f mu=%.2f F=%4.1f zoff=%+.0f xoff=%+.0f -> %s (rise %+.1f)"
                  % (i + 1, n, r["d_mm"], r["mu"], r["force_N"], r["zoff_mm"],
                     r["xoff_mm"], r["outcome"], r["rise_final_mm"]))
    _write_and_summarize(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print("M2 floating-gripper contact-transition generator (classic MuJoCo)")
    print("=" * 66)
    for i, ov in enumerate(SWEEP):
        p = dict(BASE); p.update(ov)
        r = run_trial(p, seed=i)
        rows.append(r)
        print("  [%d/%d] %-9s F=%4.1f mu=%.2f d=%.0f zoff=%+.0f xoff=%+.0f "
              "-> peak=%+6.1f final=%+6.1f  %s"
              % (i + 1, len(SWEEP), r["tag"], r["force_N"], r["mu"], r["d_mm"],
                 r["zoff_mm"], r["xoff_mm"], r["rise_peak_mm"], r["rise_final_mm"],
                 r["outcome"]))
    _write_and_summarize(rows)


def _write_and_summarize(rows):
    cols = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "trials.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    from collections import Counter
    dist = Counter(r["outcome"] for r in rows)
    print("\n[OUTCOME DISTRIBUTION] " + "  ".join("%s=%d" % kv for kv in dist.items()))
    print("[OK] wrote %s/trials.csv and per-trial traj_*.npz" % OUT_DIR)
    print("     each npz has per-step: obj_p/q/v/w, dp, drot, Fn_sum, n_obj_contacts,")
    print("     W_block_diag, W_block_offdiag_max  (the W coupling over object contacts).")


if __name__ == "__main__":
    import sys
    # `python m2_floating_gripper_grasp.py`            -> the 9-case demo sweep
    # `python m2_floating_gripper_grasp.py random 500` -> 500 randomized trials
    if len(sys.argv) >= 2 and sys.argv[1] == "random":
        n = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
        generate_dataset(n, seed=0)
    else:
        main()
