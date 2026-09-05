"""
deleak_dataset.py — build a DE-LEAKED contact dataset for training the local
contact law C_theta, on BOTH embodiments (floating gripper + Panda).

Why "de-leaked": the frozen z_local schema feeds the TRUE material parameters
(mu, solref, solimp) straight into the model, so a network "learns" nothing — it
reads the answer. Here the ONLY material handle is a categorical MATERIAL ID (you
know which object you picked up), never the physical parameters. Everything else
is a genuinely observable local quantity:

  features (observable):  penetration = max(-gap,0),  v_n,  |v_t|,  kappa_obj (from
                          the object SHAPE),  contact height in object frame,
                          |normal . world_z|
  material id          :  categorical index into a known material set (NOT the
                          solref/solimp/mu numbers)
  target               :  F_n (measured contact normal force)

Materials vary the OBJECT normal stiffness (solref time-constant), friction fixed.
At the grasp hold the object is in equilibrium so (penetration, F_n) traces the
material's constitutive line F_n ~ k_material * penetration; k_material is what the
factorization claims is EMBODIMENT-INVARIANT. Training C_theta on the floating
gripper and testing on the Panda therefore MEASURES that invariance (and exposes
MuJoCo's inertia-scaled regularizer R if it breaks it).

Run:  MENAGERIE_DIR=... python deleak_dataset.py [n_float_per_mat] [n_panda_per_mat]
Writes deleak_out/deleak_data.npz (gitignored).
"""
import os, sys, numpy as np, mujoco
import contact_probe as cp
import m2_floating_gripper_grasp as FG
import panda_embodiment as PA

OUT_DIR = "deleak_out"
# 5 known materials: object normal solref time-constant (smaller = stiffer). mu fixed.
MATERIALS = [0.004, 0.008, 0.015, 0.025, 0.040]     # material id = index
MU_FIXED = 1.0
# W_nn is the analytical per-contact normal Delassus (embodiment-carrying PORT, computed
# from M — NOT a leaked material param). It is the last feature so the "local-only"
# (naive) model can drop it while the "factorized" model uses it.
FEATURES = ["pen", "vn", "vt", "kappa", "cz_obj", "nz", "W_nn"]


def contact_obs(m, d, W, obj_gid, finger_gids, r_obj):
    """Per object-finger contact: (feature vector, F_n). Observable/analytical only."""
    out = []
    nefc = int(d.nefc)
    J = np.array(d.efc_J).reshape(nefc, m.nv) if nefc else None
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    R_obj = np.array(d.xmat[ob]).reshape(3, 3); p_obj = np.array(d.xpos[ob])
    for ci in range(int(d.ncon)):
        c = d.contact[ci]
        if not ((c.geom1 == obj_gid or c.geom2 == obj_gid) and
                (c.geom1 in finger_gids or c.geom2 in finger_gids)):
            continue
        f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6)
        Fn = float(f6[0])
        if Fn <= 1e-6:
            continue
        frame = np.array(c.frame).reshape(3, 3)
        addr, dim = int(c.efc_address), int(c.dim)
        vc = (J[addr:addr + dim] @ d.qvel) if (nefc and dim >= 1) else np.zeros(3)
        vn = float(vc[0]); vt = float(np.hypot(vc[1], vc[2])) if dim >= 3 else 0.0
        pen = max(-float(c.dist), 0.0)
        cz_obj = float((R_obj.T @ (np.array(c.pos) - p_obj))[2])   # height on object
        nz = abs(float(frame[0][2]))                               # normal alignment to world z
        kappa = 1.0 / r_obj                                        # cylinder side curvature (shape)
        W_nn = float(W[addr, addr]) if (W is not None and W.shape[0] > addr) else np.nan
        out.append(([pen, vn, vt, kappa, cz_obj, nz, W_nn], Fn))
    return out


def collect_float(n_per_mat, rng):
    rows = []                                   # (feat.., Fn, mat_id, trial_id)
    tid = 0
    for mid, sref in enumerate(MATERIALS):
        for _ in range(n_per_mat):
            p = FG.random_params(rng)
            p["mu"] = MU_FIXED; p["solref_t"] = sref
            r_obj = p["d"] / 2.0
            store = {"tid": tid}
            lg = None
            def on_step(m, d, W, obj_bid, obj_gid, tgt, _r=r_obj, _mid=mid, _st=store):
                fg = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("lfinger", "rfinger")}
                for feat, Fn in contact_obs(m, d, W, obj_gid, fg, _r):
                    rows.append(feat + [Fn, _mid, _st["tid"]])
            FG.run_trial(p, seed=tid, save_traj=False, on_step=on_step)
            tid += 1
    return rows


def collect_panda(n_per_mat, rng):
    rows = []; tid = 10000
    for mid, sref in enumerate(MATERIALS):
        for _ in range(n_per_mat):
            d_ = float(rng.uniform(0.045, 0.055)); h_ = float(rng.uniform(0.07, 0.09))
            mass = float(rng.uniform(0.03, 0.06)); hoff = float(rng.uniform(-0.015, 0.015))
            p = dict(d=d_, h=h_, mass=mass, mu=MU_FIXED, force=15.0, solref_t=sref)
            r_obj = d_ / 2.0
            store = {"tid": tid}
            fgids = {}
            def on_step(m, d, W, obj_bid, obj_gid, phase, _r=r_obj, _mid=mid, _st=store, _fg=fgids):
                if "g" not in _fg: _fg["g"] = PA._finger_geoms(m)
                for feat, Fn in contact_obs(m, d, W, obj_gid, _fg["g"], _r):
                    rows.append(feat + [Fn, _mid, _st["tid"]])
            PA.run_grasp(p, on_step=on_step)
            tid += 1
    return rows


def main():
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    npnd = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    print("collecting FLOATING (%d trials/material x %d materials)..." % (nf, len(MATERIALS)))
    rf = collect_float(nf, rng)
    print("  floating contact-step samples:", len(rf))
    print("collecting PANDA (%d trials/material x %d materials)..." % (npnd, len(MATERIALS)))
    rp = collect_panda(npnd, rng)
    print("  panda contact-step samples:", len(rp))
    F = np.array(rf, dtype=np.float64); P = np.array(rp, dtype=np.float64)
    np.savez_compressed(os.path.join(OUT_DIR, "deleak_data.npz"),
                        float_rows=F, panda_rows=P,
                        features=np.array(FEATURES), materials=np.array(MATERIALS))
    print("wrote %s/deleak_data.npz   float=%s panda=%s  cols=%s+[Fn,mat_id,trial_id]"
          % (OUT_DIR, F.shape, P.shape, FEATURES))


if __name__ == "__main__":
    main()
