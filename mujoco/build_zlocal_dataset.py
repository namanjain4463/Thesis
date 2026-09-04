"""
build_zlocal_dataset.py  —  turn randomized grasp trials into a consolidated
z_local training set (schema z_local.v1).

Each SAMPLE = one (contact, timestep): the strict-local feature vector X[i]
(25-D), the self-inertia block W_ii[i] (9-D), and the OBSERVABLE object-motion
target Y[i] (dp, drot, v, w, dt = 13-D) for that step. Grouping ids let you do
leave-one-trial-out splits and per-contact-topology analysis.

Run:  python build_zlocal_dataset.py [n_trials] [stride]
"""
import sys, numpy as np, mujoco
import m2_floating_gripper_grasp as M
import z_local_schema as Z


def build(n=40, seed=0, stride=3, out="zlocal_dataset.npz"):
    rng = np.random.default_rng(seed)
    X, Wii, Y, trial_id, step_id, prm, other, ogeom = [], [], [], [], [], [], [], []

    for ti in range(n):
        p = M.random_params(rng)
        ctr = {"s": 0}
        tracker = Z.ContactTracker()

        def on_step(m, data, W, obid, ogid, tgt, _p=p, _ti=ti, _ctr=ctr, _trk=tracker):
            _ctr["s"] += 1
            slip = _trk.update(m, data, tgt["dt"])     # EVERY step (accumulation must be continuous)
            if _ctr["s"] % stride:
                return
            tvec = np.concatenate([tgt["dp"], tgt["drot"], tgt["obj_v"], tgt["obj_w"], [tgt["dt"]]])
            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                if not (c.geom1 == ogid or c.geom2 == ogid):
                    continue
                sa, ag = slip.get(ci, (0.0, 0.0))
                f = Z.contact_features(m, data, ci, obid, ogid, W=W, slip_accum=sa, contact_age=ag)
                X.append(Z.strict_vector(f)); Wii.append(f["aux"]["W_ii"]); Y.append(tvec)
                trial_id.append(_ti); step_id.append(_ctr["s"]); ogeom.append(f["other_geom"])
                prm.append([_p["d"], _p["h"], _p["mass"], _p["mu"], _p["force"],
                            _p.get("zoff", 0.0), _p.get("xoff", 0.0)])

        M.run_trial(p, seed=ti, save_traj=False, on_step=on_step)

    X = np.asarray(X); Wii = np.asarray(Wii); Y = np.asarray(Y)
    np.savez_compressed(
        out, X=X, W_ii=Wii, Y=Y,
        trial_id=np.asarray(trial_id), step_id=np.asarray(step_id),
        other_geom=np.asarray(ogeom), params=np.asarray(prm),
        schema=Z.SCHEMA_VERSION,
        strict_fields=np.array([f"{n_}:{d_}" for n_, d_ in Z.STRICT_FIELDS]),
        target_fields=np.array([f"{n_}:{d_}" for n_, d_ in Z.TARGET_FIELDS]))
    return out, X, Wii, Y, np.asarray(trial_id)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out, X, Wii, Y, tid = build(n=n, stride=stride)
    print("wrote", out)
    print("  X (features)  :", X.shape, " expected cols =", Z.STRICT_DIM)
    print("  W_ii          :", Wii.shape)
    print("  Y (targets)   :", Y.shape, " [dp3 drot3 v3 w3 dt1]")
    print("  from %d trials -> %d contact-step samples" % (n, X.shape[0]))
    print("  NaNs in X: %d   (primitives => expect 0)" % int(np.isnan(X).sum()))
    print("  finite W_ii rows: %d / %d" % (int(np.isfinite(Wii).all(1).sum()), Wii.shape[0]))
    # quick sanity: does |vt| (cols 8:10) separate lifted vs not, using |dp_z| target?
    vt = np.linalg.norm(X[:, 8:10], axis=1)
    rising = Y[:, 2] > 1e-4          # dp_z > 0.1mm/step
    print("  median |vt|  rising-steps=%.4f  non-rising=%.4f m/s"
          % (np.median(vt[rising]) if rising.any() else float('nan'),
             np.median(vt[~rising]) if (~rising).any() else float('nan')))
