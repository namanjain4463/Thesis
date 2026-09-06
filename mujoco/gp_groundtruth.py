"""
gp_groundtruth.py — build the ground-truth outcome table for the grasp-and-place benchmark.

For every (family, set, instance, candidate, body) it runs the full episode and records the
label + metrics. Sets are DISJOINT instance pools (train on source S; calibrate + test on
target T). The table is the substrate the method comparison and adaptation-curve run on top
of (methods only SELECT candidates; outcomes are looked up here). Saves gp_out/table.npz and
a per-episode JSONL with seeds.

Run:  python gp_groundtruth.py [n_per_family]
"""
import os, json, itertools, numpy as np
from gp_core import scene, run_episode, BODIES, SOURCE_BODIES, TARGET_BODY

FAMILIES = ("A", "B", "C")
TSPEED = 0.35
# common candidate set (SAME for every method and both bodies): where + orientation + how-hard
GY   = (0.0, 0.02, 0.035, 0.05)
YAW  = (0.0, np.pi/2)
CLOSEF = (1.0, 0.6)
CANDS = [dict(gy=gy, yaw=yaw, close_frac=cf, tspeed=TSPEED)
         for gy, yaw, cf in itertools.product(GY, YAW, CLOSEF)]

SIGMA_COM = 0.008     # camera CoM-estimate noise (m), shared by ALL practical methods
SIGMA_MU  = 0.15      # friction-estimate noise, shared by all practical methods


def gen_instances(family, n, rng):
    """Hidden per-instance variation + the SHARED noisy estimates every practical method sees."""
    out = []
    for i in range(n):
        mu = float(rng.uniform(0.6, 1.2))
        if family == "B":
            shift = float(rng.uniform(0.045, 0.075)); me = float(rng.uniform(0.14, 0.26))
            inst = dict(mu=mu, shift=shift, me=me)
        else:
            inst = dict(mu=mu)
        _, meta = scene(family, SOURCE_BODIES[0], mu=mu, inst=inst)
        comy_true = meta["comy"]; mass_true = meta["mass"]
        # ONE shared estimate per instance (identical for every method — no hidden advantage)
        obs = dict(family=family, fphalf=meta["fphalf"], hz=meta["hz"],
                   pocket_hx=meta["pocket_hx"], pocket_hy=meta["pocket_hy"],
                   comy_est=float(comy_true + rng.normal(0, SIGMA_COM)),
                   mu_est=float(np.clip(mu + rng.normal(0, SIGMA_MU), 0.2, 1.6)))
        out.append(dict(inst=inst, mu=mu, comy_true=comy_true, mass_true=mass_true, obs=obs))
    return out


def close_abs(body, cf):
    return cf * BODIES[body]["frange"]


def build_table(n_per_family=8, seed=0, out_dir="gp_out"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    # DISJOINT instance pools (separate draws): train evaluated on the SOURCE-body population,
    # calib + test evaluated on the held-out TARGET body.
    pools = {}
    for fam in FAMILIES:
        pools[(fam, "train")] = gen_instances(fam, n_per_family, rng)
        pools[(fam, "calib")] = gen_instances(fam, n_per_family, rng)
        pools[(fam, "test")]  = gen_instances(fam, n_per_family, rng)
    # (set -> list of bodies to evaluate on)
    setbodies = {"train": list(SOURCE_BODIES), "calib": [TARGET_BODY], "test": [TARGET_BODY]}

    records = []
    table = {}   # (fam,set,body,inst_idx,cand_idx) -> outcome dict
    nrun = 0
    for fam in FAMILIES:
        for setname, bodies in setbodies.items():
            insts = pools[(fam, setname)]
            for body in bodies:
                for ii, ins in enumerate(insts):
                    for ci, cand in enumerate(CANDS):
                        act = dict(gy=cand["gy"], yaw=cand["yaw"],
                                   close=close_abs(body, cand["close_frac"]), tspeed=cand["tspeed"])
                        r = run_episode(fam, body, act, mu=ins["mu"], inst=ins["inst"])
                        rec = dict(family=fam, setname=setname, body=body, inst_idx=ii, cand_idx=ci,
                                   placed=int(r["placed"]), label=r["label"], pos_err=r["pos_err"],
                                   final_tilt=r["final_tilt"], z_err=r["z_err"], rel_z_err=r["rel_z_err"],
                                   gy=cand["gy"], yaw=cand["yaw"], close_frac=cand["close_frac"],
                                   tspeed=cand["tspeed"], mu_true=ins["mu"], comy_true=ins["comy_true"],
                                   mass_true=ins["mass_true"], comy_est=ins["obs"]["comy_est"],
                                   mu_est=ins["obs"]["mu_est"], seed=seed)
                        table[(fam, setname, body, ii, ci)] = rec
                        records.append(rec)
                        nrun += 1
        print(f"  family {fam}: ran {nrun} episodes so far")
    # persist
    np.savez(os.path.join(out_dir, "table.npz"),
             records=np.array(json.dumps(records)), cands=np.array(json.dumps(CANDS)),
             pools=np.array(json.dumps({f"{k[0]}|{k[1]}": v for k, v in pools.items()})),
             n_per_family=n_per_family, seed=seed)
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for rec in records: f.write(json.dumps(rec) + "\n")
    return table, pools, records


def landscape(records):
    """Sanity: per family/body, naive-grasp success, oracle-best success, and S->T gap."""
    import collections
    bykey = collections.defaultdict(list)
    for r in records: bykey[(r["family"], r["setname"], r["body"])].append(r)
    print("\n" + "=" * 90)
    print("LANDSCAPE — is there headroom and a source->target gap?")
    print("=" * 90)
    print(" family/set/body | oracle-best | naive(gy0,yaw0,full) | frac-cands-feasible | mean over insts")
    for fam in FAMILIES:
        for setname in ("train", "calib", "test"):
            bodies = sorted(set(r["body"] for r in records
                                if r["family"] == fam and r["setname"] == setname))
            for body in bodies:
                rr = [r for r in records if r["family"] == fam and r["setname"] == setname and r["body"] == body]
                insts = sorted(set(r["inst_idx"] for r in rr))
                orac = []; naive = []; frac = []
                for ii in insts:
                    ri = [r for r in rr if r["inst_idx"] == ii]
                    orac.append(1 if any(r["placed"] for r in ri) else 0)
                    nv = [r for r in ri if r["gy"] == 0.0 and r["yaw"] == 0.0 and r["close_frac"] == 1.0]
                    naive.append(nv[0]["placed"] if nv else 0)
                    frac.append(np.mean([r["placed"] for r in ri]))
                print(f"   {fam}/{setname:5s}/{body:2s} |   {np.mean(orac):.2f}      |      {np.mean(naive):.2f}          "
                      f"|     {np.mean(frac):.2f}          | n={len(insts)}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Building ground-truth table: {len(CANDS)} candidates x {n} inst/family x 3 sets x 3 families")
    table, pools, records = build_table(n_per_family=n)
    print(f"  total episodes: {len(records)}")
    landscape(records)
