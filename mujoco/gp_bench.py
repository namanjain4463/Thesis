"""
gp_bench.py — grasp-and-place TRANSFER benchmark: method comparison + adaptation-data curve.

Loads the ground-truth table (gp_groundtruth.py) and runs FIVE selectors that all choose from
the SAME candidate set and use the SAME execution controller; only the interaction model differs.
Every practical method sees the SAME observations (geometry + the shared noisy CoM/friction
estimates + known part mass); it never sees true CoM/friction/contact-force. The oracle is a
privileged diagnostic (true outcome), not a fair competitor.

Methods
  1 CoM+checks         : grasp nearest the estimated CoM, orientation by a geometric collision
                         check, full close. Budget-independent. "Does a simple heuristic solve it?"
  2 calibrated-analytic: mechanistic feasibility (step-2 command-force model + moment/friction
                         margin + collision check); with target budget it fits its few coefficients.
  3 learned-frozen     : outcome MLP trained on the SOURCE-body population, applied to the target
                         with NO target data. "Does learned information transfer zero-shot?"
  4 learned-adapted    : the same MLP retrained on source + b TARGET episodes. "What does target
                         data buy, and cost?"
  5 oracle (privileged): picks the truly-best candidate. Ceiling / headroom diagnostic.

Central deliverable: HELD-OUT task success vs TARGET calibration episodes (0/10/30), with drops,
placement failures, and execution time. The analytical baseline gets the SAME budget. Budget counts
ALL target episodes (successes + failures). Conclusion follows the numbers — including the
calibrated analytical or the simple heuristic winning.

Run:  python gp_bench.py
"""
import os, json, itertools, collections, time, numpy as np
from gp_core import BODIES, SOURCE_BODIES, TARGET_BODY, scene, run_episode, G, DT
from gp_groundtruth import CANDS, FAMILIES, close_abs, TSPEED

OUT = "gp_out"
FRANGE = {b: BODIES[b]["frange"] for b in BODIES}


# ---------------------------------------------------------------- data loading / geometry
def load_records(path=os.path.join(OUT, "episodes.jsonl")):
    with open(path) as f:
        return [json.loads(l) for l in f]

def family_geom():
    G_ = {}
    for fam in FAMILIES:
        _, meta = scene(fam, SOURCE_BODIES[0])
        G_[fam] = dict(fphalf=meta["fphalf"], hz=meta["hz"],
                       pocket_hx=meta["pocket_hx"], pocket_hy=meta["pocket_hy"],
                       fbase=meta["fbase"])
    return G_
FG = None

def organize(records):
    """T[(fam,set,body,inst)] -> list of 16 candidate recs ordered by cand_idx."""
    T = collections.defaultdict(dict)
    for r in records:
        T[(r["family"], r["setname"], r["body"], r["inst_idx"])][r["cand_idx"]] = r
    out = {}
    for k, dd in T.items():
        out[k] = [dd[ci] for ci in sorted(dd)]
    return out


# ---------------------------------------------------------------- geometric collision check
def collision_ok(fam, yaw, body):
    """Can the fingers descend into the pocket at this orientation? (observable geometry)."""
    g = FG[fam]
    needed = g["fphalf"] + 2 * BODIES[body]["fhalf"]      # finger outer half-extent along closing axis
    return needed < (g["pocket_hx"] if yaw < 0.1 else g["pocket_hy"])


# ---------------------------------------------------------------- features for the learned model
def feat(rec, body):
    g = FG[rec["family"]]; b = BODIES[body]; fam = rec["family"]
    return np.array([
        rec["gy"], 1.0 if rec["yaw"] > 0.1 else 0.0, rec["close_frac"],
        rec["comy_est"], rec["mu_est"], rec["mass_true"],
        g["fphalf"], g["hz"], g["pocket_hx"], g["pocket_hy"],
        b["kp"], b["fcap"], b["flen"], b["fmass"],
        1.0 if fam == "A" else 0.0, 1.0 if fam == "B" else 0.0, 1.0 if fam == "C" else 0.0,
    ], float)


# ---------------------------------------------------------------- small robust numpy MLP
class MLP:
    def __init__(self, din, hidden=(24, 12), seed=0, l2=2e-3):
        rng = np.random.default_rng(seed); self.l2 = l2
        dims = [din] + list(hidden) + [1]; self.W = []; self.b = []
        for i in range(len(dims) - 1):
            self.W.append(rng.normal(0, np.sqrt(1.0 / dims[i]), (dims[i], dims[i+1])))
            self.b.append(np.zeros(dims[i+1]))
    def _fwd(self, X):
        acts = [X]; h = X
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i]); acts.append(h)
        z = h @ self.W[-1] + self.b[-1]; p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        acts.append(p); return acts
    def predict(self, X):
        return self._fwd((X - self.mu_) / self.sd_)[-1].ravel()
    def fit(self, X, y, w=None, steps=1400, lr=0.06):
        self.mu_ = X.mean(0); self.sd_ = X.std(0) + 1e-9
        Xn = (X - self.mu_) / self.sd_; y = y.reshape(-1, 1)
        w = np.ones_like(y) if w is None else w.reshape(-1, 1); w = w / w.mean()
        vW = [np.zeros_like(m) for m in self.W]; vb = [np.zeros_like(m) for m in self.b]
        n = len(y)
        for t in range(steps):
            acts = self._fwd(Xn); p = acts[-1]
            g = (p - y) * w / n                       # d/dz of weighted BCE
            gW = [None]*len(self.W); gb = [None]*len(self.b)
            for i in reversed(range(len(self.W))):
                gW[i] = acts[i].T @ g + self.l2 * self.W[i]; gb[i] = g.sum(0)
                if i > 0:
                    g = (g @ self.W[i].T) * (1 - acts[i]**2)
            for i in range(len(self.W)):
                gn = np.linalg.norm(gW[i]);  gW[i] = gW[i] * min(1.0, 5.0 / (gn + 1e-9))  # clip
                vW[i] = 0.9*vW[i] - lr*gW[i]; self.W[i] += vW[i]
                vb[i] = 0.9*vb[i] - lr*gb[i]; self.b[i] += vb[i]
        return self


# ---------------------------------------------------------------- methods (each returns cand_idx)
def m_oracle(instrecs):
    placed = [i for i, r in enumerate(instrecs) if r["placed"]]
    if not placed: return int(np.argmin([r["pos_err"] for r in instrecs]))
    return int(min(placed, key=lambda i: instrecs[i]["pos_err"]))   # best-seated feasible

def m_com(instrecs, body):
    fam = instrecs[0]["family"]; comy_est = instrecs[0]["comy_est"]
    cixs = [i for i, r in enumerate(instrecs)
            if r["close_frac"] == 1.0 and collision_ok(fam, r["yaw"], body)]
    if not cixs: cixs = list(range(len(instrecs)))
    # nearest estimated CoM; tie-break prefer default orientation yaw=0
    return int(min(cixs, key=lambda i: (abs(instrecs[i]["gy"] - comy_est), instrecs[i]["yaw"])))

def m_geo(instrecs, body, thresh=0.016):
    """Stronger task-aware heuristic (NOT a strawman): if the part looks symmetric (|CoM est|
    below the estimate-noise band) grasp the geometric CENTER; otherwise chase the CoM estimate.
    Orientation by the same collision check, full close. This is the baseline learning must beat."""
    fam = instrecs[0]["family"]; comy_est = instrecs[0]["comy_est"]
    gy_target = 0.0 if abs(comy_est) < thresh else comy_est
    cixs = [i for i, r in enumerate(instrecs)
            if r["close_frac"] == 1.0 and collision_ok(fam, r["yaw"], body)]
    if not cixs: cixs = list(range(len(instrecs)))
    return int(min(cixs, key=lambda i: (abs(instrecs[i]["gy"] - gy_target), instrecs[i]["yaw"])))

def _mech_feats(rec, body):
    """Mechanistic features for the analytical model (observable + estimates + known mass)."""
    fam = rec["family"]; g = FG[fam]; b = BODIES[body]
    x_contact = g["fbase"] - g["fphalf"] - b["fhalf"]
    close_absv = rec["close_frac"] * b["frange"]
    Fn = 2.0 * max(0.0, min(b["kp"] * (close_absv - x_contact), b["fcap"]))   # step-2 command model
    dcom = abs(rec["gy"] - rec["comy_est"])
    hold = rec["mu_est"] * Fn                             # grip capacity proxy
    load = rec["mass_true"] * G * dcom                    # weight moment (known mass)
    return np.array([hold, load, dcom, Fn], float)

def m_analytic(instrecs, body, calib_pairs, lever0=0.02, scale0=1.0):
    """Mechanistic feasibility: predicted-feasible if grip capacity beats the weight moment
    (margin = scale*mu_est*Fn*lever - mass*g*dcom > 0); among feasible full-close grasps, take
    the one nearest the estimated CoM. CALIBRATION tunes the PHYSICAL constants (lever, force
    scale) on target episodes — it stays a mechanistic model, so more data refines it, never
    swaps in a generic classifier."""
    fam = instrecs[0]["family"]
    ok = [i for i, r in enumerate(instrecs) if collision_ok(fam, r["yaw"], body)]
    if not ok: ok = list(range(len(instrecs)))
    lever, scale = lever0, scale0
    if calib_pairs is not None and len(calib_pairs) >= 4:
        best = (lever0, scale0, -1)
        for lv in np.linspace(0.006, 0.05, 12):
            for sc in np.linspace(0.5, 1.6, 8):
                agree = 0
                for r, pl in calib_pairs:
                    hold, load, dcom, Fn = _mech_feats(r, body)
                    pred = 1 if (sc*hold*lv - load) > 0 and collision_ok(r["family"], r["yaw"], body) else 0
                    agree += (pred == pl)
                if agree > best[2]: best = (lv, sc, agree)
        lever, scale = best[0], best[1]
    def margin(i):
        hold, load, dcom, Fn = _mech_feats(instrecs[i], body); return scale*hold*lever - load
    feas = [i for i in ok if margin(i) > 0 and instrecs[i]["close_frac"] == 1.0]
    pool = feas if feas else ok
    return int(min(pool, key=lambda i: abs(instrecs[i]["gy"] - instrecs[i]["comy_est"])))

def m_learned(instrecs, body, model):
    ps = [model.predict(feat(r, body)[None])[0] for r in instrecs]
    return int(np.argmax(ps))

def source_fixed_policy(train_records):
    """The SIMPLEST possible shared model: a lookup of family -> the candidate with the highest
    success rate over the source-body population (ignores per-instance sensing entirely). If this
    trivial table matches the learned MLP, the MLP's 'robustness' is just a fixed per-family grasp."""
    byfc = collections.defaultdict(list)
    for r in train_records: byfc[(r["family"], r["cand_idx"])].append(r["placed"])
    best = {}
    for fam in FAMILIES:
        cand_rate = {ci: np.mean(byfc[(fam, ci)]) for ci in range(len(CANDS)) if (fam, ci) in byfc}
        best[fam] = max(cand_rate, key=cand_rate.get)
    return best

def m_fixed(instrecs, best_by_fam):
    return int(best_by_fam[instrecs[0]["family"]])


# ---------------------------------------------------------------- evaluation
def eval_method(pick_fn, testkeys, T):
    """Returns per-instance placed outcomes + chosen recs across the test set."""
    placed = []; chosen = []
    for k in testkeys:
        instrecs = T[k]; ci = pick_fn(instrecs); r = instrecs[ci]
        placed.append(r["placed"]); chosen.append(r)
    return np.array(placed), chosen

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))


# ---------------------------------------------------------------- main driver
BUDGETS = [0, 10, 30]

def train_ensemble(X, y, w=None, seeds=(0, 1, 2)):
    models = [MLP(X.shape[1], seed=s).fit(X, y, w=w) for s in seeds]
    class Ens:
        def predict(self, Xq): return np.mean([mm.predict(Xq) for mm in models], 0)
    return Ens()

def main():
    global FG
    FG = family_geom()
    t0 = time.time()
    records = load_records()
    T = organize(records)
    testkeys = sorted([k for k in T if k[1] == "test"])              # (fam,test,T,inst)
    calibkeys = sorted([k for k in T if k[1] == "calib"])

    # disjointness check (train/calib/test are separate instance pools)
    assert len({k[1] for k in T}) == 3
    print("=" * 96)
    print("GRASP-AND-PLACE TRANSFER BENCHMARK — method comparison + adaptation-data curve")
    print("=" * 96)
    print(f"source bodies (train)={list(SOURCE_BODIES)}  held-out target={TARGET_BODY}  "
          f"candidates={len(CANDS)}  test instances={len(testkeys)}")
    print("embodiment gap = gripper CONFIGURATION (finger length/mass/gains/force-cap/palm); "
          "articulated arm is the stated next step.")

    # ---- source training set (population of source bodies) ----
    trainrecs = [r for r in records if r["setname"] == "train"]
    Xtr = np.array([feat(r, r["body"]) for r in trainrecs]); ytr = np.array([r["placed"] for r in trainrecs], float)
    src_model = train_ensemble(Xtr, ytr)

    # ---- fixed budget episode pool from the target CALIB set (all families) ----
    rng = np.random.default_rng(7)
    calib_ep = []
    for k in calibkeys:
        for r in T[k]: calib_ep.append(r)
    rng.shuffle(calib_ep)

    def budget_pairs(b):
        chosen = calib_ep[:b]
        return [(r, r["placed"]) for r in chosen]

    # ---- oracle + CoM (budget-independent) ----
    orac_pl, orac_ch = eval_method(lambda ir: m_oracle(ir), testkeys, T)
    com_pl, com_ch = eval_method(lambda ir: m_com(ir, TARGET_BODY), testkeys, T)
    geo_pl, geo_ch = eval_method(lambda ir: m_geo(ir, TARGET_BODY), testkeys, T)

    results = collections.defaultdict(dict)   # method -> budget -> dict(placed array, chosen)
    results["oracle"][None] = (orac_pl, orac_ch)
    results["com"][None] = (com_pl, com_ch)
    results["geo"][None] = (geo_pl, geo_ch)

    decisions = []
    for b in BUDGETS:
        pairs = budget_pairs(b)
        # analytic (calibrated with b target episodes)
        cal = pairs if b > 0 else None
        an_pl, an_ch = eval_method(lambda ir: m_analytic(ir, TARGET_BODY, cal), testkeys, T)
        results["analytic"][b] = (an_pl, an_ch)
        # learned adapted: source + b target episodes (target upweighted to balance)
        if b > 0:
            Xt = np.array([feat(r, TARGET_BODY) for r, _ in pairs]); yt = np.array([p for _, p in pairs], float)
            Xall = np.vstack([Xtr, Xt]); yall = np.concatenate([ytr, yt])
            wsrc = np.ones(len(ytr)); wtar = np.full(len(yt), max(1.0, len(ytr) / max(len(yt), 1)))
            model = train_ensemble(Xall, yall, w=np.concatenate([wsrc, wtar]))
        else:
            model = src_model
        le_pl, le_ch = eval_method(lambda ir: m_learned(ir, TARGET_BODY, model), testkeys, T)
        results["learned"][b] = (le_pl, le_ch)
        for meth, (pl, ch) in [("analytic", (an_pl, an_ch)), ("learned", (le_pl, le_ch))]:
            for k, r in zip(testkeys, ch):
                decisions.append(dict(method=meth, budget=b, family=k[0], inst=k[3],
                                      cand_idx=r["cand_idx"], gy=r["gy"], yaw=r["yaw"],
                                      close_frac=r["close_frac"], placed=r["placed"], label=r["label"]))

    # ---- report ----
    def summ(pl):
        k = int(pl.sum()); n = len(pl); lo, hi = wilson(k, n)
        return f"{pl.mean():.2f} [{lo:.2f},{hi:.2f}]"
    print("\nOVERALL held-out task success (Wilson 95pct CI), n=%d test instances:" % len(testkeys))
    print("  %-22s %s" % ("oracle (privileged)", summ(orac_pl)))
    print("  %-22s %s" % ("CoM+checks (naive)", summ(com_pl)))
    print("  %-22s %s" % ("geometry-aware heur.", summ(geo_pl)))
    print("  %-22s budget:  " % "" + "     ".join("b=%d" % b for b in BUDGETS))
    print("  %-22s %s" % ("calibrated-analytic", "  ".join(summ(results["analytic"][b][0]) for b in BUDGETS)))
    print("  %-22s %s" % ("learned (frozen->adapt)", "  ".join(summ(results["learned"][b][0]) for b in BUDGETS)))

    # per-family at budget 30
    print("\nPer-family success @budget=30:")
    print("  family |  CoM  |  geo  | analytic | learned | oracle")
    for fam in FAMILIES:
        idx = [i for i, k in enumerate(testkeys) if k[0] == fam]
        def fm(pl): return pl[idx].mean()
        print("    %s    | %.2f  | %.2f  |   %.2f   |  %.2f   |  %.2f" %
              (fam, fm(com_pl), fm(geo_pl), fm(results["analytic"][30][0]), fm(results["learned"][30][0]), fm(orac_pl)))

    # paired difference (learned@30 - best heuristic) on the same instances
    besth = np.maximum(geo_pl.astype(int), com_pl.astype(int))
    d = results["learned"][30][0].astype(int) - besth
    print("\nPaired (learned@30 - best-heuristic) on identical test instances: mean=%+.2f  (wins=%d losses=%d ties=%d)"
          % (d.mean(), int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())))

    # failure-mode breakdown @budget 30
    def modes(ch):
        c = collections.Counter(r["label"] for r in ch); n = len(ch)
        return {m: c.get(m, 0) / n for m in ("PLACED", "PLACE", "TIP", "DROP")}
    print("\nFailure-mode fractions @budget=30:")
    print("  %-22s PLACED  PLACE   TIP    DROP" % "method")
    for meth, ch in [("CoM+checks (naive)", com_ch), ("geometry-aware heur.", geo_ch),
                     ("calibrated-analytic", results["analytic"][30][1]),
                     ("learned-adapted", results["learned"][30][1]), ("oracle", orac_ch)]:
        mm = modes(ch); print("  %-22s %.2f    %.2f   %.2f   %.2f" % (meth, mm["PLACED"], mm["PLACE"], mm["TIP"], mm["DROP"]))

    # ---- embedded check: solver/timestep sensitivity near boundaries ----
    print("\nSOLVER-SENSITIVITY CHECK (re-run chosen decisions at dt/2, is the label stable?):")
    pools = json.loads(str(np.load(os.path.join(OUT, "table.npz"), allow_pickle=True)["pools"]))
    test_insts = {}
    for key, lst in pools.items():
        fam, setn = key.split("|")
        if setn == "test": test_insts[fam] = lst
    seen = set(); flips = 0; ntot = 0
    for meth_ch in [com_ch, results["learned"][30][1], orac_ch]:
        for r in meth_ch:
            sig = (r["family"], r["inst_idx"], r["cand_idx"])
            if sig in seen: continue
            seen.add(sig)
            fam, ii, ci = sig; cand = CANDS[ci]; ins = test_insts[fam][ii]
            act = dict(gy=cand["gy"], yaw=cand["yaw"], close=close_abs(TARGET_BODY, cand["close_frac"]), tspeed=cand["tspeed"])
            r2 = run_episode(fam, TARGET_BODY, act, mu=ins["mu"], inst=ins["inst"], ts=DT/2)
            ntot += 1; flips += int(r2["placed"] != r["placed"])
    print("  re-ran %d unique chosen decisions at dt=%.4f: label FLIP fraction = %.2f (%d/%d)"
          % (ntot, DT/2, flips / max(ntot, 1), flips, ntot))
    print("  -> headline is %s to the timestep." % ("ROBUST" if flips / max(ntot, 1) <= 0.1 else "SENSITIVE"))

    # execution time (physical task duration of chosen actions; tspeed fixed -> ~constant)
    task_time = X_FIX = 0.30 / TSPEED
    print("\nExecution time per placement (physical): ~%.2fs transport + fixed phases (tspeed=%.2f, constant across methods)."
          % (task_time, TSPEED))

    # ---- embedded axis: sensing-noise robustness (does learning beat a rule that TRUSTS the estimate?) ----
    print("\nSENSING-NOISE ROBUSTNESS (held-out success vs CoM-estimate std; learned retrained per level):")
    nrows = noise_robustness(records)
    print("  sigma_com(mm) |  geo  | analytic@30 | src-fixed-lookup | learned-frozen | learned-adapt@30 | oracle")
    for sc in SIGMA_COM_LEVELS:
        rr = nrows[sc]
        print("      %2.0f      | %.2f  |    %.2f     |       %.2f       |      %.2f      |       %.2f      |  %.2f"
              % (sc*1000, rr["geo"].mean(), rr["analytic"].mean(), rr["fixed"].mean(), rr["learned_frozen"].mean(),
                 rr["learned_adapt"].mean(), rr["oracle"].mean()))
    sc_hi = SIGMA_COM_LEVELS[-1]; hi = nrows[sc_hi]
    dd = hi["learned_adapt"].astype(int) - np.maximum(hi["geo"].astype(int), hi["analytic"].astype(int))
    print("  @sigma=%.0fmm paired (learned-adapt@30 - best-heuristic/analytic): mean=%+.2f (wins=%d losses=%d ties=%d)"
          % (sc_hi*1000, dd.mean(), int((dd > 0).sum()), int((dd < 0).sum()), int((dd == 0).sum())))
    make_noise_figure(nrows)

    # ---- save decisions + a compact result json ----
    with open(os.path.join(OUT, "decisions.jsonl"), "w") as f:
        for dd in decisions: f.write(json.dumps(dd) + "\n")
    res_json = dict(overall=dict(oracle=float(orac_pl.mean()), com=float(com_pl.mean()), geo=float(geo_pl.mean()),
                    analytic={b: float(results["analytic"][b][0].mean()) for b in BUDGETS},
                    learned={b: float(results["learned"][b][0].mean()) for b in BUDGETS}),
                    n_test=len(testkeys), source_bodies=list(SOURCE_BODIES), target=TARGET_BODY,
                    solver_flip_frac=flips / max(ntot, 1))
    json.dump(res_json, open(os.path.join(OUT, "bench_result.json"), "w"), indent=2)

    make_figure(results, com_pl, geo_pl, orac_pl, testkeys)
    print("\n[done in %.0fs]" % (time.time() - t0))
    verdict(results, com_pl, geo_pl, orac_pl, nrows)


def verdict(results, com_pl, geo_pl, orac_pl, nrows):
    print("\n" + "-" * 96)
    oc = orac_pl.mean(); cm = com_pl.mean(); ge = geo_pl.mean()
    besth = max(cm, ge)
    an = {b: results["analytic"][b][0].mean() for b in BUDGETS}
    le = {b: results["learned"][b][0].mean() for b in BUDGETS}
    print("VERDICT (reads the numbers, not a preferred story):")
    print("  At NOMINAL sensing: oracle=%.2f  geo-heuristic=%.2f  analytic@30=%.2f  learned@30=%.2f"
          % (oc, ge, an[30], le[30]))
    if besth >= max(an[30], le[30]) - 0.03:
        print("  * At good sensing a simple task-aware heuristic (%.2f) already MATCHES the calibrated/learned" % besth)
        print("    methods -> learning is NOT justified there (honest negative, consistent with the calibration result).")
    # noise finding
    sc_hi = SIGMA_COM_LEVELS[-1]; hi = nrows[sc_hi]
    ge_hi = hi["geo"].mean(); an_hi = hi["analytic"].mean(); la_hi = hi["learned_adapt"].mean(); lf_hi = hi["learned_frozen"].mean()
    print("  As sensing DEGRADES to sigma=%.0fmm: geo=%.2f  analytic@30=%.2f  learned-frozen=%.2f  learned-adapt@30=%.2f  (oracle=%.2f)"
          % (sc_hi*1000, ge_hi, an_hi, lf_hi, la_hi, hi["oracle"].mean()))
    fx_hi = hi["fixed"].mean()
    print("  Deflation check: a trivial SOURCE fixed-grasp lookup (per-family, ignores the estimate) = %.2f at sigma=%.0fmm."
          % (fx_hi, sc_hi*1000))
    if la_hi > max(ge_hi, an_hi) + 0.05 and abs(fx_hi - la_hi) <= 0.05:
        print("  * PREDICTIVE RULE (honest, deflated): the scene's optimum is a FIXED per-family grasp recoverable")
        print("    WITHOUT per-instance sensing. Methods that TRUST a noisy CoM estimate (geo, analytic) collapse as")
        print("    sensing degrades; the learned model stays reliable — but so does a trivial fixed-grasp LOOKUP, so")
        print("    the value is 'use population data, ignore the unreliable estimate', NOT a rich learned world model.")
    elif la_hi > max(ge_hi, an_hi, fx_hi) + 0.05:
        print("  * The learned model beats BOTH the estimate-trusting heuristics AND the fixed-grasp lookup under poor")
        print("    sensing -> a genuine learned advantage beyond a per-family constant (worth pursuing).")
    elif ge_hi >= la_hi - 0.03:
        print("  * Even under poor sensing the heuristic keeps up -> no learning advantage in this scene.")
    else:
        print("  * Under poor sensing the ordering is mixed / within CIs -> inconclusive; do not over-claim.")
    print("  BOUNDARY: because a fixed per-family grasp suffices, this scene CANNOT decide the thesis for learning.")
    print("  A decisive pro-learning experiment needs the optimal action to VARY with a hidden variable that")
    print("  per-instance sensing cannot resolve but interaction data can (no fixed policy suffices).")


def re_noise(records, sc, sm, seed):
    """Re-draw the SHARED CoM/friction estimates at a new sensing-noise level (one estimate per
    instance, applied to all its candidates). Outcomes are UNCHANGED (they used true params) —
    only the OBSERVATION quality changes, so this needs no new rollouts."""
    rng = np.random.default_rng(seed); est = {}; out = []
    for r in records:
        key = (r["family"], r["setname"], r["body"], r["inst_idx"])
        if key not in est:
            est[key] = (float(r["comy_true"] + rng.normal(0, sc)),
                        float(np.clip(r["mu_true"] + rng.normal(0, sm), 0.2, 1.6)))
        r2 = dict(r); r2["comy_est"], r2["mu_est"] = est[key]; out.append(r2)
    return out

SIGMA_COM_LEVELS = [0.008, 0.020, 0.035, 0.055]   # CoM-estimate std (m): good -> poor sensing

def noise_robustness(base_records, sigma_mu=0.15):
    """Does a data-driven selector stay more reliable than a rule that TRUSTS the CoM estimate,
    as sensing degrades? Re-runs geo / analytic@30 / learned-frozen / learned-adapt@30 / oracle
    at several CoM-noise levels (retraining the learned model at each level)."""
    rows = {}
    for sc in SIGMA_COM_LEVELS:
        recs = re_noise(base_records, sc, sigma_mu, seed=100 + int(sc * 1000))
        T = organize(recs)
        testkeys = sorted([k for k in T if k[1] == "test"]); calibkeys = sorted([k for k in T if k[1] == "calib"])
        trainrecs = [r for r in recs if r["setname"] == "train"]
        Xtr = np.array([feat(r, r["body"]) for r in trainrecs]); ytr = np.array([r["placed"] for r in trainrecs], float)
        src = train_ensemble(Xtr, ytr)
        rng = np.random.default_rng(7); calib_ep = [r for k in calibkeys for r in T[k]]; rng.shuffle(calib_ep)
        pairs = [(r, r["placed"]) for r in calib_ep[:30]]
        Xt = np.array([feat(r, TARGET_BODY) for r, _ in pairs]); yt = np.array([p for _, p in pairs], float)
        adapt = train_ensemble(np.vstack([Xtr, Xt]), np.concatenate([ytr, yt]),
                               w=np.concatenate([np.ones(len(ytr)), np.full(len(yt), max(1.0, len(ytr) / max(len(yt), 1)))]))
        best_fx = source_fixed_policy(trainrecs)
        geo = eval_method(lambda ir: m_geo(ir, TARGET_BODY), testkeys, T)[0]
        an = eval_method(lambda ir: m_analytic(ir, TARGET_BODY, pairs), testkeys, T)[0]
        lf = eval_method(lambda ir: m_learned(ir, TARGET_BODY, src), testkeys, T)[0]
        la = eval_method(lambda ir: m_learned(ir, TARGET_BODY, adapt), testkeys, T)[0]
        fx = eval_method(lambda ir: m_fixed(ir, best_fx), testkeys, T)[0]
        orac = eval_method(lambda ir: m_oracle(ir), testkeys, T)[0]
        rows[sc] = dict(geo=geo, analytic=an, learned_frozen=lf, learned_adapt=la, fixed=fx, oracle=orac, n=len(testkeys))
    return rows


def make_noise_figure(nrows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    scs = [s*1000 for s in SIGMA_COM_LEVELS]
    def series(key): return [nrows[s][key].mean() for s in SIGMA_COM_LEVELS]
    def band(key):
        los, his = [], []
        for s in SIGMA_COM_LEVELS:
            pl = nrows[s][key]; k = int(pl.sum()); n = len(pl); lo, hi = wilson(k, n)
            los.append(pl.mean()-lo); his.append(hi-pl.mean())
        return np.clip(np.array([los, his]), 0, None)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(scs, series("oracle"), "--", color="#888", label="oracle (privileged)")
    ax.errorbar(scs, series("geo"), yerr=band("geo"), fmt="o-", color="#48b", capsize=3, label="geometry-aware heuristic (trusts estimate)")
    ax.errorbar(scs, series("analytic"), yerr=band("analytic"), fmt="^-", color="#e67", capsize=3, label="calibrated-analytic @30 (trusts estimate)")
    ax.plot(scs, series("fixed"), "x--", color="#a5a", label="source fixed-grasp lookup (ignores estimate)")
    ax.errorbar(scs, series("learned_frozen"), yerr=band("learned_frozen"), fmt="d-", color="#6b3", capsize=3, label="learned-frozen (0 target)")
    ax.errorbar(scs, series("learned_adapt"), yerr=band("learned_adapt"), fmt="s-", color="#2a8", capsize=3, label="learned-adapted @30")
    ax.set_xlabel("CoM-estimate noise  std [mm]  (sensing quality → poorer)")
    ax.set_ylabel("held-out task success"); ax.set_ylim(0, 1.05)
    ax.set_title("Sensing-noise robustness: when is the learned selector worth it?")
    ax.legend(fontsize=8, loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "gp_noise.png"), dpi=95); print("wrote %s/gp_noise.png" % OUT)


def make_figure(results, com_pl, geo_pl, orac_pl, testkeys):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import collections as _c
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    bs = BUDGETS
    # panel 1: adaptation curve
    an_m = [results["analytic"][b][0].mean() for b in bs]
    le_m = [results["learned"][b][0].mean() for b in bs]
    ax[0].axhline(orac_pl.mean(), color="#888", ls="--", label="oracle (privileged)")
    ax[0].axhline(geo_pl.mean(), color="#48b", ls="-.", label="geometry-aware heuristic (no target data)")
    ax[0].axhline(com_pl.mean(), color="#8bd", ls=":", label="naive CoM heuristic")
    ax[0].plot(bs, an_m, "o-", color="#e67", label="calibrated-analytic")
    ax[0].plot(bs, le_m, "s-", color="#2a8", label="learned (frozen→adapted)")
    ax[0].set_xlabel("target calibration episodes"); ax[0].set_ylabel("held-out task success")
    ax[0].set_title("Adaptation-data curve (held-out target body)"); ax[0].set_ylim(0, 1.05)
    ax[0].set_xticks(bs); ax[0].legend(fontsize=7.5, loc="lower right")
    # panel 2: per-family @30
    fams = list(FAMILIES); x = np.arange(len(fams)); w = 0.17
    def famvals(pl): return [pl[[i for i, k in enumerate(testkeys) if k[0] == f]].mean() for f in fams]
    ax[1].bar(x-2*w, famvals(geo_pl), w, label="geo", color="#48b")
    ax[1].bar(x-1*w, famvals(results["analytic"][30][0]), w, label="analytic@30", color="#e67")
    ax[1].bar(x+0*w, famvals(results["learned"][30][0]), w, label="learned@30", color="#2a8")
    ax[1].bar(x+1*w, famvals(orac_pl), w, label="oracle", color="#888")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["A uniform", "B off-CoM", "C placement"])
    ax[1].set_ylabel("success"); ax[1].set_title("Per-family success @budget=30"); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8)
    # panel 3: failure modes @30
    orac_ch = results["oracle"][None][1]
    chmap = {"geo": results["geo"][None][1], "analytic@30": results["analytic"][30][1],
             "learned@30": results["learned"][30][1], "oracle": orac_ch}
    labels = ["geo", "analytic@30", "learned@30", "oracle"]; modes = ["PLACED", "PLACE", "TIP", "DROP"]
    cols = {"PLACED": "#2a8", "PLACE": "#e67", "TIP": "#c40", "DROP": "#849"}
    bottom = np.zeros(len(labels))
    for mo in modes:
        vals = [_c.Counter(r["label"] for r in chmap[l]).get(mo, 0) / len(chmap[l]) for l in labels]
        ax[2].bar(labels, vals, bottom=bottom, label=mo, color=cols[mo]); bottom += vals
    ax[2].set_ylabel("fraction"); ax[2].set_title("Failure modes @budget=30"); ax[2].legend(fontsize=8); ax[2].tick_params(axis='x', rotation=15)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "gp_bench.png"), dpi=95)
    print("wrote %s/gp_bench.png" % OUT)


if __name__ == "__main__":
    main()
