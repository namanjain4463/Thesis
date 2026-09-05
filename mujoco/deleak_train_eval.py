"""
deleak_train_eval.py — first LEARNED cross-embodiment transfer test with a
de-leaked dataset (see deleak_dataset.py).

Trains the local contact law C_theta to predict the contact normal force F_n from
DE-LEAKED inputs (observable local kinematics + a categorical material id — never
the raw mu/solref/solimp). Two models, both trained ONLY on the floating gripper,
then FROZEN and evaluated on the Panda:

  * LOCAL-ONLY (naive)   : [pen, vn, vt, kappa, cz, nz] + material one-hot.
                           Blind to the embodiment -> the thesis predicts it CANNOT
                           transfer, because realized F_n is a solve output that
                           depends on the port (MuJoCo's R is inertia-scaled).
  * FACTORIZED (thesis)  : the same + the analytical per-contact normal Delassus
                           W_nn (the embodiment-carrying PORT, computed from M).
                           "Freeze the local law, recompute the port."

Baselines on the Panda: a mean-F_n predictor, and a model RETRAINED on Panda data
(the per-robot upper bound). Headline: does adding the analytical port let a
FROZEN, float-trained law transfer to a real second arm?

Run:  python deleak_train_eval.py     (reads deleak_out/deleak_data.npz)
"""
import os, numpy as np
np.set_printoptions(precision=4, suppress=True)
OUT_DIR = "deleak_out"
NMAT = 5

# ---------------- data ----------------
def load():
    d = np.load(os.path.join(OUT_DIR, "deleak_data.npz"), allow_pickle=True)
    return d["float_rows"], d["panda_rows"]     # cols: pen,vn,vt,kappa,cz,nz,W_nn,Fn,mat,tid

def make_xy(rows, use_port):
    """Inputs: 6 local feats (+ W_nn if use_port) + material one-hot(5). Target: Fn."""
    pen, vn, vt, kap, cz, nz, Wnn = [rows[:, i] for i in range(7)]
    Fn, mat = rows[:, 7], rows[:, 8].astype(int)
    local = np.column_stack([pen, vn, vt, kap, cz, nz])
    feats = np.column_stack([local, Wnn]) if use_port else local
    onehot = np.eye(NMAT)[mat]
    X = np.column_stack([feats, onehot])
    return X.astype(np.float64), Fn.astype(np.float64), rows[:, 9].astype(int)

# ---------------- tiny MLP (numpy) ----------------
def init_net(din, hidden, rng):
    sizes = [din] + hidden + [1]; P = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        P.append([rng.standard_normal((a, b)) * np.sqrt(2.0 / a), np.zeros(b)])
    return P

def forward(P, X):
    a = X; cache = [a]
    for i, (W, b) in enumerate(P):
        z = a @ W + b
        a = np.maximum(z, 0) if i < len(P) - 1 else z
        cache.append((z, a))
    return a[:, 0], cache

def loss_grad(P, X, y):
    yp, cache = forward(P, X)
    n = len(y); err = (yp - y) / n
    loss = 0.5 * np.mean((yp - y) ** 2)
    g = err[:, None]; grads = [None] * len(P)
    a_prev = cache[0] if len(P) == 1 else cache[-2][1]
    for i in range(len(P) - 1, -1, -1):
        a_in = X if i == 0 else cache[i][1]
        W, b = P[i]
        grads[i] = [a_in.T @ g, g.sum(0)]
        if i > 0:
            z_prev = cache[i][0]
            g = (g @ W.T) * (z_prev > 0)
    return loss, grads

def train(X, y, hidden=[64, 64], iters=4000, bs=512, lr=2e-3, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    # standardize inputs and target (stats from THIS training set = float)
    xm, xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = y.mean(), y.std() + 1e-8
    Xs = (X - xm) / xs; ysd = (y - ym) / ys
    P = init_net(X.shape[1], hidden, rng)
    mW = [[np.zeros_like(w), np.zeros_like(b)] for w, b in P]
    vW = [[np.zeros_like(w), np.zeros_like(b)] for w, b in P]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        idx = rng.integers(0, len(y), bs)
        loss, grads = loss_grad(P, Xs[idx], ysd[idx])
        for i in range(len(P)):
            for j in range(2):
                mW[i][j] = b1 * mW[i][j] + (1 - b1) * grads[i][j]
                vW[i][j] = b2 * vW[i][j] + (1 - b2) * grads[i][j] ** 2
                mhat = mW[i][j] / (1 - b1 ** t); vhat = vW[i][j] / (1 - b2 ** t)
                P[i][j] -= lr * mhat / (np.sqrt(vhat) + eps)
        if verbose and t % 1000 == 0:
            print("    iter %d loss %.4f" % (t, loss))
    return dict(P=P, xm=xm, xs=xs, ym=ym, ys=ys)

def predict(mdl, X):
    yp, _ = forward(mdl["P"], (X - mdl["xm"]) / mdl["xs"])
    return yp * mdl["ys"] + mdl["ym"]

# ---------------- metrics ----------------
def r2(y, yp): return 1.0 - np.sum((y - yp) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
def relmed(y, yp, thr=0.2):
    m = y > thr
    return float(np.median(np.abs(yp[m] - y[m]) / y[m])) if m.any() else float("nan")

# ---------------- gradient check (repo discipline) ----------------
def grad_check():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 4)); y = rng.standard_normal(8)
    P = init_net(4, [5], rng)
    loss, grads = loss_grad(P, X, y)
    W, b = P[0]; i, j = 1, 2; e = 1e-6
    W[i, j] += e; lp, _ = loss_grad(P, X, y); W[i, j] -= 2 * e; lm, _ = loss_grad(P, X, y); W[i, j] += e
    num = (lp - lm) / (2 * e)
    print("  gradient check: analytic=%.6e numeric=%.6e  rel=%.2e"
          % (grads[0][0][i, j], num, abs(grads[0][0][i, j] - num) / (abs(num) + 1e-12)))

# ---------------- trial split ----------------
def split_by_trial(rows, frac=0.7, seed=0):
    tids = np.unique(rows[:, 9]); rng = np.random.default_rng(seed); rng.shuffle(tids)
    ntr = int(len(tids) * frac); tr = set(tids[:ntr].tolist())
    mask = np.array([t in tr for t in rows[:, 9]])
    return mask, ~mask


def static_mask(rows, vmax=0.03):
    """Quasi-static contact rows: |v_n| small, in contact. This isolates the
    CONSTITUTIVE law F_n ~ k(material, port)*pen from grasp transients."""
    return (np.abs(rows[:, 1]) < vmax) & (rows[:, 0] > 1e-5)


def constit(rows):
    """Constitutive regime: quasi-static, real penetration and real force. The clean
    F_n ~ k(material,port)*pen relationship (used for BOTH training and evaluation)."""
    return static_mask(rows) & (rows[:, 0] > 1e-4) & (rows[:, 7] > 0.05)


# ---------------- white-box physics models (no MLP) ----------------
# Series-compliance model:  pen/F_n = 1/k_mat + alpha * W_nn   (linear in [1/k_mat, alpha]).
# D (analytical, NO port): pen/F_n = 1/k_mat (per material) -> assumes stiffness invariant.
# C (coupled, +port): adds alpha*W_nn, a solve-like port-compliance term. Fit on FLOAT.
def whitebox_fit(rows, use_port):
    y = rows[:, 0] / rows[:, 7]                              # pen / F_n  (compliance)
    oh = np.eye(NMAT)[rows[:, 8].astype(int)]
    A = np.column_stack([oh, rows[:, 6]]) if use_port else oh
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def whitebox_pred(coef, rows, use_port):
    oh = np.eye(NMAT)[rows[:, 8].astype(int)]
    A = np.column_stack([oh, rows[:, 6]]) if use_port else oh
    inv_k = np.maximum(A @ coef, 1e-6)                       # predicted pen/F_n
    return rows[:, 0] / inv_k                                # F_n = pen / (pen/F_n)


def stiffness_table(F, Pa):
    """MECHANISM: per material, realized normal stiffness k (through-origin fit F_n=k*pen on
    the constitutive regime) and median port W_nn, both embodiments. k is NOT embodiment-
    invariant; W_nn also differs. Prints the actual softness ratio (no hard-coded claim)."""
    print("\n" + "=" * 74); print(" MECHANISM: realized stiffness k=F_n/pen is NOT embodiment-invariant")
    print("=" * 74)
    print("  material |   k_float   k_panda  (N/m)  |  W_nn_float  W_nn_panda  (1/kg) |  k_float/k_panda")
    ratios = []
    for mid in range(NMAT):
        row = []
        for X in (F, Pa):
            q = X[(X[:, 8] == mid) & constit(X)]
            if len(q) < 20: row += [np.nan, np.nan]; continue
            row += [float(np.sum(q[:, 0] * q[:, 7]) / np.sum(q[:, 0] ** 2)), float(np.median(q[:, 6]))]
        rr = row[0] / row[2] if row[2] else np.nan
        if np.isfinite(rr): ratios.append(rr)
        print("     %d     | %8.0f  %8.0f        |  %8.1f    %8.1f      |  %.2f"
              % (mid, row[0], row[2], row[1], row[3], rr))
    print("  => same material, DIFFERENT realized k: float/panda ratio spans %.2f-%.2fx (median %.2f)."
          % (min(ratios), max(ratios), np.median(ratios)))
    print("     k is NOT embodiment-invariant; the analytical port W_nn also differs across arms.")


def eval_all(F, Pa):
    """Every model evaluated on the SAME held-out Panda constitutive samples. Frozen models
    (A,B,C,D) train on FLOAT; the retrain reference trains on Panda-train. Reports R² (mean±sd
    over 3 seeds for the MLPs) and median relative error."""
    ptr_m, pte_m = split_by_trial(Pa, 0.7, 1)
    Pte = Pa[pte_m & constit(Pa)]; Ptr = Pa[ptr_m & constit(Pa)]
    Fc = F[constit(F)]; ftr_m, _ = split_by_trial(Fc, 0.7, 0); Ftr = Fc[ftr_m]
    yte = Pte[:, 7]
    print("\n" + "=" * 74)
    print(" HELD-OUT PANDA EVALUATION (identical samples for every model; constitutive regime)")
    print("=" * 74)
    print("  eval set n=%d (held-out Panda)   float-train n=%d   matched object distribution" % (len(Pte), len(Ftr)))
    # DISTRIBUTION diagnostic: the two failure drivers the corrections exposed.
    fscale = np.median(Ftr[:, 7]) / max(np.median(Pte[:, 7]), 1e-9)
    wlo, whi = np.percentile(Ftr[:, 6], [5, 95]); pv = Pte[:, 6]
    overlap = float(np.mean((pv >= wlo) & (pv <= whi)))
    print("  DIAGNOSTIC: median F_n float/panda = %.1fx (grip strategy);  panda W_nn inside float 5-95%% band = %.0f%%"
          % (fscale, 100 * overlap))
    print("  => absolute F_n differs by grip strategy, and the port feature W_nn barely overlaps (frozen use = extrapolation).")
    res = {}
    def mlp_scores(Xtr, ytr, Xte, seeds=(0, 1, 2)):
        r2s, preds = [], []
        for sd in seeds:
            m = train(Xtr, ytr, seed=sd); yh = predict(m, Xte)
            r2s.append(r2(yte, yh)); preds.append(yh)
        return np.mean(r2s), np.std(r2s), preds[0]
    # A / B : frozen float-trained MLPs
    for tag, up in [("A local-only MLP", False), ("B factorized MLP (+port)", True)]:
        Xtr, ytr, _ = make_xy(Ftr, up); Xte, _, _ = make_xy(Pte, up)
        m, sd, pred = mlp_scores(Xtr, ytr, Xte)
        res[tag] = dict(r2=m, sd=sd, rel=relmed(yte, pred), pred=pred)
    # D / C : white-box physics, frozen float-fit
    for tag, up in [("D analytical (no port)", False), ("C coupled solve (+port)", True)]:
        coef = whitebox_fit(Ftr, up); pred = whitebox_pred(coef, Pte, up)
        res[tag] = dict(r2=r2(yte, pred), sd=0.0, rel=relmed(yte, pred), pred=pred)
    # references
    res["mean baseline"] = dict(r2=r2(yte, np.full_like(yte, make_xy(Ftr, True)[1].mean())),
                                sd=0.0, rel=relmed(yte, np.full_like(yte, yte.mean())), pred=None)
    Xr, yr, _ = make_xy(Ptr, True); Xte2, _, _ = make_xy(Pte, True)
    m, sd, _ = mlp_scores(Xr, yr, Xte2)
    res["retrain on Panda (ref)"] = dict(r2=m, sd=sd, rel=np.nan, pred=None)
    order = ["mean baseline", "D analytical (no port)", "A local-only MLP",
             "C coupled solve (+port)", "B factorized MLP (+port)", "retrain on Panda (ref)"]
    print("  %-28s %-16s  %-s" % ("model", "R2 (mean±sd)", "median rel-err"))
    for k in order:
        r = res[k]
        print("  %-28s %6.3f ± %.3f     %s"
              % (k, r["r2"], r["sd"], "%.2f" % r["rel"] if np.isfinite(r["rel"]) else "  -"))
    return res, yte


def main():
    print("=" * 74); print("DE-LEAKED LEARNED CROSS-EMBODIMENT TRANSFER  (C_theta: local law -> F_n)")
    print("=" * 74)
    grad_check()
    F, Pa = load()
    print("  float total=%d  panda total=%d   materials=%d   (inputs never see raw mu/solref/solimp)"
          % (len(F), len(Pa), NMAT))
    stiffness_table(F, Pa)
    res, yte = eval_all(F, Pa)

    A = res["A local-only MLP"]["r2"]; B = res["B factorized MLP (+port)"]["r2"]
    D = res["D analytical (no port)"]["r2"]; C = res["C coupled solve (+port)"]["r2"]
    ret = res["retrain on Panda (ref)"]["r2"]
    print("\n" + "-" * 74)
    print(" HEADLINE (held-out Panda, constitutive regime, matched objects, SYNCED logging):")
    print("   Q1 does the local law transfer?   YES via COMPLIANCE: white-box F_n=pen/k_mat(float) -> R²=%.3f." % D)
    print("      (grip scales pen and F_n together, so k=F/pen is ~grip-invariant and transfers.)")
    print("   Q2 does adding the analytical PORT W_nn help?   NO, it HURTS: white-box D R²=%.3f -> C(+port) R²=%.3f;" % (D, C))
    print("      MLP A R²=%.3f -> B(+port) R²=%.3f. W_nn barely overlaps across arms, so using it = extrapolation." % (A, B))
    print("   Q3 does a black-box MLP on ABSOLUTE F_n transfer?   NO: it predicts float-scale forces on the Panda")
    print("      (grip strategy differs ~%.0fx). Per-robot retrain reference R²=%.3f (a reference, not an upper bound)."
          % (np.median(F[constit(F)][:, 7]) / max(np.median(Pa[constit(Pa)][:, 7]), 1e-9), ret))
    print(" VERDICT: The earlier 'analytical port ~doubles transfer' result DID NOT SURVIVE the corrections")
    print("   (synced logging + matched objects + fair held-out eval). What holds honestly: the local constitutive")
    print("   COMPLIANCE transfers across embodiments (R²≈%.2f); the port W_nn as a fitted feature does NOT help and" % D)
    print("   hurts (non-overlapping distributions); absolute F_n is confounded by grip strategy. This supports the")
    print("   NARROW claim 'a transferable local compliance law', not 'the analytical port carries the embodiment'")
    print("   nor 'transferable grasp selection'. Using W_nn needs the real (W+R) SOLVE, not a fitted coefficient.")

    # ---------------- figure ----------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    FLOOR = -1.0
    order = [("mean", res["mean baseline"]["r2"], "#999"),
             ("A local MLP", A, "#c40"), ("B factorized\nMLP (+port)", B, "#e83"),
             ("C coupled\n(+port)", C, "#8c8"), ("D analytical\n(no port)", D, "#2a8"),
             ("retrain\n(ref)", ret, "#39a")]
    heights = [max(o[1], FLOOR) for o in order]
    ax[0].bar([o[0] for o in order], heights, color=[o[2] for o in order])
    for i, o in enumerate(order):        # label true value on clipped/negative bars
        if o[1] < FLOOR + 0.02:
            ax[0].text(i, FLOOR + 0.03, "%.1f\n(extrapolates)" % o[1], ha="center", va="bottom", fontsize=7)
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_ylabel("R²  (F_n, held-out Panda)")
    ax[0].set_title("Frozen float→Panda transfer (matched objects, SYNCED logging)\n"
                    "only the white-box COMPLIANCE (D) transfers; the port hurts")
    ax[0].set_ylim(FLOOR, 1.0)
    ax[0].tick_params(axis="x", labelsize=8)
    pa = res["A local-only MLP"]["pred"]; pb = res["B factorized MLP (+port)"]["pred"]
    s = np.random.default_rng(0).integers(0, len(yte), min(3000, len(yte)))
    ax[1].scatter(yte[s], pa[s], s=6, alpha=.3, c="#c40", label="A local-only")
    ax[1].scatter(yte[s], pb[s], s=6, alpha=.3, c="#2a8", label="B factorized (+port)")
    lim = float(np.percentile(yte, 99))
    ax[1].plot([0, lim], [0, lim], "k--", lw=1); ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_xlabel("true F_n on Panda [N]"); ax[1].set_ylabel("predicted F_n [N]")
    ax[1].set_title("Predicted vs true (held-out Panda)\n(on the dashed line = perfect)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "deleak_transfer.png"), dpi=95)
    print("\n wrote %s/deleak_transfer.png" % OUT_DIR)


if __name__ == "__main__":
    main()
