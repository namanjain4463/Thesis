"""Batch analysis of m2_out/trials.csv: what drives outcome, and how the
Delassus W behaves. Prints established facts + writes a multi-panel figure."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("m2_out/trials.csv")))
def col(k, f=float): return np.array([f(r[k]) for r in rows])
out = np.array([r["outcome"] for r in rows])
xoff = np.abs(col("xoff_mm")); zoff = col("zoff_mm"); mu = col("mu"); force = col("force_N")
d = col("d_mm"); rise = col("rise_final_mm")
Wd = col("W_diag_mean"); Woff = col("W_offdiag_max"); Fn = col("Fn_max"); dpm = col("dp_step_max_mm")
lifted = (out == "stable_lift")
N = len(rows)
print("N=%d  outcomes:" % N, {o: int((out == o).sum()) for o in sorted(set(out))})

def rate_by_bin(x, mask_ok, edges, name):
    print("\n success rate vs %s:" % name)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum():
            print("   [%6.1f,%6.1f): n=%3d  lift=%.0f%%" % (lo, hi, m.sum(), 100 * mask_ok[m].mean()))

rate_by_bin(xoff, lifted, [0, 3, 6, 9, 12, 15, 18.1], "|xoff| mm (alignment)")
rate_by_bin(zoff, lifted, [-6, 0, 6, 12, 18, 22.1], "zoff mm (vertical)")
rate_by_bin(mu, lifted, [0.1, 0.4, 0.7, 1.0, 1.4001], "mu (friction)")
rate_by_bin(force, lifted, [0.5, 2, 5, 12, 40.001], "squeeze force N")

# point-biserial-ish correlations of each feature with lift
def corr(x): return float(np.corrcoef(x, lifted.astype(float))[0, 1])
print("\n corr(feature, lift):  |xoff|=%.2f  zoff=%.2f  mu=%.2f  logF=%.2f  d=%.2f"
      % (corr(xoff), corr(zoff), corr(mu), corr(np.log10(force)), corr(d)))

# W physics: diag ~ inverse effective inertia -> should FALL as object gets heavier/bigger
mass = col("mass")
print("\n W_diag_mean vs object mass  corr=%.2f (expect <0: heavier -> lower inverse-inertia)"
      % float(np.corrcoef(mass, Wd)[0, 1]))
print(" W_offdiag_max: median=%.1f  (nonzero coupling between contacts on the object)"
      % np.median(Woff[Woff > 0]))
print(" among lifts: median lateral dp_step=%.3f mm ; among no-lift: %.3f mm"
      % (np.median(dpm[lifted]), np.median(dpm[~lifted])))

# ---- figure ----
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
oc = {"stable_lift": "#2f8f4f", "grip_no_lift": "#c34317", "lift_then_drop": "#e0a020", "no_contact": "#7a7a7a"}

# (0,0) outcome counts
labels, counts = zip(*sorted([(o, int((out == o).sum())) for o in set(out)], key=lambda t: -t[1]))
ax[0, 0].bar(range(len(labels)), counts, color=[oc.get(l, "#888") for l in labels])
ax[0, 0].set_xticks(range(len(labels))); ax[0, 0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax[0, 0].set_title("outcome distribution (N=%d)" % N)

# (0,1) success vs |xoff|
e = np.array([0, 3, 6, 9, 12, 15, 18.1]); mids = 0.5 * (e[:-1] + e[1:])
r = [lifted[(xoff >= lo) & (xoff < hi)].mean() if ((xoff >= lo) & (xoff < hi)).sum() else np.nan
     for lo, hi in zip(e[:-1], e[1:])]
ax[0, 1].plot(mids, np.array(r) * 100, "o-", color="#c34317"); ax[0, 1].set_ylim(0, 105)
ax[0, 1].set_xlabel("|lateral offset| (mm)"); ax[0, 1].set_ylabel("lift %"); ax[0, 1].set_title("alignment drives outcome")

# (0,2) success vs mu
e2 = np.array([0.1, 0.4, 0.7, 1.0, 1.4001]); m2 = 0.5 * (e2[:-1] + e2[1:])
r2 = [lifted[(mu >= lo) & (mu < hi)].mean() if ((mu >= lo) & (mu < hi)).sum() else np.nan
      for lo, hi in zip(e2[:-1], e2[1:])]
ax[0, 2].plot(m2, np.array(r2) * 100, "s-", color="#2f6f9f"); ax[0, 2].set_ylim(0, 105)
ax[0, 2].set_xlabel("friction mu"); ax[0, 2].set_ylabel("lift %"); ax[0, 2].set_title("friction DOES matter at low mu")

# (1,0) W_diag vs mass, colored by lift
ax[1, 0].scatter(mass[lifted], Wd[lifted], s=10, c="#2f8f4f", label="lift", alpha=.6)
ax[1, 0].scatter(mass[~lifted], Wd[~lifted], s=10, c="#c34317", label="no lift", alpha=.6)
ax[1, 0].set_xlabel("object mass (kg)"); ax[1, 0].set_ylabel("mean W diagonal"); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title("W diag ~ inverse inertia (falls w/ mass)")

# (1,1) W offdiag distribution
ax[1, 1].hist(Woff[Woff > 0], bins=30, color="#7a4fa0")
ax[1, 1].set_xlabel("max off-diagonal W (contact coupling)"); ax[1, 1].set_ylabel("count")
ax[1, 1].set_title("off-diagonal W coupling (the compositional term)")

# (1,2) lateral dp per outcome
data = [dpm[out == o] for o in labels]
ax[1, 2].boxplot(data, tick_labels=labels, showfliers=False)
ax[1, 2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax[1, 2].set_ylabel("max per-step |dp| (mm)"); ax[1, 2].set_title("object motion signature by outcome")

fig.suptitle("M2 floating-gripper contact-transition dataset — batch analysis (N=%d)" % N, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("batch_analysis.png", dpi=115); print("\nwrote batch_analysis.png")
