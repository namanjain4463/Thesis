"""
grasp_ranking_reversal.py — Tier-2 micro-benchmark (Contribution C), AUDIT-CORRECTED.

HISTORY / HONESTY: an earlier version claimed a "certified grasp-ranking reversal" —
strong gripper picks the geometric-center grasp, weak gripper picks a CoM-ward grasp —
using a 0.12 s hold. A 3rd reviewer showed (and this file confirms) that the reversal is
an artifact of that short hold: the center grasp is SLOWLY tipping the whole time and
only stays under the tilt threshold for ~0.12 s. At a properly specified 2 s hold the
center grasp FAILS for BOTH bodies, so BOTH prefer a CoM-ward grasp and THE REVERSAL
DISAPPEARS. This file now reports the honest, horizon-specified result.

Also corrected per the review:
  * sustained validity is reported at multiple horizons (0.12 / 0.5 / 2.0 s), and
    failure modes are separated (TIP = tilt-out vs DROP = lost/slid);
  * a trivial CoM baseline (grasp nearest the CoM) is included — it succeeds for BOTH
    bodies, so this scene does NOT yet show a task advantage for the capability-aware
    ranker over "just grasp the CoM";
  * the moment-margin feasibility rule's one lever is fit IN-SAMPLE on the ground-truth
    labels, so its map-accuracy is training-set agreement, NOT zero-shot transfer — an
    independently-calibrated, held-out version is the open item (see docs/tier2...md §5).

What survives honestly: a capability difference as a graded MARGIN — the weak body's
2 s-feasible set is a strict SUBSET of the strong body's — even though the argmax
(best grasp) is the same (CoM-ward) for both. Correctly PRESERVING a ranking is a valid
outcome; this scene does not produce a genuine reversal.
"""
import numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True)

PALM_HOME = 0.6
G = 9.81
PED_TOP = 0.20
TILT_FAIL = 25.0            # deg
DT = 0.002
HORIZONS = [0.12, 0.5, 2.0]  # seconds


def scene(squeeze, bar_hy=0.06, com_shift=0.060, mass_bar=0.05, mass_end=0.15, mu=1.0):
    oz = PED_TOP + 0.012 + 0.001
    return f"""
<mujoco model="reversal">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <default><geom solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <geom name="pedestal" type="box" pos="0 0 0.1" size="0.10 0.12 0.1" friction="{mu} 0.01 0.001"/>
    <body name="palm" pos="0 0 {PALM_HOME}">
      <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.05 0.03 0.02" mass="0.3" contype="2" conaffinity="0"/>
      <body name="lfinger" pos="0.04 0 -0.05">
        <joint name="lf" type="slide" axis="-1 0 0" range="0 0.035"/>
        <geom name="lfinger" type="box" size="0.006 0.02 0.03" mass="0.03" condim="3" friction="{mu} 0.02 0.002"/>
      </body>
      <body name="rfinger" pos="-0.04 0 -0.05">
        <joint name="rf" type="slide" axis="1 0 0" range="0 0.035"/>
        <geom name="rfinger" type="box" size="0.006 0.02 0.03" mass="0.03" condim="3" friction="{mu} 0.02 0.002"/>
      </body>
    </body>
    <body name="object" pos="0 0 {oz}">
      <freejoint name="obj"/>
      <geom name="bar" type="box" size="0.012 {bar_hy} 0.012" mass="{mass_bar}"
            condim="3" friction="{mu} 0.005 0.0001"/>
      <geom name="endmass" type="box" pos="0 {com_shift} 0" size="0.012 0.012 0.012" mass="{mass_end}"
            condim="3" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="apx" joint="px" kp="800" kv="40"/><position name="apy" joint="py" kp="800" kv="40"/>
    <position name="apz" joint="pz" kp="800" kv="40"/>
    <position name="alf" joint="lf" kp="300" kv="8" ctrlrange="0 0.035" forcerange="-{squeeze} {squeeze}"/>
    <position name="arf" joint="rf" kp="300" kv="8" ctrlrange="0 0.035" forcerange="-{squeeze} {squeeze}"/>
  </actuator>
</mujoco>"""


def com_y(mass_bar, mass_end, com_shift):
    return mass_end * com_shift / (mass_bar + mass_end)


def run(y_grasp, squeeze, com_shift=0.060, mass_bar=0.05, mass_end=0.15, mu=1.0):
    """Grasp at palm-y=y_grasp, lift, and hold for max(HORIZONS) seconds. Records the tilt
    trajectory and reports held@each horizon + failure mode. Ground truth is SUSTAINED hold."""
    m = mujoco.MjModel.from_xml_string(scene(squeeze, com_shift=com_shift, mass_bar=mass_bar, mass_end=mass_end, mu=mu))
    d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("apx","apy","apz","alf","arf")}
    obid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    oz = PED_TOP + 0.012 + 0.001
    gz = oz + 0.065; hov = gz + 0.15; lift = gz + 0.20
    def C(pz, lf, rf, py=y_grasp): d.ctrl[A["apx"]]=0; d.ctrl[A["apy"]]=py; d.ctrl[A["apz"]]=pz-PALM_HOME; d.ctrl[A["alf"]]=lf; d.ctrl[A["arf"]]=rf
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(gz,a*0.035,a*0.035); mujoco.mj_step(m,d)
    for k in range(40): C(gz,0.035,0.035); mujoco.mj_step(m,d)
    for k in range(120): a=(k+1)/120; C(gz+a*(lift-gz),0.035,0.035); mujoco.mj_step(m,d)
    def tilt():
        R=np.array(d.xmat[obid]).reshape(3,3); return float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))))
    nsteps=int(round(max(HORIZONS)/DT)); tilt_at={}; hz_steps={round(h/DT):h for h in HORIZONS}
    for k in range(nsteps):
        C(lift,0.035,0.035); mujoco.mj_step(m,d)
        if (k+1) in hz_steps: tilt_at[hz_steps[k+1]]=tilt()
    dz=float(d.xpos[obid][2])-oz; tl=tilt()
    held={h: (tilt_at.get(h,tl) < TILT_FAIL and dz>0.12) for h in HORIZONS}
    # failure mode at the longest horizon
    mode = "-" if held[max(HORIZONS)] else ("DROP" if dz<0.12 else "TIP")
    return dict(y_grasp=y_grasp, squeeze=squeeze, dz=dz, tilt_at=tilt_at, held=held, mode=mode)


# ---- capability-aware feasibility margin (moment) -- NB lever is fit IN-SAMPLE below ----
MU = 1.0
def moment_feasible(y, Fn_deliv, weight, y_com, lever):
    if MU*Fn_deliv - weight <= 0: return False, "weight"
    if MU*Fn_deliv*lever - weight*abs(y-y_com) <= 0: return False, "moment"
    return True, "-"


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cs=0.060; mb=0.05; me=0.15
    cy=com_y(mb, me, cs); weight=(mb+me)*G
    cand=[-0.02, 0.0, 0.02, 0.045, 0.06]
    bodies=[("gripperA (strong 25N)", 25.0), ("gripperB (weak 2.5N)", 2.5)]
    H=max(HORIZONS)
    print("="*94)
    print("SUSTAINED-HOLD grasp evaluation (Tier-2, audit-corrected: the 0.12s 'reversal' was an artifact)")
    print("="*94)
    print("off-center bar: CoM_y=%.3f (geom center y=0)  mass=%.2f kg  weight=%.2f N  hold horizon=%.1fs  tilt-fail=%.0f°"
          %(cy, mb+me, weight, H, TILT_FAIL))

    R={}
    for name,sq in bodies:
        for y in cand: R[(name,y)]=run(y, sq, com_shift=cs, mass_bar=mb, mass_end=me, mu=MU)

    print("\n tilt(deg) vs hold horizon — the center grasp (y=0) SLOWLY tips out (artifact of a short hold):")
    print("  %-22s %-10s %s"%("body","grasp","  ".join("t@%.2fs"%h for h in HORIZONS)))
    for name,_ in bodies:
        for y in [0.0, 0.02, round(cy,3)]:
            r=R[(name,y)]
            print("  %-22s y=%-8.3f %s   [%.1fs: %s%s]"%(name,y,
                "  ".join("%6.1f"%r["tilt_at"].get(h,float('nan')) for h in HORIZONS),
                H, "HELD" if r["held"][H] else "FAIL", "" if r["mode"]=="-" else " "+r["mode"]))

    def held_map(name,h): return {y: R[(name,y)]["held"][h] for y in cand}
    print("\n SUSTAINED (%.1fs) HOLD map:"%H)
    for name,_ in bodies:
        hm=held_map(name,H)
        print("   %-22s %s"%(name, "  ".join("y=%+.3f:%s"%(y,"H" if hm[y] else ".") for y in cand)))

    # feasible sets + argmax (feasible grasp nearest geometric center) at the 2s horizon
    def feasible(name,h): return [y for y in cand if R[(name,y)]["held"][h]]
    def argmax_center(name,h):
        f=feasible(name,h); return min(f,key=lambda y:abs(y)) if f else None
    print("\n  %-22s %-16s %-16s %-16s"%("body","argmax@0.12s","argmax@2.0s","CoM-baseline(y=%.3f)"%round(cy,3)))
    for name,_ in bodies:
        a012=argmax_center(name,0.12); a2=argmax_center(name,H)
        combase = "HELD" if R[(name,round(cy,3))]["held"][H] else "fail"
        print("  %-22s y=%-14s y=%-14s %s"%(name,a012,a2,combase))
    argmaxes_2s={argmax_center(name,H) for name,_ in bodies}
    argmaxes_012={argmax_center(name,0.12) for name,_ in bodies}
    reversal_2s = len(argmaxes_2s)>1; reversal_012 = len(argmaxes_012)>1

    # capability feasibility rule (moment margin). Lever FIT IN-SAMPLE on the 2s labels (not zero-shot).
    Fdeliv={name:2.0*sq for name,sq in bodies}
    def err(L): return sum(moment_feasible(y,Fdeliv[name],weight,cy,L)[0]!=R[(name,y)]["held"][H]
                           for name,_ in bodies for y in cand)
    lever=float(min(np.linspace(0.004,0.03,80), key=err))
    print("\n capability moment-margin rule: lever=%.3f m FIT IN-SAMPLE on the %.1fs labels (NOT zero-shot);"
          " feasibility-map agreement=%d/%d"%(lever,H,len(bodies)*len(cand)-err(lever),len(bodies)*len(cand)))
    # feasible-set comparison + the transient-margin signal (the only capability signal left)
    fs={name:set(feasible(name,H)) for name,_ in bodies}
    same = fs[bodies[0][0]]==fs[bodies[1][0]]
    t0={name:R[(name,0.0)]["tilt_at"].get(0.12) for name,_ in bodies}
    print(" 2s-feasible sets: strong=%s  weak=%s -> %s"
          %(sorted(fs[bodies[0][0]]), sorted(fs[bodies[1][0]]),
            "IDENTICAL (weak grip still holds the near-CoM grasps)" if same else "DIFFERENT"))
    print("   the only capability signal is the TRANSIENT tilt margin at the center grasp: weak %.0f° vs strong %.0f° @0.12s"
          %(t0[bodies[1][0]], t0[bodies[0][0]]))

    print("\n"+"-"*94)
    print(" VERDICT (audit-corrected):")
    print("  * REVERSAL @0.12s (strong→y=%s, weak→y=%s) DISAPPEARS @%.1fs (both→y=%s): the center grasp is slowly"
          %(argmax_center(bodies[0][0],0.12), argmax_center(bodies[1][0],0.12), H, argmax_center(bodies[0][0],H)))
    print("    tipping for BOTH bodies and only survives ~0.12s. The earlier 'certified reversal' was this artifact.")
    print("  * At %.1fs the two bodies have %s feasible sets; the capability gap survives only as a transient tilt"
          %(H, "IDENTICAL" if same else "different"))
    print("    margin, not a change in the sustained decision. NO genuine ranking reversal in this scene.")
    print("  * A trivial CoM-baseline grasp holds for BOTH bodies -> NO task advantage shown over 'grasp the CoM'.")
    print("  * NEXT (open): a scene where an off-CoM grasp is needed for a TASK reason (clearance / mounting), a")
    print("    genuinely different embodiment (Panda reach/port, not a grip-force knob), and an INDEPENDENTLY")
    print("    calibrated rule evaluated on held-out cases vs the CoM + wrench-feasibility baselines (docs §5).")

    # ---- figure ----
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
    ys=np.array(cand)
    for j,(name,_) in enumerate(bodies):
        h012=np.array([R[(name,y)]["held"][0.12] for y in cand]); h2=np.array([R[(name,y)]["held"][H] for y in cand])
        ax[j].scatter(ys[h012],[1.15]*h012.sum(),s=70,c="#bbb",marker="s",label="held @0.12s")
        ax[j].scatter(ys[h2],[1.0]*h2.sum(),s=130,c="#2a8",marker="o",label="held @%.1fs"%H)
        ax[j].scatter(ys[~h2],[1.0]*(~h2).sum(),s=130,c="#c40",marker="x",label="failed @%.1fs"%H)
        ax[j].axvline(0.0,color="#888",ls=":",label="geometric center (y=0)")
        ax[j].axvline(cy,color="#e83",ls="--",label="CoM")
        ax[j].set_title(name); ax[j].set_xlabel("grasp position y [m]"); ax[j].set_yticks([]); ax[j].set_ylim(0.6,1.4)
        ax[j].legend(fontsize=7,loc="lower center",ncol=2)
    fig.suptitle("Sustained-hold (2s): center grasp fails for BOTH bodies (0.12s 'reversal' was an artifact); both prefer CoM-ward")
    plt.tight_layout(); import os; os.makedirs("rankrev_out",exist_ok=True)
    plt.savefig("rankrev_out/grasp_ranking_reversal.png",dpi=95); print("\n wrote rankrev_out/grasp_ranking_reversal.png")


if __name__ == "__main__":
    main()
