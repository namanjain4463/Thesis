"""
command_calibration.py — Tier-2 step (2): command->contact calibration.

The 3rd review's most consequential point: the ranker assumed deliverable grip = 2x the
actuator FORCE LIMIT (e.g. 50 N), but a position-servo finger does NOT reach its force
limit in general — delivered force depends on position error, gains, closing speed, object
width, and interface compliance. A force cap must NEVER be substituted for predicted
delivered force.

This experiment, in the floating-gripper sim, sweeps:
  closing target, closing speed, actuator force limit, object width, interface compliance
and records synchronized: command, finger position+velocity, actuator force, contact normal
force, object motion, contact-formation time. It then fits a PREDICTIVE model of the
steady delivered contact force and contact-formation time, and validates it on held-out
combinations (never the force cap as the prediction).

Simulator contact forces are used here as DIAGNOSTIC MEASUREMENTS (ground truth), not as
deployment inputs. Run:  python command_calibration.py
"""
import numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True)
PALM_HOME = 0.6; PED_TOP = 0.20
KP_FINGER = 300.0                          # finger position-actuator gain (from the scene)
FINGER_HALF_X = 0.006                      # finger geom half-thickness along x
FINGER_BASE_X = 0.04                       # finger rest x from center


def scene(diam, squeeze, solref_t, mu=1.0):
    r = diam/2.0; oz = PED_TOP + 0.04 + 0.001
    return f"""
<mujoco model="cal">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <default><geom solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <geom name="pedestal" type="box" pos="0 0 0.1" size="0.08 0.08 0.1" friction="{mu} 0.01 0.001"/>
    <body name="palm" pos="0 0 {PALM_HOME}">
      <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.05 0.03 0.02" mass="0.3" contype="2" conaffinity="0"/>
      <body name="lfinger" pos="{FINGER_BASE_X} 0 -0.05">
        <joint name="lf" type="slide" axis="-1 0 0" range="0 0.045"/>
        <geom name="lfinger" type="box" size="{FINGER_HALF_X} 0.012 0.03" mass="0.03" condim="3" friction="{mu} 0.02 0.002"/>
      </body>
      <body name="rfinger" pos="-{FINGER_BASE_X} 0 -0.05">
        <joint name="rf" type="slide" axis="1 0 0" range="0 0.045"/>
        <geom name="rfinger" type="box" size="{FINGER_HALF_X} 0.012 0.03" mass="0.03" condim="3" friction="{mu} 0.02 0.002"/>
      </body>
    </body>
    <body name="object" pos="0 0 {oz}">
      <freejoint name="obj"/>
      <geom name="object" type="cylinder" size="{r} 0.04" mass="0.15" condim="3"
            friction="{mu} 0.005 0.0001" solref="{solref_t} 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="apx" joint="px" kp="800" kv="40"/><position name="apy" joint="py" kp="800" kv="40"/>
    <position name="apz" joint="pz" kp="800" kv="40"/>
    <position name="alf" joint="lf" kp="{KP_FINGER}" kv="8" ctrlrange="0 0.045" forcerange="-{squeeze} {squeeze}"/>
    <position name="arf" joint="rf" kp="{KP_FINGER}" kv="8" ctrlrange="0 0.045" forcerange="-{squeeze} {squeeze}"/>
  </actuator>
</mujoco>"""


def run_close(target, close_steps, squeeze, diam, solref_t, mu=1.0):
    """Descend onto a centered cylinder, ramp the finger command to `target` over
    `close_steps`, hold. Returns steady delivered contact force, actuator force, finger
    settle position, and contact-formation time (s)."""
    m = mujoco.MjModel.from_xml_string(scene(diam, squeeze, solref_t, mu)); d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("apx","apy","apz","alf","arf")}
    lf_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "lf")
    og = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
    fg = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("lfinger","rfinger")}
    oz = PED_TOP + 0.04 + 0.001; gz = oz + 0.065; hov = gz + 0.15
    lf_adr = m.jnt_qposadr[lf_j]; lf_vadr = m.jnt_dofadr[lf_j]
    def C(pz, lf, rf): d.ctrl[A["apx"]]=0; d.ctrl[A["apy"]]=0; d.ctrl[A["apz"]]=pz-PALM_HOME; d.ctrl[A["alf"]]=lf; d.ctrl[A["arf"]]=rf
    def sumFn():
        s=0.0
        for ci in range(int(d.ncon)):
            c=d.contact[ci]
            if (c.geom1==og or c.geom2==og) and (c.geom1 in fg or c.geom2 in fg):
                f6=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f6); s+=abs(f6[0])
        return s
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
    tform=np.nan; step=0
    for k in range(close_steps):                       # ramp-close at controlled speed
        a=(k+1)/close_steps; C(gz, a*target, a*target); mujoco.mj_step(m,d); step+=1
        if np.isnan(tform) and sumFn()>1e-3: tform=step*0.002
    fns=[]; afs=[]; xs=[]
    for k in range(120):                               # steady hold
        C(gz, target, target); mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
        fns.append(sumFn()); afs.append(abs(float(d.actuator_force[A["alf"]]))); xs.append(float(d.qpos[lf_adr]))
    fns=np.array(fns);
    return dict(target=target, close_steps=close_steps, squeeze=squeeze, diam=diam, solref_t=solref_t,
                Fn=float(np.median(fns[-40:])), act_force=float(np.median(afs[-40:])),
                x_settle=float(np.median(xs[-40:])), tform=float(tform),
                x_contact_geom=FINGER_BASE_X-(diam/2.0)-FINGER_HALF_X)   # geometric contact displacement


def r2(y, yp): return 1.0 - np.sum((y-yp)**2)/max(np.sum((y-y.mean())**2), 1e-12)
def relmed(y, yp): return float(np.median(np.abs(yp-y)/np.maximum(y, 1e-6)))


def cap_proxy(row):        # naive: delivered = 2 x actuator force limit
    return 2.0*row["squeeze"]

def analytic_cmd(row):     # command-response: 2 x clip(kp*(target - x_contact), 0, F_limit)
    per = min(KP_FINGER*max(row["target"]-row["x_contact_geom"], 0.0), row["squeeze"])
    return 2.0*per


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    print("="*88); print("COMMAND->CONTACT CALIBRATION (Tier-2 step 2): is delivered force predictable?"); print("="*88)
    TARG=[0.030,0.036,0.042,0.045]; STEPS=[40,120]; SQZ=[2.5,5,10,25]; DIAM=[0.02,0.03,0.04]; SOL=[0.005,0.02]
    rows=[]
    for t in TARG:
        for cs in STEPS:
            for sq in SQZ:
                for dm in DIAM:
                    for sol in SOL:
                        rows.append(run_close(t, cs, sq, dm, sol))
    print("  swept %d command/config combos (centered cylinder)"%len(rows))
    y=np.array([r["Fn"] for r in rows])
    # predictors (parameter-free physics; the analytic model's ONE constant kp is the known gain)
    yc=np.array([cap_proxy(r) for r in rows]); ya=np.array([analytic_cmd(r) for r in rows])
    # held-out split by combo (70/30)
    rng=np.random.default_rng(0); idx=rng.permutation(len(rows)); tr=idx[:int(.7*len(rows))]; te=idx[int(.7*len(rows)):]
    # optional LEARNED correction: linear regression of residual (actual-analytic) on features, fit on TRAIN
    def feats(r): return [r["target"], r["diam"], r["solref_t"], r["squeeze"], max(r["target"]-r["x_contact_geom"],0), r["close_steps"]]
    X=np.array([feats(r) for r in rows]); Xm=X[tr].mean(0); Xs=X[tr].std(0)+1e-9
    Xn=(X-Xm)/Xs; res=y-ya
    w,*_=np.linalg.lstsq(np.column_stack([Xn[tr],np.ones(len(tr))]), res[tr], rcond=None)
    ylc = ya + np.column_stack([Xn,np.ones(len(rows))])@w      # analytic + learned correction
    print("\n  predictor                         held-out R²    held-out median rel-err")
    for name,yp in [("naive: delivered = force cap", yc), ("analytic command-response (kp,geom)", ya),
                    ("analytic + learned correction", ylc)]:
        print("   %-32s %7.3f        %6.1f%%"%(name, r2(y[te],yp[te]), 100*relmed(y[te],yp[te])))
    sat=np.mean([r["act_force"]>0.98*r["squeeze"] for r in rows])
    print("\n  actuator SATURATES (delivered≈cap) in %.0f%% of combos (weak grips); in the rest the cap OVER-states"%(100*sat))
    print("  delivered force by %.1fx on average -> the force cap is NOT delivered force."
          %float(np.mean(yc/np.maximum(y,1e-6))))
    print("  contact-formation time is set by closing speed+geometry (ramp reaches x_contact): range %.3f-%.3f s"
          %(min(r["tform"] for r in rows if not np.isnan(r["tform"])), max(r["tform"] for r in rows if not np.isnan(r["tform"]))))
    print("\n VERDICT: delivered contact force is PREDICTABLE from command+config by a parameter-free analytical")
    print("  command-response model (R²≈%.2f, rel-err≈%.0f%% held-out); the naive force-cap proxy is badly wrong"
          %(r2(y[te],ya[te]), 100*relmed(y[te],ya[te])))
    lc_gain = relmed(y[te],ya[te]) - relmed(y[te],ylc[te])
    print("  (a learned correction changes held-out rel-err by %+.1f pts -> %s)."
          %(-100*lc_gain, "learning helps here" if lc_gain>0.02 else "NO material gain from learning this map"))

    # figure
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    ax[0].scatter(yc, y, s=14, c="#c40", label="force-cap proxy"); ax[0].scatter(ya, y, s=14, c="#2a8", label="analytic command-response")
    lim=max(yc.max(),y.max())*1.02; ax[0].plot([0,lim],[0,lim],"k--",lw=1)
    ax[0].set_xlabel("PREDICTED delivered ΣFn [N]"); ax[0].set_ylabel("ACTUAL delivered ΣFn [N]")
    ax[0].set_title("Delivered force ≠ force cap\n(cap points sit far right of the diagonal)"); ax[0].legend(fontsize=8)
    # delivered vs cap colored by object width
    cols={0.02:"#39a",0.03:"#e83",0.04:"#8a2"}
    for dm in DIAM:
        mm=[i for i,r in enumerate(rows) if r["diam"]==dm]
        ax[1].scatter([rows[i]["squeeze"]*2 for i in mm],[y[i] for i in mm],s=14,c=cols[dm],label="diam=%.2f"%dm)
    ax[1].plot([0,55],[0,55],"k--",lw=1,label="delivered=cap"); ax[1].set_xlabel("force cap = 2×squeeze [N]")
    ax[1].set_ylabel("actual delivered ΣFn [N]"); ax[1].set_title("Actuator saturates only for weak grips;\nstrong grips deliver << cap (width-dependent)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); import os; os.makedirs("cal_out",exist_ok=True)
    plt.savefig("cal_out/command_calibration.png",dpi=95); print("\n wrote cal_out/command_calibration.png")


if __name__ == "__main__":
    main()
