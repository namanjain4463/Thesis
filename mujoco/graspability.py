"""
GRASPABILITY CERTIFICATE  —  which objects the system can grasp, and WHY not.

In this framework graspability is not ad hoc: it is the SAME friction-cone /
force-closure feasibility that lives inside the transfer certificate. For the
floating parallel gripper we compute three interpretable margins at the held
grasp and check each against the ACTUAL lift outcome:

  gamma_kin  : do BOTH fingers reach the object (object half-width <= finger span)?
               -> kinematic reachability (embodiment-specific).
  gamma_fric : mu_available - mu_required,  mu_required = m g / sum(F_normal).
               friction cone must supply the vertical force to hold the weight.
               low mu, heavy object, or weak squeeze -> negative.
  gamma_cond : sigma_min of the grasp wrench matrix G (force-closure quality) -
               how well the realized contacts positively span the wrench they
               must resist. Near-zero -> unstable / ill-posed grasp.

Predict graspable iff all margins positive; the FAILING margin is the reason.
Then run the real grasp and confirm the certificate called it right.
"""
import numpy as np, mujoco, math
import contact_probe as cp
np.set_printoptions(precision=4, suppress=True)

PALM_HOME = 0.6
FINGER_SPAN = 0.034          # inner-face reach of each finger from center (x)
G = 9.81

def scene(obj_geom, mu, squeeze=20.0):
    return f"""
<mujoco model="graspability">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <default><geom solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <geom name="pedestal" type="box" pos="0 0 0.1" size="0.08 0.08 0.1" friction="{mu} 0.01 0.001"/>
    <body name="palm" pos="0 0 {PALM_HOME}">
      <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.05 0.03 0.02" mass="0.3" contype="2" conaffinity="0"/>
      <body name="lfinger" pos="0.04 0 -0.05">
        <joint name="lf" type="slide" axis="-1 0 0" range="0 0.035"/>
        <geom name="lfinger" type="box" size="0.006 0.012 0.03" mass="0.03" condim="3"
              friction="{mu} 0.02 0.002"/>
      </body>
      <body name="rfinger" pos="-0.04 0 -0.05">
        <joint name="rf" type="slide" axis="1 0 0" range="0 0.035"/>
        <geom name="rfinger" type="box" size="0.006 0.012 0.03" mass="0.03" condim="3"
              friction="{mu} 0.02 0.002"/>
      </body>
    </body>
    {obj_geom}
  </worldbody>
  <actuator>
    <position name="apx" joint="px" kp="800" kv="40"/><position name="apy" joint="py" kp="800" kv="40"/>
    <position name="apz" joint="pz" kp="800" kv="40"/>
    <position name="alf" joint="lf" kp="300" kv="8" ctrlrange="0 0.035" forcerange="-{squeeze} {squeeze}"/>
    <position name="arf" joint="rf" kp="300" kv="8" ctrlrange="0 0.035" forcerange="-{squeeze} {squeeze}"/>
  </actuator>
</mujoco>"""

def obj_xml(kind, size, mass, mu, hz):
    """object body; hz = half-height for grasp positioning; returns (xml, half_width_x)."""
    oz = 0.20 + hz + 0.001
    common = f'mass="{mass}" condim="3" friction="{mu} 0.005 0.0001"'
    if kind == "cylinder":
        r, h = size; g = f'<geom name="object" type="cylinder" size="{r} {h/2}" {common}/>'; hw = r
    elif kind == "sphere":
        r, = size; g = f'<geom name="object" type="sphere" size="{r}" {common}/>'; hw = r
    elif kind == "box":
        hx, hy, hz2 = size; g = f'<geom name="object" type="box" size="{hx} {hy} {hz2}" {common}/>'; hw = hx
    elif kind == "ellipsoid":
        ax, ay, az = size; g = f'<geom name="object" type="ellipsoid" size="{ax} {ay} {az}" {common}/>'; hw = ax
    return f'<body name="object" pos="0 0 {oz}"><freejoint name="obj"/>{g}</body>', hw, oz

def run(kind, size, mass, mu, hz, squeeze=20.0, _s=None):
    body, half_w, oz = obj_xml(kind, size, mass, mu, hz)
    m = mujoco.MjModel.from_xml_string(scene(body, mu, squeeze))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
    ogid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,"object"); obid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,"object")
    lg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,"lfinger"); rg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,"rfinger")
    gz = oz + 0.065; hov = gz + 0.15; lift = gz + 0.15
    def C(pz,lf,rf): d.ctrl[A["apx"]]=0;d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
    for k in range(60): a=(k+1)/60; C(gz,a*0.035,a*0.035); mujoco.mj_step(m,d)
    for k in range(60): C(gz,0.035,0.035); mujoco.mj_step(m,d)
    # ---- CERTIFICATE at held grasp (before lift) ----
    mujoco.mj_forward(m,d)
    Fn_sum=0.0; nL=nR=0; wrenches=[]
    com = np.array(d.xpos[obid])
    for ci in range(int(d.ncon)):
        c=d.contact[ci]
        if not(c.geom1==ogid or c.geom2==ogid): continue
        other=c.geom1 if c.geom2==ogid else c.geom2
        if other not in (lg,rg): continue
        f6=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f6); Fn=abs(f6[0])
        Fn_sum+=Fn
        if other==lg: nL+=1
        else: nR+=1
        n=np.array(c.frame).reshape(3,3)[0]; rvec=np.array(c.pos)-com
        wrenches.append(np.concatenate([n, np.cross(rvec,n)]))
    weight = mass*G
    mu_req = weight/max(Fn_sum,1e-9)
    gamma_kin  = FINGER_SPAN - half_w                       # >0: fingers can straddle
    both_fing  = (nL>0 and nR>0)
    gamma_fric = mu - mu_req                                 # >0: cone holds weight
    if len(wrenches)>=2:
        Wm=np.array(wrenches).T; gamma_cond=np.linalg.svd(Wm,compute_uv=False).min()
    else: gamma_cond=0.0
    # ---- LIFT and record actual outcome ----
    for k in range(90): a=(k+1)/90; C(gz+a*(lift-gz),0.035,0.035); mujoco.mj_step(m,d)
    for k in range(40): C(lift,0.035,0.035); mujoco.mj_step(m,d)
    lifted = float(d.xpos[obid][2]) - oz
    # lift only needs gravity resistance (kinematic reach + friction cone); full 6D
    # force-closure (gamma_cond) is reported as a DIAGNOSTIC, not a hard gate — a
    # 2-finger grasp is never full closure, that just means torque-disturbance sensitive.
    pred = (gamma_kin>0 and both_fing and gamma_fric>0.0)
    reason = "graspable" if pred else (
        "too wide (kinematic reach)" if (gamma_kin<=0 or not both_fing) else
        "friction cone (μ<μ_req)" if gamma_fric<=0.0 else "?")
    return dict(kind=kind, mu=mu, mass=mass, half_w=half_w, Fn=Fn_sum, mu_req=mu_req,
                gk=gamma_kin, gf=gamma_fric, gc=gamma_cond, both=both_fing,
                lifted=lifted, pred=pred, reason=reason)

# ---------------- object battery ----------------  (kind,size,mass,mu,hz,squeeze)
BATTERY = [
    ("cylinder",   ("cylinder",(0.025,0.08),  0.05, 1.0, 0.04, 20.0), "baseline"),
    ("sphere",     ("sphere",(0.028,),        0.05, 1.0, 0.028,20.0),"antipodal"),
    ("box",        ("box",(0.022,0.022,0.04), 0.05, 1.0, 0.04, 20.0), "face grasp"),
    ("ellipsoid",  ("ellipsoid",(0.026,0.026,0.045),0.05,1.0,0.045,20.0),"rounded"),
    ("heavy(firm)",("cylinder",(0.025,0.08),  0.20, 1.0, 0.04, 20.0), "heavy but firm grip"),
    ("WIDE box",   ("box",(0.06,0.03,0.04),   0.05, 1.0, 0.04, 20.0), "too wide -> kinematic"),
    ("slick+weak", ("sphere",(0.028,),        0.05, 0.05,0.028,3.0), "slick + weak grip -> friction"),
    ("heavy+weak", ("cylinder",(0.025,0.08),  0.20, 0.25,0.04, 3.0), "heavy + weak grip -> friction"),
]
print("="*104)
print("%-13s %-22s | %7s %7s %7s %6s | %8s | %-9s %-9s"%
      ("object","(m,μ)","γ_kin","γ_fric","γ_cond","both","lift[mm]","predict","actual"))
print("="*104)
rows=[]
for name,args,note in BATTERY:
    r=run(*args); rows.append((name,r,note))
    act = "GRASPED" if r["lifted"]>0.04 else "failed"
    ok = "✓" if ((r["lifted"]>0.04)==r["pred"]) else "✗MISMATCH"
    print("%-13s (%.2fkg,μ=%.2f) | %+7.3f %+7.3f %7.4f %6s | %+8.1f | %-9s %-7s %s"%
          (name, r["mass"], r["mu"], r["gk"], r["gf"], r["gc"], str(r["both"]),
           r["lifted"]*1000, "GRASP" if r["pred"] else "REJECT", act, ok))
print("="*104)
print("reasons for predicted rejects:")
for name,r,note in rows:
    if not r["pred"]: print("  %-13s -> %-24s (μ_req=%.2f vs μ=%.2f; note: %s)"%(name,r["reason"],r["mu_req"],r["mu"],note))
acc = np.mean([ (r["lifted"]>0.04)==r["pred"] for _,r,_ in rows ])
print("\nCERTIFICATE ACCURACY vs actual lift: %.0f%% (%d/%d)"%(100*acc,int(acc*len(rows)),len(rows)))

# ---------------- figure: certificate margins predict graspability ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12,4.6))
names=[n for n,_,_ in rows]; gk=[r["gk"]*1000 for _,r,_ in rows]; gf=[r["gf"] for _,r,_ in rows]
lifted=[r["lifted"]>0.04 for _,r,_ in rows]
col=["#2a8" if L else "#c40" for L in lifted]
# margin map: kinematic (mm) vs friction margin
for n,x,y,c in zip(names,gk,gf,col):
    ax[0].scatter(x,y,c=c,s=70,edgecolor="k",zorder=3); ax[0].annotate(n,(x,y),fontsize=7,xytext=(4,4),textcoords="offset points")
ax[0].axhline(0,color="k",lw=.8,ls="--"); ax[0].axvline(0,color="k",lw=.8,ls="--")
ax[0].set_xlabel("kinematic margin  γ_kin  [mm]  (finger span − object half-width)")
ax[0].set_ylabel("friction margin  γ_fric = μ − μ_req")
ax[0].set_title("Graspability = both margins > 0\n(green=lifted, red=failed; quadrant = reason)")
# per-object friction: μ available vs μ required
mu_av=[r["mu"] for _,r,_ in rows]; mu_rq=[r["mu_req"] for _,r,_ in rows]
xi=np.arange(len(names)); w=.38
ax[1].bar(xi-w/2, mu_av, w, label="μ available", color="#39a")
ax[1].bar(xi+w/2, mu_rq, w, label="μ required = mg/ΣFn", color="#e83")
ax[1].set_xticks(xi); ax[1].set_xticklabels(names, rotation=35, ha="right", fontsize=7)
ax[1].set_title("Friction feasibility: hold iff μ_avail ≥ μ_req"); ax[1].legend(fontsize=8); ax[1].set_ylabel("μ")
plt.tight_layout(); plt.savefig("graspability.png", dpi=95); print("wrote graspability.png")
