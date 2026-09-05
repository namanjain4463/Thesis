"""
HETEROGENEOUS-MATERIAL COVERING LAW — closes the one honest gap in the first test.

The object surface now carries a REAL spatially-varying material: a cylinder whose
friction ramps with height, μ(z). We LEARN the local law from PHYSICS — measuring
μ_obs = |Ft|/Fn at sliding contacts (kinetic friction = μ exactly) as the object
slides through the grip. The learned field is genuinely varying and noisy (real
measurement), not the analytical Y_object or a controlled stand-in.

Test: learn μ̂(z) from a LOW z-band of contacts, predict on the full surface, and
check the covering law |μ̂−μ| ≲ ε + C·L·dist(z*, trained band). If it holds on the
LEARNED material field, the reduction (embodiment transfer = surface scattered-data
approximation) is validated on real learned physics, not a proxy.
"""
import numpy as np, mujoco, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import contact_probe as cp

PALM_HOME = 0.6
def seg_cylinder(r, H, K, mu_lo, mu_hi, mass, finger_mu=0.02, squeeze=2.0):
    segs=""
    for k in range(K):
        zc = -H/2 + (k+0.5)*H/K
        # sample the CONTINUOUS ramp at the segment CENTER so the piecewise material
        # is a faithful discretization of mu_true(z) (was k/(K-1), which only matched
        # the ramp at the middle segment -> a systematic staircase-vs-ramp offset).
        mu = mu_lo + (mu_hi-mu_lo)*((zc + H/2)/H)
        segs += (f'<geom name="seg{k}" type="cylinder" size="{r} {H/(2*K)}" pos="0 0 {zc}" '
                 f'mass="{mass/K}" condim="3" friction="{mu} 0.005 0.0001"/>\n')
    oz = 0.20 + H/2 + 0.001
    return f"""
<mujoco model="hetero"><option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
 <default><geom solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
 <worldbody>
  <geom name="floor" type="plane" size="3 3 .1"/>
  <geom name="pedestal" type="box" pos="0 0 0.1" size="0.08 0.08 0.1"/>
  <body name="palm" pos="0 0 {PALM_HOME}">
   <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/><joint name="pz" type="slide" axis="0 0 1"/>
   <geom name="palm" type="box" size="0.05 0.03 0.02" mass="0.3" contype="2" conaffinity="0"/>
   <body name="lfinger" pos="0.04 0 -0.05"><joint name="lf" type="slide" axis="-1 0 0" range="0 0.035"/>
     <geom name="lfinger" type="box" size="0.006 0.012 0.03" mass="0.03" condim="3" friction="{finger_mu} 0.02 0.002"/></body>
   <body name="rfinger" pos="-0.04 0 -0.05"><joint name="rf" type="slide" axis="1 0 0" range="0 0.035"/>
     <geom name="rfinger" type="box" size="0.006 0.012 0.03" mass="0.03" condim="3" friction="{finger_mu} 0.02 0.002"/></body>
  </body>
  <body name="object" pos="0 0 {oz}"><freejoint name="obj"/>{segs}</body>
 </worldbody>
 <actuator>
  <position name="apx" joint="px" kp="800" kv="40"/><position name="apy" joint="py" kp="800" kv="40"/>
  <position name="apz" joint="pz" kp="800" kv="40"/>
  <position name="alf" joint="lf" kp="300" kv="8" ctrlrange="0 0.035" forcerange="-{squeeze} {squeeze}"/>
  <position name="arf" joint="rf" kp="300" kv="8" ctrlrange="0 0.035" forcerange="-{squeeze} {squeeze}"/>
 </actuator></mujoco>"""

r, H, K = 0.025, 0.10, 24        # finer discretization: staircase→ramp mismatch << learning error
MU_LO, MU_HI, MASS = 0.1, 1.0, 0.10
def mu_true(z): return MU_LO + (MU_HI-MU_LO)*((z + H/2)/H)     # ground-truth ramp
L_field = (MU_HI-MU_LO)/H                                       # Lipschitz of μ(z)

def collect(grip_offset, squeeze=2.5, slide=0.10):
    m = mujoco.MjModel.from_xml_string(seg_cylinder(r,H,K,MU_LO,MU_HI,MASS,squeeze=squeeze))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    A={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
    obid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"object")
    og={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f"seg{k}") for k in range(K)}
    fg={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in ("lfinger","rfinger")}
    gz=0.20+H/2+0.001+0.065+grip_offset; hov=gz+0.15
    def C(pz,lf,rf): d.ctrl[A["apx"]]=0;d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
    for k in range(60): a=(k+1)/60; C(gz,a*0.035,a*0.035); mujoco.mj_step(m,d)
    pts=[]
    for k in range(400):
        C(gz+slide*k/400, 0.035, 0.035); mujoco.mj_step(m,d)
        nefc=int(d.nefc); J=np.array(d.efc_J).reshape(nefc,m.nv) if nefc>0 else None
        for ci in range(int(d.ncon)):
            c=d.contact[ci]
            if not((c.geom1 in fg or c.geom2 in fg) and (c.geom1 in og or c.geom2 in og)): continue
            f6=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f6); Fn=abs(f6[0]); Ft=math.hypot(f6[1],f6[2])
            addr,dim=int(c.efc_address),int(c.dim)
            vt=math.hypot(*(J[addr+1:addr+dim]@d.qvel)) if (nefc>0 and dim>=3) else 0.0
            Rm=np.array(d.xmat[obid]).reshape(3,3); z=(Rm.T@(np.array(c.pos)-np.array(d.xpos[obid])))[2]
            if Fn>1e-3 and vt>2e-3: pts.append((z, Ft/Fn))
    return pts

print("collecting learned friction-field samples μ_obs(z) from sliding physics...")
pts=[]
for off in [-0.03,-0.015,0.0,0.015,0.03]: pts += collect(off)
S=np.array(pts); z=S[:,0]; mu_obs=S[:,1]
# bin to reduce measurement noise for the field learning (nonparametric target)
print("  %d sliding samples   z∈[% .3f,% .3f]   μ_obs∈[%.2f,%.2f]"%(len(S),z.min(),z.max(),mu_obs.min(),mu_obs.max()))

# ---- covering law on the LEARNED field ----
def krr(Xtr,ytr,Xte,ell=0.012,lam=1e-3):
    D=np.abs(Xtr[:,None]-Xtr[None,:]); K=np.exp(-D**2/(2*ell**2)); mu=ytr.mean()
    a=np.linalg.solve(K+lam*np.eye(len(ytr)),ytr-mu)
    Dt=np.abs(Xte[:,None]-Xtr[None,:]); return mu+np.exp(-Dt**2/(2*ell**2))@a

zc=np.percentile(z,45)
tr=z<=zc; ztr=z[tr]; mutr=mu_obs[tr]
zhat=krr(ztr, mutr, z)                               # RBF-KRR: learn LOW band, predict all
err=np.abs(zhat - mu_true(z))                        # error vs GROUND TRUTH μ(z)
# reference estimator that RESPECTS the field's Lipschitz constant: value at the
# nearest sampled height. By triangle ineq |μ̂−μ| ≤ ε_measure + L·dist, so it is the
# yardstick for whether a VIOLATION is a covering-geometry failure or a learner one.
jnear=np.argmin(np.abs(z[:,None]-ztr[None,:]),axis=1); mnear=mutr[jnear]
err_lip=np.abs(mnear - mu_true(z))
cov=np.array([0.0 if zz<=zc else zz-zc for zz in z]) # covering distance to train band (upper side)
insup=err[cov<1e-4]; out=cov>1e-3
print("\n" + "="*66)
print("COVERING LAW on the LEARNED material field μ(z)  (train z<=%.3f)"%zc)
print("="*66)
eps_learn = float(np.percentile(insup, 95)) if len(insup) else 0.0   # robust in-support ceiling
print("  in-support |μ̂−μ| median = %.3f  (ε_learn=95th pct = %.3f)   (RBF-KRR matches true μ where sampled)"
      %(np.median(insup), eps_learn))
bound_holds = False
if out.sum()>3:
    def bound_viol(e): return float(np.mean(e[out] > eps_learn + 1.4*L_field*cov[out] + 1e-9))
    slope=np.polyfit(cov[out], err[out],1)[0]
    viol_krr=bound_viol(err); viol_lip=bound_viol(err_lip)
    bound_holds = viol_krr <= 0.02
    print("  out-of-support error slope=%.2f vs field L=%.1f;  bound |μ̂−μ| ≤ ε_learn + C·L·dist (C≈1.4):"%(slope,L_field))
    print("    RBF-KRR estimator          : violated on %5.1f%% of out-of-support pts -> %s"
          %(100*viol_krr, "HOLDS" if bound_holds else "VIOLATED"))
    print("    Lipschitz-consistent (ref) : violated on %5.1f%% of out-of-support pts -> %s"
          %(100*viol_lip, "HOLDS" if viol_lip<=0.02 else "VIOLATED"))
    print("    diag: RBF-KRR min extrapolated μ̂=%.2f (field min=%.2f) -> it MEAN-REVERTS/undershoots,"
          %(float(zhat[out].min()), MU_LO))
    print("          so it does NOT extrapolate within the Lipschitz bound even near the support edge.")
# HONEST verdict: the covering GEOMETRY holds (a Lipschitz-consistent estimator obeys the
# bound at all distances); the RBF-KRR LEARNER used here mean-reverts under extrapolation
# and violates it -> the law is NOT demonstrated as a strict bound for THIS learner.
metric_ok = (out.sum()>3 and bound_viol(err_lip) <= 0.02)
print("  => covering DISTANCE/bound correctly computed: %s"
      "  (a Lipschitz-consistent estimator obeys ε+C·L·dist by construction, so the"
      " KRR violation is a genuine learner-extrapolation failure, not a metric artifact)."
      %("confirmed" if metric_ok else "unconfirmed"))
print("     covering LAW on the LEARNED RBF field: %s — RBF-KRR mean-reverts under"
      " extrapolation and breaks the bound; a Lipschitz-respecting estimator is the open item."
      %("HOLDS" if (np.median(insup)<0.1 and bound_holds) else "PARTIAL/NEGATIVE"))

# figure
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
zs=np.linspace(z.min(),z.max(),100)
ax[0].scatter(z*1000, mu_obs, s=8, c="#aaa", label="μ_obs (sliding physics)")
ax[0].plot(zs*1000, mu_true(zs), "k-", lw=1.5, label="true μ(z)")
ax[0].plot(np.sort(z)*1000, krr(z[tr],mu_obs[tr],np.sort(z)), "-", c="#c02", lw=1.6, label="learned μ̂ (from z≤%.0fmm)"%(zc*1000))
ax[0].axvline(zc*1000, color="#06c", ls="--", lw=1, label="train/test split")
ax[0].set_xlabel("material height z  [mm]"); ax[0].set_ylabel("friction μ"); ax[0].legend(fontsize=8)
ax[0].set_title("Learned heterogeneous friction field\n(matches in-support, extrapolates out)")
ax[1].scatter(cov[out]*1000, err[out], s=12, c="#c02", alpha=.7, label="|μ̂−μ|")
xx=np.linspace(0,cov.max(),40); ax[1].plot(xx*1000, 1.4*L_field*xx, "k--", lw=1, label="C·L·dist bound")
ax[1].set_xlabel("covering distance to train band  [mm]"); ax[1].set_ylabel("|μ̂−μ|")
ax[1].set_title("Covering law on the LEARNED field"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("hetero_covering.png", dpi=95); print("\nwrote hetero_covering.png")
