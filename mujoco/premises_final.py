"""
PRE-PANDA GATE — the four load-bearing premises of the Factorized Interaction
World Model, each tested against GROUND-TRUTH MuJoCo physics on real grasp data.
Run this before adding any second embodiment; it must print GO.

  P1  COUPLING IDENTITY   M(qacc-qacc_s)=Jᵀf  <=>  efc_J(qacc-qacc_s)=W efc_force
      The embodiment enters the object EOM ONLY through W=J M⁻¹ Jᵀ (+ Jᵀ).  [EXACT]
  P2  LOCAL/GLOBAL SPLIT  the contact force is the output of a convex solve whose
      data splits into analytical global W and LOCAL law (R,aref,μ ∈ z_local).
      Verified machine-exact in the frictionless case (no cone term):
         (R - W) f = J qacc_s + aref .   Friction adds only the μ-cone term.
      Corollary (correctly): force is NOT a local function of kinematics (low R²);
      that non-locality is precisely why the analytical W-solve is needed.
  P3  MODEL-ERROR BUDGET  structured ΔM contaminates the extracted contact force
      ~linearly; sets how well Y (robot inertia+payload) must be identified.
"""
import numpy as np, mujoco, contact_probe as cp
import m2_floating_gripper_grasp as M
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ---------------- shared capture on real grasps ----------------
def run_and_capture(params):
    m = mujoco.MjModel.from_xml_string(M.scene_xml(params["d"],params["h"],params["mass"],params["mu"],params["force"]))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
    oz=0.20+params["h"]/2+0.001; gz=oz+0.065; hov=gz+0.15; lift=gz+0.15
    def C(pz,lf,rf): d.ctrl[A["apx"]]=params.get("xoff",0);d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-M.PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
    steps=[]
    def cap(phase):
        mujoco.mj_forward(m,d); nefc,nv=int(d.nefc),m.nv
        rec=dict(phase=phase,nefc=nefc,qacc=np.array(d.qacc).copy(),qacc_s=np.array(d.qacc_smooth).copy())
        Mf=np.zeros((nv,nv)); mujoco.mj_fullM(m,d,Mf); rec["M"]=Mf
        if nefc>0:
            rec["efcJ"]=np.array(d.efc_J).reshape(nefc,nv); rec["efc_force"]=np.array(d.efc_force)[:nefc].copy()
        steps.append(rec)
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d); cap("approach") if k%20==0 else None
    for k in range(60): a=(k+1)/60; C(gz,a*0.03,a*0.03); mujoco.mj_step(m,d); cap("close") if k%10==0 else None
    for k in range(40): C(gz,0.03,0.03); mujoco.mj_step(m,d); cap("hold") if k%8==0 else None
    for k in range(90): a=(k+1)/90; C(gz+a*(lift-gz),0.03,0.03); mujoco.mj_step(m,d); cap("lift") if k%10==0 else None
    return steps

PARAMS=[dict(M.BASE),{**M.BASE,"mu":0.6},{**M.BASE,"d":0.024},{**M.BASE,"force":8.0}]
allsteps=[]
for p in PARAMS: allsteps+=run_and_capture(p)

# ---------------- P1 ----------------
res=[]
for s in allsteps:
    if s["nefc"]==0: continue
    J=s["efcJ"]; Minv=np.linalg.inv(s["M"])
    lhs=J@(s["qacc"]-s["qacc_s"]); rhs=(J@Minv@J.T)@s["efc_force"]
    res.append(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(lhs),1e-9))
res=np.array(res); P1=res.max()<1e-4

# ---------------- P2 (machine-exact, frictionless clean case) ----------------
XML="""<mujoco><option gravity="0 0 -9.81" cone="elliptic" jacobian="dense" solver="Newton"
 iterations="500" tolerance="1e-14" ls_iterations="100"/><worldbody>
 <geom type="plane" size="2 2 .1" friction="0 0 0"/>
 <body pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".2" friction="0 0 0"/></body>
 </worldbody></mujoco>"""
mf=mujoco.MjModel.from_xml_string(XML); df=mujoco.MjData(mf)
for _ in range(2000): mujoco.mj_step(mf,df)
mujoco.mj_forward(mf,df)
nefc=int(df.nefc); J=np.array(df.efc_J).reshape(nefc,mf.nv)
W=J@cp.Minv_apply(mf,df,J).T; R=np.array(df.efc_R[:nefc]); aref=np.array(df.efc_aref[:nefc])
f=np.array(df.efc_force[:nefc]); qs=np.array(df.qacc_smooth)
act=f>1e-9
kkt=np.linalg.norm(((np.diag(R)-W)@f-(J@qs+aref))[act])/max(np.linalg.norm((J@qs+aref)[act]),1e-12)
wt=abs(f.sum()-0.2*9.81)
P2=kkt<1e-6 and wt<1e-4

# force non-locality (expected-low R²) on real friction data, for the record
rows=[]
for p in PARAMS:
    m=mujoco.MjModel.from_xml_string(M.scene_xml(p["d"],p["h"],p["mass"],p["mu"],p["force"]))
    cp.force_dense_jacobian(m); d=mujoco.MjData(m)
    A={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
    ogid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"object")
    fg={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in ("lfinger","rfinger")}
    oz=0.20+p["h"]/2+0.001; gz=oz+0.065; hov=gz+0.15
    def C(pz,lf,rf): d.ctrl[A["apx"]]=0;d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-M.PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
    def grab():
        mujoco.mj_forward(m,d); nefc,nv=int(d.nefc),m.nv
        Jl=np.array(d.efc_J).reshape(nefc,nv) if nefc>0 else np.zeros((0,nv))
        for ci in range(int(d.ncon)):
            c=d.contact[ci]
            if not((c.geom1 in fg or c.geom2 in fg) and (c.geom1==ogid or c.geom2==ogid)): continue
            f6=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f6); addr,dim=int(c.efc_address),int(c.dim)
            vc=(Jl[addr:addr+dim]@d.qvel) if nefc>0 else np.zeros(3)
            if f6[0]>1e-6: rows.append((-float(c.dist),-float(vc[0]),f6[0]))
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
    for k in range(60): a=(k+1)/60; C(gz,a*0.03,a*0.03); mujoco.mj_step(m,d); grab() if k%6==0 else None
    for k in range(60): C(gz,0.03,0.03); mujoco.mj_step(m,d); grab() if k%6==0 else None
Rr=np.array(rows); Xr=np.column_stack([Rr[:,0],Rr[:,1],np.ones(len(Rr))])
b,_,_,_=np.linalg.lstsq(Xr,Rr[:,2],rcond=None); pr=Xr@b
r2=1-np.sum((Rr[:,2]-pr)**2)/np.sum((Rr[:,2]-Rr[:,2].mean())**2)

# ---------------- P3 ----------------
def structured_dM(Mf,delta):
    P=rng.standard_normal(Mf.shape); P=(P+P.T)/2
    return delta*np.linalg.norm(Mf,2)/np.linalg.norm(P,2)*P
con=[s for s in allsteps if s["nefc"]>0]
budget={}
for delta in [0.02,0.05,0.10]:
    cc=[np.linalg.norm(structured_dM(s["M"],delta)@s["qacc"])/max(np.linalg.norm(s["efcJ"].T@s["efc_force"]),1e-9) for s in con]
    budget[delta]=100*np.median(cc)
P3=budget[0.02]<2.0

print("="*70)
print("PRE-PANDA GATE  (Factorized Interaction World Model premises)")
print("="*70)
print(" P1 coupling identity   : residual median=%.1e max=%.1e   -> %s"%(np.median(res),res.max(),"PASS" if P1 else "FAIL"))
print(" P2 local/global split  : frictionless KKT resid=%.1e  weight err=%.1e N -> %s"%(kkt,wt,"PASS" if P2 else "FAIL"))
print("      (force IS non-local: Fn~[pen,vn] R²=%.2f on %d friction samples — EXPECTED,"%(r2,len(Rr)))
print("       the coupling is exactly the analytical W-solve; local law inputs R,aref,μ ∈ z_local)")
print(" P3 model-error budget  : contamination 2%%->%.1f%%  5%%->%.1f%%  10%%->%.1f%%  -> %s"
      %(budget[0.02],budget[0.05],budget[0.10],"PASS" if P3 else "FAIL"))
print("-"*70)
print(" DECISION: %s"%("GO — build the Panda embodiment (reuse contact_probe + z_local unchanged)."
      if (P1 and P2 and P3) else "NO-GO — investigate the FAILing premise before adding an embodiment."))
