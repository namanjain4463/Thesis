"""
Pre-Panda robustness validation on REAL MuJoCo grasp data.
P1  port-relation consistency:  efc_J (qacc - qacc_smooth) == W efc_force   (W=efc_J M^-1 efc_J^T)
P2  z_local sufficiency:        is per-contact normal force recoverable from local kinematics?
P3  2A robustness:              structured robot-model error ΔM -> contamination of extracted
                                contact generalized force  (= ΔM qacc). "How accurate must Y be?"
"""
import numpy as np, mujoco, contact_probe as cp
import m2_floating_gripper_grasp as M
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

def run_and_capture(params):
    m = mujoco.MjModel.from_xml_string(M.scene_xml(params["d"],params["h"],params["mass"],params["mu"],params["force"]))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
    ogid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
    fg = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,g) for g in ("lfinger","rfinger")}
    oz = 0.20+params["h"]/2+0.001; gz=oz+0.065; hov=gz+0.15; lift=gz+0.15
    def C(pz,lf,rf): d.ctrl[A["apx"]]=params.get("xoff",0);d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-M.PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
    steps=[]  # each: dict with raw arrays + contact records; phase tag
    def cap(phase):
        mujoco.mj_forward(m,d)
        nefc,nv=int(d.nefc),m.nv
        rec=dict(phase=phase, nefc=nefc)
        if nefc>0:
            rec["efcJ"]=np.array(d.efc_J).reshape(nefc,nv)
            rec["efc_force"]=np.array(d.efc_force)[:nefc].copy()
        rec["qacc"]=np.array(d.qacc).copy(); rec["qacc_s"]=np.array(d.qacc_smooth).copy()
        Mf=np.zeros((nv,nv)); mujoco.mj_fullM(m,d,Mf); rec["M"]=Mf
        # per-contact local records (finger-object only)
        crs=[]
        for ci in range(int(d.ncon)):
            c=d.contact[ci]
            if not((c.geom1 in fg or c.geom2 in fg) and (c.geom1==ogid or c.geom2==ogid)): continue
            f6=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f6)
            addr,dim=int(c.efc_address),int(c.dim)
            vc=rec.get("efcJ",np.zeros((0,nv)))
            vcvel=(vc[addr:addr+dim]@d.qvel) if nefc>0 else np.zeros(dim)
            crs.append(dict(gap=float(c.dist), vn=float(vcvel[0]),
                            vt=float(np.hypot(vcvel[1],vcvel[2])) if dim>2 else 0.0,
                            Fn=float(f6[0]), Ft=float(np.hypot(f6[1],f6[2])), mu=float(c.friction[0])))
        rec["contacts"]=crs
        steps.append(rec)
    # approach (free), close, lift  -- capture every few steps
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80):
        a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
        if k%20==0: cap("approach")           # FREE motion (no contact)
    for k in range(60):
        a=(k+1)/60; C(gz,a*0.03,a*0.03); mujoco.mj_step(m,d)
        if k%10==0: cap("close")
    for k in range(40): C(gz,0.03,0.03); mujoco.mj_step(m,d); cap("hold") if k%8==0 else None
    for k in range(90):
        a=(k+1)/90; C(gz+a*(lift-gz),0.03,0.03); mujoco.mj_step(m,d)
        if k%10==0: cap("lift")
    return steps

# gather data across several grasp configs
PARAMS=[dict(M.BASE), {**M.BASE,"mu":0.6}, {**M.BASE,"d":0.024}, {**M.BASE,"force":8.0}]
allsteps=[]
for p in PARAMS: allsteps += run_and_capture(p)
print("captured %d steps across %d grasps\n"%(len(allsteps),len(PARAMS)))

# ---------- P1: port relation ----------
print("="*68); print("P1  PORT RELATION  efc_J(qacc-qacc_smooth) == W efc_force"); print("="*68)
res=[]
for s in allsteps:
    if s["nefc"]==0: continue
    Minv=np.linalg.inv(s["M"]); J=s["efcJ"]
    lhs=J@(s["qacc"]-s["qacc_s"]); W=J@Minv@J.T; rhs=W@s["efc_force"]
    den=max(np.linalg.norm(lhs),1e-9); res.append(np.linalg.norm(lhs-rhs)/den)
res=np.array(res)
print("  relative residual over %d contact steps: median=%.2e  max=%.2e"%(len(res),np.median(res),res.max()))
print("  VERDICT: %s\n"%("PASS - discrete factorization holds on real data" if res.max()<1e-4 else "CHECK"))

# ---------- P2: z_local sufficiency ----------
print("="*68); print("P2  z_local SUFFICIENCY  (normal force from local kinematics?)"); print("="*68)
rows=[(cr["gap"],cr["vn"],cr["mu"],cr["Fn"],cr["Ft"],cr["vt"]) for s in allsteps for cr in s["contacts"] if cr["Fn"]>1e-6]
R=np.array(rows)
if len(R)>30:
    pen=-R[:,0]; vn=-R[:,1]; Fn=R[:,3]
    X=np.column_stack([pen,vn,np.ones_like(pen)])
    beta,_,_,_=np.linalg.lstsq(X,Fn,rcond=None); pred=X@beta
    r2=1-np.sum((Fn-pred)**2)/np.sum((Fn-Fn.mean())**2)
    print("  N=%d finger-object contact samples"%len(R))
    print("  Fn ~ [penetration, -vn, 1]   R^2 = %.3f   (fit: Fn ~ %.3e*pen + %.3e*(-vn) + %.3e)"%(r2,*beta))
    # friction cone respected?  |Ft| <= mu Fn
    viol=np.mean(R[:,4] > R[:,2]*R[:,3]+1e-6)
    print("  friction-cone check |Ft|<=mu*Fn: violated on %.1f%% of samples"%(100*viol))
    print("  READ: high R^2 => local normal compliance is recoverable from z_local kinematics;")
    print("        residual variance is the coupling/squeeze part (belongs to the SOLVE, not the law).\n")

# ---------- P3: 2A robustness (model error -> contact-force contamination) ----------
print("="*68); print("P3  ROBUSTNESS TO ROBOT-MODEL ERROR (2A)  on real trajectories"); print("="*68)
# structured ΔM (SPD perturbation) scaled to ||ΔM||=delta*||M||
def structured_dM(Mf, delta):
    P=rng.standard_normal(Mf.shape); P=(P+P.T)/2
    return delta*np.linalg.norm(Mf,2)/np.linalg.norm(P,2)*P
free=[s for s in allsteps if s["phase"]=="approach"]
con =[s for s in allsteps if s["nefc"]>0 and len(s["contacts"])>0]
for delta in [0.02,0.05,0.10]:
    ph=[]; cc=[]
    for s in free:                      # phantom force in FREE motion (true contact=0)
        dM=structured_dM(s["M"],delta); ph.append(np.linalg.norm(dM@s["qacc"]))
    for s in con:                       # contamination of extracted contact gen-force during contact
        dM=structured_dM(s["M"],delta)
        gc=s["efcJ"].T@s["efc_force"]   # true contact generalized force
        cc.append(np.linalg.norm(dM@s["qacc"])/max(np.linalg.norm(gc),1e-9))
    print("  model err %2.0f%% : phantom free-motion force ||ΔM qacc|| median=%.3f N·(gen)   "
          "contact-force contamination median=%.1f%%"%(delta*100,np.median(ph),100*np.median(cc)))
print("  READ: contamination ~ linear in model error. This is the sim-to-real budget:")
print("        it sets how well Y (robot inertia+payload) must be identified before C_theta")
print("        stays embodiment-invariant. In sim ΔM=0 (clean); on hardware this is the fight.")
