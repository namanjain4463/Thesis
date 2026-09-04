"""Probe: does a height-varying object friction control the CONTACT friction, and
can we MEASURE the local μ(z) field from physics? Measure |Ft|/Fn at SLIDING
contacts (where |Ft|/Fn = μ_kinetic exactly) as the object slides through a weak grip."""
import numpy as np, mujoco, math
import contact_probe as cp
np.set_printoptions(precision=3, suppress=True)

PALM_HOME=0.6
def seg_cylinder(r, H, K, mu_lo, mu_hi, mass, finger_mu=0.02, squeeze=2.0):
    segs=""
    for k in range(K):
        zc = -H/2 + (k+0.5)*H/K
        mu = mu_lo + (mu_hi-mu_lo)*k/(K-1)
        segs += f'<geom name="seg{k}" type="cylinder" size="{r} {H/(2*K)}" pos="0 0 {zc}" mass="{mass/K}" condim="3" friction="{mu} 0.005 0.0001"/>\n'
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

r,H,K = 0.025, 0.10, 8
m = mujoco.MjModel.from_xml_string(seg_cylinder(r,H,K,0.1,1.0,0.10, squeeze=2.5))
cp.force_dense_jacobian(m); d=mujoco.MjData(m)
A={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
obid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"object")
og={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f"seg{k}") for k in range(K)}
fg={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in ("lfinger","rfinger")}
gz=0.20+H/2+0.001+0.065; hov=gz+0.15
def C(pz,lf,rf): d.ctrl[A["apx"]]=0;d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
for k in range(60): a=(k+1)/60; C(gz,a*0.035,a*0.035); mujoco.mj_step(m,d)
# now hold with weak grip and lift the palm slowly -> object slides down through fingers
samples=[]
for k in range(400):
    C(gz+0.10*k/400, 0.035, 0.035); mujoco.mj_step(m,d)
    nefc=int(d.nefc); J=np.array(d.efc_J).reshape(nefc,m.nv) if nefc>0 else None
    for ci in range(int(d.ncon)):
        c=d.contact[ci]
        if not((c.geom1 in fg or c.geom2 in fg) and (c.geom1 in og or c.geom2 in og)): continue
        f6=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f6)
        Fn=abs(f6[0]); Ft=math.hypot(f6[1],f6[2])
        addr,dim=int(c.efc_address),int(c.dim)
        vt = math.hypot(*(J[addr+1:addr+dim]@d.qvel)) if (nefc>0 and dim>=3) else 0.0
        Rm=np.array(d.xmat[obid]).reshape(3,3); z=(Rm.T@(np.array(c.pos)-np.array(d.xpos[obid])))[2]
        seg=c.geom1 if c.geom2 in fg else c.geom2
        mu_true=float(m.geom_friction[seg][0])
        if Fn>1e-3 and vt>2e-3:                          # SLIDING => |Ft|/Fn = mu
            samples.append((z, Ft/Fn, mu_true, vt))
S=np.array(samples)
print("sliding contact samples: %d"%len(S))
if len(S):
    print(" z_mat range [% .3f,% .3f]   measured μ_obs range [%.2f,%.2f]   true μ range [%.2f,%.2f]"
          %(S[:,0].min(),S[:,0].max(),S[:,1].min(),S[:,1].max(),S[:,2].min(),S[:,2].max()))
    err=np.abs(S[:,1]-S[:,2])
    print(" |μ_obs − μ_true| median=%.3f  90th=%.3f   corr(μ_obs,μ_true)=%.3f"
          %(np.median(err),np.percentile(err,90),np.corrcoef(S[:,1],S[:,2])[0,1]))
    print(" => %s"%("μ_obs tracks the segment field: heterogeneous friction is CONTACT-controlling and MEASURABLE at slip."
          if np.corrcoef(S[:,1],S[:,2])[0,1]>0.7 else "μ_obs does NOT track segment μ — check friction-combine rule."))
