"""De-risk: can both embodiments grasp a shared cylinder and give contact points
across a controllable band of surface heights? Print (theta,z_material) spread."""
import numpy as np, mujoco, math
import contact_probe as cp
import m2_floating_gripper_grasp as Mfg
import panda_embodiment as P
np.set_printoptions(precision=4, suppress=True)

OBJ = dict(d=0.05, h=0.10, mass=0.05, mu=1.0, force=20.0)

def material_coords(m, d, ci, obid):
    c = d.contact[ci]
    R = np.array(d.xmat[obid]).reshape(3,3); p = np.array(d.xpos[obid])
    mc = R.T @ (np.array(c.pos) - p)               # contact point in object frame
    theta = math.atan2(mc[1], mc[0]); z = mc[2]; r = math.hypot(mc[0], mc[1])
    return theta, z, r

def float_contacts(obj, zoff, yaw=0.0):
    m = mujoco.MjModel.from_xml_string(Mfg.scene_xml(obj["d"],obj["h"],obj["mass"],obj["mu"],obj["force"]))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    ojid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,"objfree"); adr = m.jnt_qposadr[ojid]
    mujoco.mj_resetData(m,d)
    d.qpos[adr+3:adr+7] = [math.cos(yaw/2),0,0,math.sin(yaw/2)]
    mujoco.mj_forward(m,d)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("apx","apy","apz","alf","arf")}
    ogid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,"object"); obid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,"object")
    fg = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,g) for g in ("lfinger","rfinger")}
    oz = 0.20+obj["h"]/2+0.001; gz = oz+zoff+0.065; hov = gz+0.15
    def C(pz,lf,rf): d.ctrl[A["apx"]]=0;d.ctrl[A["apy"]]=0;d.ctrl[A["apz"]]=pz-Mfg.PALM_HOME;d.ctrl[A["alf"]]=lf;d.ctrl[A["arf"]]=rf
    for k in range(60): C(hov,0,0); mujoco.mj_step(m,d)
    for k in range(80): a=(k+1)/80; C(hov+a*(gz-hov),0,0); mujoco.mj_step(m,d)
    for k in range(60): a=(k+1)/60; C(gz,a*0.03,a*0.03); mujoco.mj_step(m,d)
    for k in range(80): C(gz,0.03,0.03); mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    pts=[]
    for ci in range(int(d.ncon)):
        c=d.contact[ci]
        if (c.geom1 in fg or c.geom2 in fg) and (c.geom1==ogid or c.geom2==ogid):
            th,z,r = material_coords(m,d,ci,obid); pts.append((th,z,r))
    return pts

def panda_contacts(obj, hoff=0.0):
    m = P.make_model(obj); d = mujoco.MjData(m); P.set_home(m,d)
    hold={}
    def on_step(mm,dd,W,obid,ogid,phase):
        if phase=="hold": hold["m"],hold["d"]=mm,dd
    P.run_grasp(obj, on_step=on_step, hoff=hoff)
    m,d=hold["m"],hold["d"]; mujoco.mj_forward(m,d)
    ogid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"object"); obid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"object")
    fg=P._finger_geoms(m); pts=[]
    for ci in range(int(d.ncon)):
        c=d.contact[ci]
        if (c.geom1 in fg or c.geom2 in fg) and (c.geom1==ogid or c.geom2==ogid):
            t=int(m.geom_type[c.geom1 if c.geom2==ogid else c.geom2])
            if t==int(mujoco.mjtGeom.mjGEOM_MESH): continue
            th,z,r=material_coords(m,d,ci,obid); pts.append((th,z,r))
    return pts

def _main():
  print("FLOATING contacts across grasp heights (zoff):")
  for zoff in [-0.02,-0.01,0.0,0.01,0.02]:
    pts=float_contacts(OBJ, zoff)
    if pts:
        a=np.array(pts); print("  zoff=%+.3f  n=%2d  z_mat[% .3f,% .3f]  theta[% .2f,% .2f]  r~%.3f"
                                %(zoff,len(a),a[:,1].min(),a[:,1].max(),a[:,0].min(),a[:,0].max(),np.median(a[:,2])))
    else: print("  zoff=%+.3f  NO CONTACTS"%zoff)
  print("PANDA contacts across grasp heights (hoff):")
  for hoff in [-0.03,-0.015,0.0,0.015,0.03]:
    try:
        pts=panda_contacts(OBJ, hoff)
        if pts:
            a=np.array(pts); print("  hoff=%+.3f  n=%2d  z_mat[% .3f,% .3f]  theta[% .2f,% .2f]  r~%.3f"
                                    %(hoff,len(a),a[:,1].min(),a[:,1].max(),a[:,0].min(),a[:,0].max(),np.median(a[:,2])))
        else: print("  hoff=%+.3f  NO CONTACTS"%hoff)
    except Exception as e:
        print("  hoff=%+.3f  ERR %s"%(hoff,e))

if __name__ == "__main__":
    _main()
