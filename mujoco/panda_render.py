import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np, mujoco, panda_embodiment as P
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

m = P.make_model(dict(P.BASE)); d = mujoco.MjData(m); P.set_home(m, d)
hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
home_quat = np.array(d.xquat[hb]).copy()
oz = P.PED_TOP + P.BASE["h"]/2 + 0.001
gp = np.array([P.OBJ_X, P.OBJ_Y, oz+0.075]); pp = gp+[0,0,0.12]
qh = np.array(d.qpos).copy()
qpre = P.ik_hand(m, qh, pp, home_quat); qgr = P.ik_hand(m, qpre, gp, home_quat)
qlift = P.ik_hand(m, qgr, gp+[0,0,0.18], home_quat)

ren = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.5,0,0.5]; cam.distance=1.4; cam.azimuth=90; cam.elevation=-15
def shot(title):
    mujoco.mj_forward(m,d); ren.update_scene(d, cam); return ren.render(), title

frames=[]
def command(qarm,grip): d.ctrl[:7]=qarm[:7]; d.ctrl[7]=grip
# home
frames.append(shot("home"))
# approach
for k in range(150):
    a=(k+1)/150; command(qh*(1-a)+qpre*a, 255); mujoco.mj_step(m,d)
frames.append(shot("pre-grasp (open)"))
# descend
for k in range(150):
    a=(k+1)/150; command(qpre*(1-a)+qgr*a, 255); mujoco.mj_step(m,d)
frames.append(shot("descended"))
# close
for k in range(200):
    a=(k+1)/200; command(qgr, 255*(1-a)); mujoco.mj_step(m,d)
for k in range(150): command(qgr, 0); mujoco.mj_step(m,d)
frames.append(shot("closed"))
# lift
for k in range(250):
    a=(k+1)/250; command(qgr*(1-a)+qlift*a, 0); mujoco.mj_step(m,d)
frames.append(shot("lifted  obj_z=%.3f"%float(d.xpos[ob][2])))

fig,ax=plt.subplots(1,len(frames),figsize=(4*len(frames),4))
for a,(img,t) in zip(ax,frames): a.imshow(img); a.set_title(t,fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig("panda_grasp_strip.png",dpi=90); print("wrote panda_grasp_strip.png  final obj_z=%.3f (start %.3f)"%(float(d.xpos[ob][2]),oz))
