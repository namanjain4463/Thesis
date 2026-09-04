"""Real MuJoCo 3D render (OSMesa, headless) of a grasp trial -> montage PNG."""
import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contact_probe as cp
import m2_floating_gripper_grasp as M

def render_trial(params, path, title, W=520, H=440):
    m = mujoco.MjModel.from_xml_string(
        M.scene_xml(params["d"], params["h"], params["mass"], params["mu"], params["force"]))
    cp.force_dense_jacobian(m); d = mujoco.MjData(m)
    # cosmetic (no physics effect): brighter headlight + colored geoms
    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]; m.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    for g, rgba in [("object",[0.15,0.45,0.85,1]),("lfinger",[0.85,0.35,0.1,1]),
                    ("rfinger",[0.85,0.35,0.1,1]),("palm",[0.45,0.45,0.5,1]),
                    ("pedestal",[0.75,0.75,0.75,1])]:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if gid >= 0: m.geom_rgba[gid] = rgba
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
         for n in ("apx","apy","apz","alf","arf")}
    oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_z = 0.20 + params["h"]/2 + 0.001
    gx = params.get("xoff",0.0); gz = obj_z + params.get("zoff",0.0) + 0.065
    hover, lift = gz+0.15, gz+0.15
    def C(pz,lf,rf):
        d.ctrl[A["apx"]]=gx; d.ctrl[A["apy"]]=0; d.ctrl[A["apz"]]=pz-M.PALM_HOME
        d.ctrl[A["alf"]]=lf; d.ctrl[A["arf"]]=rf
    cam = mujoco.MjvCamera(); cam.lookat[:] = [gx*0.5, 0, 0.27]
    cam.distance = 0.62; cam.azimuth = 50; cam.elevation = -18
    ren = mujoco.Renderer(m, H, W)
    plan = [("hover",hover,hover,0,0,60),("descend",hover,gz,0,0,80),
            ("close",gz,gz,0.03,0.03,60),("lift",gz,lift,0.03,0.03,90),
            ("held",lift,lift,0.03,0.03,60)]
    frames=[]
    for name,z0,z1,lf,rf,n in plan:
        for k in range(n):
            a=(k+1)/n; C(z0+a*(z1-z0),lf,rf); mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d); ren.update_scene(d,cam)
        frames.append((name, ren.render().copy(), float(d.xpos[oid][2])))
    fig,axes=plt.subplots(1,len(frames),figsize=(2.5*len(frames),2.6))
    for ax,(name,img,oz) in zip(axes,frames):
        ax.imshow(img); ax.axis("off"); ax.set_title("%s\nobjZ=%.3f"%(name,oz),fontsize=9)
    rise=(frames[-1][2]-frames[0][2])*1000
    fig.suptitle("%s   (rise %+.0f mm)"%(title,rise),fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig(path,dpi=115); plt.close(fig)
    print("wrote",path,"rise %+.0f mm"%rise)

if __name__=="__main__":
    render_trial(dict(M.BASE),"mj_success.png","MuJoCo 3D — grasp SUCCESS")
    s=dict(M.BASE); s.update(dict(mu=0.12)); render_trial(s,"mj_slip.png","MuJoCo 3D — SLIP (mu=0.12)")
