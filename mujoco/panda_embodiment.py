"""
panda_embodiment.py — SECOND embodiment for the Factorized Interaction World Model.

Real Franka Panda (7-DOF arm + tendon-coupled hand, mujoco_menagerie) grasps the
SAME object the floating gripper grasps. The whole point: a genuinely different
mass matrix M  =>  a different analytical coupling  W = J M⁻¹ Jᵀ  at the contact,
while the LOCAL contact situation (object-frame geometry, μ, compliance) — i.e.
z_local — is schema-identical. That is the first real cross-embodiment test of the
quotient-collapse + transfer certificate.

Reuses contact_probe.py and z_local_schema.py UNCHANGED (that reuse is the claim).

Grasp control = damped-least-squares IK on the hand body (position + fixed
downward orientation) computed kinematically, then commanded to the Panda's
position-servo actuators; tendon actuator8 closes the gripper (255=open -> 0=closed).
"""
import os, numpy as np, mujoco
import contact_probe as cp
import z_local_schema as Z

# Path to mujoco_menagerie's franka_emika_panda. Override with $MENAGERIE_DIR
# (pointing at a mujoco_menagerie checkout) or $FRANKA_DIR (the panda dir itself).
_MEN = os.environ.get("MENAGERIE_DIR")
FRANKA_DIR = os.environ.get("FRANKA_DIR") or (
    os.path.join(_MEN, "franka_emika_panda") if _MEN
    else os.path.join(os.path.dirname(__file__), "menagerie", "franka_emika_panda"))
PANDA_XML  = os.path.join(FRANKA_DIR, "panda.xml")

# object defaults mirror m2_floating_gripper_grasp.BASE (same object!)
BASE = dict(d=0.05, h=0.08, mass=0.05, mu=1.0, force=15.0)   # 'force' unused (servo grip), kept for parity
OBJ_X, OBJ_Y = 0.50, 0.0
PED_TOP = 0.45                      # pedestal top height (object sits on it)


def _wrapper_xml(params):
    """Wrapper that includes the real panda.xml and adds pedestal + object.
    Written INTO the franka dir so `include` and meshdir='assets' resolve."""
    d, h, mass, mu = params["d"], params["h"], params["mass"], params["mu"]
    oz = PED_TOP + h/2 + 0.001
    return f"""<mujoco model="panda_grasp">
  <include file="panda.xml"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"
          solver="Newton" iterations="200" tolerance="1e-12"/>
  <worldbody>
    <geom name="floor2" type="plane" size="2 2 0.1" pos="0 0 0" contype="1" conaffinity="1"/>
    <body name="pedestal" pos="{OBJ_X} {OBJ_Y} {PED_TOP/2}">
      <geom name="pedestal" type="box" size="0.06 0.06 {PED_TOP/2}" rgba="0.5 0.4 0.3 1"
            friction="1.5 0.02 0.001"/>
    </body>
    <body name="object" pos="{OBJ_X} {OBJ_Y} {oz}">
      <freejoint name="objfree"/>
      <geom name="object" type="cylinder" size="{d/2} {h/2}" mass="{mass}"
            friction="{mu} 0.02 0.001" rgba="0.2 0.6 0.9 1"
            solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>"""


def make_model(params):
    path = os.path.join(FRANKA_DIR, "_wrap_grasp.xml")
    with open(path, "w") as f:
        f.write(_wrapper_xml(params))
    m = mujoco.MjModel.from_xml_path(path)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    return m


# ----------------------------------------------------------------------
# damped-least-squares IK on the hand body (position + orientation)
# ----------------------------------------------------------------------
def ik_hand(m, q_init, target_pos, target_quat, arm_dofs=7, iters=200, damp=1e-2):
    """Kinematic DLS IK: return qpos (nq,) placing the hand at target_pos/quat,
    moving only the 7 arm joints. Fingers held at q_init."""
    d = mujoco.MjData(m)
    d.qpos[:] = q_init
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        pos = np.array(d.xpos[hb]); quat = np.array(d.xquat[hb])
        dp = target_pos - pos
        dq = np.zeros(3)
        qc = quat.copy(); qc[1:] *= -1
        dquat = np.zeros(4); mujoco.mju_mulQuat(dquat, target_quat, qc)
        mujoco.mju_quat2Vel(dq, dquat, 1.0)
        err = np.concatenate([dp, dq])
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jacBody(m, d, jacp, jacr, hb)
        J = np.vstack([jacp[:, :arm_dofs], jacr[:, :arm_dofs]])   # 6 x 7
        dqj = J.T @ np.linalg.solve(J @ J.T + damp*np.eye(6), err)
        d.qpos[:arm_dofs] += dqj
        # clamp to joint ranges
        for j in range(arm_dofs):
            lo, hi = m.jnt_range[j]
            if lo < hi:
                d.qpos[j] = np.clip(d.qpos[j], lo, hi)
    return np.array(d.qpos).copy()


HOME_ARM = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])

def set_home(m, d):
    """Reset to XML defaults (object freejoint keeps its pose) then place the 9
    Panda dofs at the home pose by joint address — NOT mj_resetDataKeyframe, which
    would clobber the object freejoint (keyframe only sizes the 9 arm dofs)."""
    mujoco.mj_resetData(m, d)
    names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7",
             "finger_joint1","finger_joint2"]
    for nm, val in zip(names, HOME_ARM):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)


def _finger_geoms(m):
    lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    fg = set()
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue                          # visual-only
        if m.geom_bodyid[g] in (lf, rf):
            fg.add(g)
    return fg


def run_grasp(params, save_render=False, on_step=None, verbose=False, hoff=0.0):
    m = make_model(params)
    d = mujoco.MjData(m)
    set_home(m, d)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
    home_quat = np.array(d.xquat[hb]).copy()  # keep hand pointing down
    oz = PED_TOP + params["h"]/2 + 0.001
    # main fingertip pad sits 0.075 below the hand; center it on the object mid-height.
    # hoff shifts the grasp up/down the cylinder to sample different surface heights.
    grasp_pos = np.array([OBJ_X, OBJ_Y, oz + 0.075 + hoff])
    pre_pos   = grasp_pos + np.array([0, 0, 0.12])

    q_home = np.array(d.qpos).copy()
    q_pre   = ik_hand(m, q_home, pre_pos,   home_quat)
    q_grasp = ik_hand(m, q_pre,  grasp_pos, home_quat)

    GRIP_OPEN, GRIP_CLOSE = 255.0, 0.0
    def command(qarm, grip):
        d.ctrl[:7] = qarm[:7]; d.ctrl[7] = grip
    def maybe_log(phase):
        if on_step is None: return
        W, J = cp.assemble_W(m, d)
        on_step(m, d, W, obj_bid, obj_gid, phase)

    # phase timings
    def servo(qA, qB, grip, n, phase):
        for k in range(n):
            a = (k+1)/n
            command(qA*(1-a) + qB*a, grip)
            mujoco.mj_step(m, d)
            if on_step is not None: maybe_log(phase)

    servo(q_home, q_pre,   GRIP_OPEN, 150, "approach")
    servo(q_pre,  q_grasp, GRIP_OPEN, 150, "descend")
    # close gripper (hold arm at grasp)
    for k in range(200):
        a=(k+1)/200; command(q_grasp, GRIP_OPEN*(1-a)+GRIP_CLOSE*a); mujoco.mj_step(m,d)
        if on_step is not None: maybe_log("close")
    for k in range(150):
        command(q_grasp, GRIP_CLOSE); mujoco.mj_step(m,d)
        if on_step is not None: maybe_log("hold")
    # lift: raise the pre pose target with gripper closed
    q_lift = ik_hand(m, q_grasp, grasp_pos+np.array([0,0,0.18]), home_quat)
    for k in range(250):
        a=(k+1)/250; command(q_grasp*(1-a)+q_lift*a, GRIP_CLOSE); mujoco.mj_step(m,d)
        if on_step is not None: maybe_log("lift")

    z_final = float(d.xpos[obj_bid][2])
    lifted = z_final - oz
    if verbose:
        print("  object z: start=%.3f final=%.3f  lift=%.3f m" % (oz, z_final, lifted))
    return dict(lift=lifted, z_final=z_final, oz=oz, model=m, data=d)


if __name__ == "__main__":
    r = run_grasp(dict(BASE), verbose=True)
    print("Panda grasp lift = %.3f m  -> %s" % (r["lift"], "GRASPED+LIFTED" if r["lift"] > 0.05 else "no lift"))
