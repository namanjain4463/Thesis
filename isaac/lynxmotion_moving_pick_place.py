# ======================================================================
# LYNXMOTION SES-PRO 550 + CGE-10-10  —  ON-THE-FLY MOVING PICK & PLACE
# Isaac Sim 5.1  |  Script Editor  |  run with Ctrl+Enter
# ======================================================================
#
# Sequence (deterministic, no random motion, stays clear of belt/rails):
#   1. Stop -> reconfigure the 3-finger gripper drives -> Play
#      (removes the mimic-vs-drive fight that desynced the fingers)
#   2. Gripper SELF-TEST: prove symmetric open/close, else ABORT with a fix
#   3. Move to a FIXED known-good READY pose (no random IK search)
#   4. HOVER over the cube lane at the robot's own X (the best-reach
#      intercept) -- we do NOT chase far upstream, which would push the
#      550 mm arm past its comfortable reach and break the IK
#   5. As the cube arrives: descend, track it, and MATCH its velocity so
#      the close happens as if the cube were standing still
#   6. Close the 3 fingers while still tracking
#   7. QUICK LIFT and VERIFY the cube actually rose. If not -> open,
#      return to READY, STOP (never transfer a cube we did not grasp)
#   8. Up-and-over transfer to the pedestal (joint-space between poses
#      solved once -> no snapping), lower, release, retreat
#
# Anti-snap:  one fixed tool orientation + a per-frame joint RATE LIMIT
#             (caps speed, always makes progress -> smooth, never freezes)
# ======================================================================

import asyncio
import time
import traceback

import numpy as np

import omni.kit.app
import omni.timeline
import omni.usd

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver


print("\n\n[LYNX] ================================================")
print("[LYNX] moving pick & place script LOADED — starting up")
print("[LYNX] ================================================")


# ======================================================================
# 1. USER TUNABLES  (edit these; everything else is derived live)
# ======================================================================

# ---- Local Lula config (EDIT if your paths differ) -------------------
CFG_ROOT = (
    "/home/njain376/Desktop/Thesis/"
    "Lynxmotion-SES-Pro-550mm-6DOF-Robot-Arm/"
    "SES-P-ROS2-Arms/isaac/config/lynxmotion"
)
ROBOT_DESCRIPTION = CFG_ROOT + "/robot_description.yaml"
URDF              = CFG_ROOT + "/lynxmotion_cge1010.urdf"

# ---- Scene prim paths (factory_cell_grasp_lift_v05) ------------------
ROBOT_PATH = "/World/Lynxmotion/root_joint"
BASE_PATH  = "/World/Lynxmotion"
EE_PATH    = "/World/Lynxmotion/gripper/pro_arm_ee"
CUBE_PATH  = "/World/Cube"
PEDESTAL_PATH = "/World/Pedestal"
EE_FRAME   = "pro_arm_ee"

# ---- Articulation layout ---------------------------------------------
ARM_INDICES     = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
GRIPPER_INDICES = np.array([6, 7, 8], dtype=np.int32)   # joint_7,8,9
FINGER_JOINT_NAMES = ["joint_7", "joint_8", "joint_9"]  # 7 = master

# ---- Known-good READY branch (deg) -----------------------------------
READY_Q_DEG = [-75.455, -36.334, 11.833, 3.131, 84.152, -97.696]

# ---- Gripper aperture (joint_7 stroke 0 .. +0.010 m) -----------------
#   OPEN  : q7 = +0.010 -> q8 = q9 = -0.010
#   CLOSED: q7 =  0.000 -> q8 = q9 =  0.000
GRIPPER_OPEN_Q7   = 0.010
GRIPPER_CLOSED_Q7 = 0.000

# ---- Finger drive gains applied at startup (the real grasp fix) -------
FINGER_DRIVE_STIFFNESS = 5.0e4     # N/m
FINGER_DRIVE_DAMPING   = 8.0e2     # N/(m/s)
FINGER_MAX_FORCE       = 20.0      # N  (grip force cap; raise if it slips)

# ---- Heights (m) ------------------------------------------------------
READY_HEIGHT = 0.220               # EE above cube at READY
HOVER_HEIGHT = 0.060               # EE hovers this far above cube at the intercept
GRASP_Z_BIAS = -0.008              # final TCP Z offset at grasp (tune +/- few mm)

# ---- Intercept / capture ---------------------------------------------
#   The arm hovers over the lane at the robot's own X (best reach) and
#   grabs the cube as it arrives, matching the cube's velocity.  We do NOT
#   chase far upstream -- that is what over-extended the arm before.
TAU_CAPTURE    = 0.05              # s  aim this far ahead of the cube (velocity match)
LEAD_X         = 0.020             # m  small +X lead early in approach, decays to 0
MAX_JOINT_STEP = 0.060             # rad/frame command cap (rate limit -> smooth, no snap)

# ---- Zone geometry, RELATIVE to robot-base X (= intercept X) (m) ------
APPROACH_START_OFFSET = -0.120     # begin descend + track when cube reaches here
CAPTURE_START_OFFSET  = -0.030     # allowed to accept the capture after here
DEADLINE_OFFSET       = +0.120     # give up (do NOT chase downstream) past here

# ---- Capture gate -----------------------------------------------------
CAP_XY_TOL   = 0.008               # m
CAP_Z_TOL    = 0.008               # m
CAP_RELV_TOL = 0.035               # m/s  relative EE-cube speed
REQUIRED_GOOD_FRAMES = 4

# ---- Closing ----------------------------------------------------------
CLOSE_FRAMES    = 18
POST_CLOSE_HOLD = 12

# ---- Quick lift + success gate ---------------------------------------
QUICK_LIFT_HEIGHT = 0.080
QUICK_LIFT_FRAMES = 45
CUBE_RISE_GATE    = 0.015
FINAL_RISE_GATE   = 0.040

# ---- Transfer to pedestal --------------------------------------------
CARRY_HEIGHT    = 0.230
PLACE_CLEARANCE = 0.004
TRANSFER_FRAMES = 150
PLACE_FRAMES    = 90
RETREAT_FRAMES  = 70

# ---- Motion timing / IK ----------------------------------------------
READY_MOVE_FRAMES = 120
HOVER_MOVE_FRAMES = 70
IK_POS_TOL   = 0.002
IK_ORI_TOL   = 0.30                # rad (loose -> IK stays solvable near reach limit)
MAX_IK_FAILS = 30
MIN_UPSTREAM = 0.30


# ======================================================================
# 2. SMALL HELPERS
# ======================================================================

async def next_frame():
    await omni.kit.app.get_app().next_update_async()

async def wait_frames(n):
    for _ in range(n):
        await next_frame()

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def as64(a):
    return np.asarray(a, dtype=np.float64)

def gripper_targets(q7):
    """Master + mirrored followers: q7 -> [q7, -q7, -q7]."""
    return np.array([+q7, -q7, -q7], dtype=np.float32)

def rot_to_quat(R):
    """3x3 rotation matrix -> (w, x, y, z) quaternion.  Lula FK returns a
    matrix but Lula IK expects a wxyz quaternion for target_orientation."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S; x = (m21-m12)/S; y = (m02-m20)/S; z = (m10-m01)/S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21-m12)/S; x = 0.25*S; y = (m01+m10)/S; z = (m02+m20)/S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02-m20)/S; x = (m01+m10)/S; y = 0.25*S; z = (m12+m21)/S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10-m01)/S; x = (m02+m20)/S; y = (m12+m21)/S; z = 0.25*S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)

async def interp(cmd_arm, cmd_grip, q0, q1, frames, grip_q7):
    """Smooth joint-space interpolation between two configs."""
    q0 = as64(q0); q1 = as64(q1)
    for f in range(frames):
        a = smoothstep((f + 1) / frames)
        cmd_arm(q0 + a * (q1 - q0))
        cmd_grip(grip_q7)
        await next_frame()


# ======================================================================
# 3. GRIPPER DRIVE / MIMIC CONFIGURATION  (run while STOPPED)
# ======================================================================

def find_finger_joint_prims(stage):
    found = {}
    root = stage.GetPrimAtPath(BASE_PATH)
    if not root.IsValid():
        return found
    for prim in Usd.PrimRange(root):
        if prim.GetName() in FINGER_JOINT_NAMES and prim.IsA(UsdPhysics.PrismaticJoint):
            found[prim.GetName()] = prim
    if len(found) < 3:
        for prim in Usd.PrimRange(root):
            n = prim.GetName()
            if n in FINGER_JOINT_NAMES and n not in found:
                found[n] = prim
    return found

def try_remove_mimic(prim):
    out = []
    try:
        for s in list(prim.GetAppliedSchemas()):
            if "MimicJoint" not in s:
                continue
            inst = s.split(":", 1)[1] if ":" in s else ""
            ok = False
            try:
                ok = bool(prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, inst)
                          if inst else prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI))
            except Exception:
                ok = False
            if not ok:
                try:
                    ok = bool(prim.RemoveAppliedSchema(s))
                except Exception:
                    ok = False
            out.append((s, ok))
    except Exception as exc:
        print("[GRIPPER]   mimic inspect error:", exc)
    return out

def set_linear_drive(prim, k, d, fmax):
    drive = UsdPhysics.DriveAPI.Get(prim, "linear") or UsdPhysics.DriveAPI.Apply(prim, "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(k))
    drive.CreateDampingAttr().Set(float(d))
    drive.CreateMaxForceAttr().Set(float(fmax))

def configure_gripper(stage):
    print("\n[GRIPPER] configuring finger drives (sim stopped)...")
    prims = find_finger_joint_prims(stage)
    if len(prims) < 3:
        print("[GRIPPER]   WARNING found only", list(prims.keys()))
    for name in FINGER_JOINT_NAMES:
        prim = prims.get(name)
        if prim is None:
            continue
        rm = try_remove_mimic(prim)
        if rm:
            print(f"[GRIPPER]   {name}: mimic ->", rm)
        try:
            set_linear_drive(prim, FINGER_DRIVE_STIFFNESS, FINGER_DRIVE_DAMPING,
                             FINGER_MAX_FORCE)
            print(f"[GRIPPER]   {name}: drive set (k={FINGER_DRIVE_STIFFNESS:.0f}, "
                  f"d={FINGER_DRIVE_DAMPING:.0f}, Fmax={FINGER_MAX_FORCE:.0f})")
        except Exception as exc:
            print(f"[GRIPPER]   {name}: drive set FAILED:", exc)


# ======================================================================
# 4. GEOMETRY
# ======================================================================

def world_bbox(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError("Invalid prim: " + prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    dims = np.array([hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]], dtype=np.float64)
    center = np.array([(hi[0]+lo[0])/2, (hi[1]+lo[1])/2, (hi[2]+lo[2])/2], dtype=np.float64)
    return dims, center, float(hi[2])


# ======================================================================
# 5. MAIN
# ======================================================================

async def moving_pick_place():
    print("\n" + "=" * 70)
    print("LYNXMOTION — ON-THE-FLY MOVING PICK & PLACE")
    print("=" * 70)

    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()

    # (a) STOP -> configure gripper -> PLAY
    if not timeline.is_stopped():
        print("\n[1] stopping timeline to configure gripper...")
        timeline.stop(); await wait_frames(5)
    configure_gripper(stage)
    print("\n[1] starting simulation...")
    timeline.play(); await wait_frames(30)

    # (b) init robot + prims
    robot = SingleArticulation(prim_path=ROBOT_PATH, name="lynx_mpp")
    robot.initialize()
    base_prim = SingleXFormPrim(prim_path=BASE_PATH, name="lynx_base")
    ee_prim   = SingleXFormPrim(prim_path=EE_PATH,   name="lynx_ee")
    cube_prim = SingleXFormPrim(prim_path=CUBE_PATH, name="lynx_cube")

    print("\n[2] DOF order:", list(robot.dof_names))
    base_pos, base_quat = base_prim.get_world_pose()
    base_pos, base_quat = as64(base_pos), as64(base_quat)
    cube_start, _ = cube_prim.get_world_pose(); cube_start = as64(cube_start)
    cube_dims, _, _ = world_bbox(CUBE_PATH)
    cube_z0 = float(cube_start[2]); lane_y = float(cube_start[1])
    print("    base :", np.round(base_pos, 4))
    print("    cube :", np.round(cube_start, 4), " dims(mm):", np.round(cube_dims*1000, 1))

    _, ped_center, ped_top_z = world_bbox(PEDESTAL_PATH)
    place_pos = np.array([ped_center[0], ped_center[1],
                          ped_top_z + cube_dims[2]/2.0 + PLACE_CLEARANCE])
    print("    pedestal top Z:", round(ped_top_z, 4), " -> place:", np.round(place_pos, 4))

    pick_x    = float(base_pos[0])                 # intercept X (best reach)
    approach_x = pick_x + APPROACH_START_OFFSET
    capture_x = pick_x + CAPTURE_START_OFFSET
    dead_x    = pick_x + DEADLINE_OFFSET
    print("    intercept X:", round(pick_x, 4), " approach@", round(approach_x, 4),
          " capture@", round(capture_x, 4), " deadline@", round(dead_x, 4))
    print("    lane Y offset from base: %.3f m  (arm reach check)"
          % (lane_y - float(base_pos[1])))

    upstream = pick_x - float(cube_start[0])
    print("    cube upstream distance:", round(upstream, 4), "m")
    if upstream < MIN_UPSTREAM:
        print("\n[ABORT] cube not far enough upstream — stop the sim so it "
              "resets, then re-run.")
        return

    # (c) Lula
    solver = LulaKinematicsSolver(robot_description_path=ROBOT_DESCRIPTION, urdf_path=URDF)
    solver.set_robot_base_pose(base_pos, base_quat)

    def command_arm(q):
        robot.apply_action(ArticulationAction(
            joint_positions=np.asarray(q, dtype=np.float32), joint_indices=ARM_INDICES))

    def command_gripper(q7):
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets(q7), joint_indices=GRIPPER_INDICES))

    def read_gripper():
        return as64(robot.get_joint_positions())[GRIPPER_INDICES]

    def solve(pos, R, warm):
        if R is None:
            q, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME, target_position=as64(pos),
                target_orientation=None, warm_start=as64(warm),
                position_tolerance=IK_POS_TOL)
        else:
            q, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME, target_position=as64(pos),
                target_orientation=as64(R), warm_start=as64(warm),
                position_tolerance=IK_POS_TOL, orientation_tolerance=IK_ORI_TOL)
        return as64(q), bool(ok)

    # (d) GRIPPER SELF-TEST
    print("\n[3] GRIPPER SELF-TEST")
    for _ in range(25): command_gripper(GRIPPER_OPEN_Q7); await next_frame()
    q_open = read_gripper()
    for _ in range(25): command_gripper(GRIPPER_CLOSED_Q7); await next_frame()
    q_closed = read_gripper()
    for _ in range(20): command_gripper(GRIPPER_OPEN_Q7); await next_frame()
    q_reopen = read_gripper()
    print("    OPEN  :", np.round(q_open*1000, 2), "mm")
    print("    CLOSED:", np.round(q_closed*1000, 2), "mm")
    print("    REOPEN:", np.round(q_reopen*1000, 2), "mm")

    def symmetric(q, tol=0.0015):
        return abs(q[1]+q[0]) < tol and abs(q[2]+q[0]) < tol
    if not (symmetric(q_open) and q_open[0] > 0.006 and
            symmetric(q_closed) and q_closed[0] < 0.004):
        print("\n[ABORT] gripper not symmetric — remove the Physx Mimic Joint API")
        print("        from joint_8 & joint_9 in the USD, save, stop, re-run.")
        return
    print("    --> gripper OK (symmetric open & close)")

    # (e) fixed grasp orientation + tracking helpers
    ready_q = np.deg2rad(READY_Q_DEG)
    _, R_grasp_mat = solver.compute_forward_kinematics(EE_FRAME, ready_q)
    R_grasp = rot_to_quat(R_grasp_mat)
    print("\n[4] grasp orientation locked (wxyz):", np.round(R_grasp, 4))
    pick_probe = np.array([pick_x, lane_y, cube_z0 + GRASP_Z_BIAS])
    _, ok = solve(pick_probe, R_grasp, ready_q)
    R_track = R_grasp if ok else None
    if not ok:
        print("    fixed-orientation IK failed at pick point -> position-only tracking")

    def solve_track(pos, seed):
        q, ok = solve(pos, R_track, seed)
        if not ok and R_track is not None:
            q, ok = solve(pos, None, seed)          # reach fallback
        return q, ok

    def step_toward(seed, q_new):
        d = q_new - seed
        m = float(np.max(np.abs(d)))
        if m > MAX_JOINT_STEP:
            d = d * (MAX_JOINT_STEP / m)
        return seed + d

    # (f) open + move to READY
    print("\n[5] opening gripper + moving to READY")
    q_now = as64(robot.get_joint_positions())[:6]
    await interp(command_arm, command_gripper, q_now, ready_q, READY_MOVE_FRAMES,
                 GRIPPER_OPEN_Q7)

    # (f2) move to HOVER over the intercept (best-reach)
    intercept = np.array([pick_x, lane_y, cube_z0 + HOVER_HEIGHT])
    q_hover, ok = solve(intercept, R_track, ready_q)
    if not ok:
        q_hover, ok = solve(intercept, None, ready_q)
    if not ok:
        print("\n[ABORT] cannot reach the hover/intercept pose — the cube lane "
              "is likely beyond the arm's reach. Move the robot closer to the "
              "conveyor (or the conveyor toward the robot) and re-run.")
        return
    await interp(command_arm, command_gripper, ready_q, q_hover, HOVER_MOVE_FRAMES,
                 GRIPPER_OPEN_Q7)
    q_seed = q_hover.copy()
    print("\n[6] hovering at intercept — waiting for cube")

    cube_p, _ = cube_prim.get_world_pose()
    if float(as64(cube_p)[0]) >= capture_x:
        print("[ABORT] cube already at capture zone before hover was ready — "
              "stop sim (cube resets) and re-run.")
        return

    # (g) WAIT until cube reaches the approach zone
    while True:
        cube_p, _ = cube_prim.get_world_pose()
        command_arm(q_seed); command_gripper(GRIPPER_OPEN_Q7)
        if float(as64(cube_p)[0]) >= approach_x:
            break
        await next_frame()

    # (h) APPROACH: descend + track + match velocity -> capture gate
    print("\n[7] APPROACH  (descend, track, match cube speed)")
    prev_cube = as64(cube_p); prev_ee, _ = ee_prim.get_world_pose(); prev_ee = as64(prev_ee)
    prev_t = time.monotonic(); v_cube = np.zeros(3); v_ee = np.zeros(3)
    ik_fails = 0; good = 0; frame = 0

    while True:
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        cube_x = float(cube_p[0])
        now = time.monotonic(); dt = max(now - prev_t, 1e-3)
        v_cube = 0.7*v_cube + 0.3*((cube_p - prev_cube)/dt)
        prev_cube, prev_t = cube_p.copy(), now

        # descend HOVER_HEIGHT -> GRASP_Z_BIAS as the cube crosses approach->capture
        prog = np.clip((cube_x - approach_x) / max(capture_x - approach_x, 1e-6), 0, 1)
        hz = HOVER_HEIGHT + smoothstep(prog) * (GRASP_Z_BIAS - HOVER_HEIGHT)
        lead = LEAD_X * (1.0 - prog)                 # small +X lead, decays to 0

        target = cube_p.copy()
        target[0] = cube_p[0] + v_cube[0]*TAU_CAPTURE + lead
        target[1] = cube_p[1] + v_cube[1]*TAU_CAPTURE
        target[2] = cube_p[2] + hz

        q_new, ok = solve_track(target, q_seed)
        if ok:
            q_seed = step_toward(q_seed, q_new); ik_fails = 0
        else:
            ik_fails += 1
        command_arm(q_seed); command_gripper(GRIPPER_OPEN_Q7)
        await next_frame()

        ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
        v_ee = 0.7*v_ee + 0.3*((ee_p - prev_ee)/dt); prev_ee = ee_p.copy()
        grasp_pt = cube_p.copy(); grasp_pt[2] = cube_p[2] + GRASP_Z_BIAS
        err = ee_p - grasp_pt
        exy = float(np.linalg.norm(err[:2])); ez = float(err[2])
        relv = float(np.linalg.norm(v_ee - v_cube))

        if frame % 6 == 0:
            print(f"    x={cube_x:.3f} h={hz*1000:+5.1f}mm exy={exy*1000:5.1f} "
                  f"ez={ez*1000:+5.1f}mm vcx={v_cube[0]:+.3f} vex={v_ee[0]:+.3f} "
                  f"relv={relv*1000:5.1f}mm/s good={good}")

        aligned = exy < CAP_XY_TOL and abs(ez) < CAP_Z_TOL and relv < CAP_RELV_TOL
        good = good + 1 if (cube_x >= capture_x and aligned) else 0
        if good >= REQUIRED_GOOD_FRAMES:
            print("\n    >>> CAPTURE ALIGNED  exy=%.1fmm ez=%.1fmm relv=%.1fmm/s"
                  % (exy*1000, ez*1000, relv*1000))
            break
        if ik_fails > MAX_IK_FAILS:
            print("\n[ABORT] repeated IK failure — pick point may be out of reach.")
            return
        if cube_x >= dead_x:
            print("\n[ABORT] cube passed deadline without alignment "
                  "(exy=%.1fmm relv=%.1fmm/s). Not chasing downstream."
                  % (exy*1000, relv*1000))
            return
        frame += 1

    # (i) CLOSE while tracking (velocity matched)
    print("\n[8] CLOSING (still tracking)")
    for f in range(CLOSE_FRAMES):
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        now = time.monotonic(); dt = max(now - prev_t, 1e-3)
        v_cube = 0.7*v_cube + 0.3*((cube_p - prev_cube)/dt); prev_cube, prev_t = cube_p.copy(), now
        target = cube_p.copy()
        target[0] += v_cube[0]*TAU_CAPTURE; target[1] += v_cube[1]*TAU_CAPTURE
        target[2] += GRASP_Z_BIAS
        q_new, ok = solve_track(target, q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm(q_seed)
        a = smoothstep((f + 1) / CLOSE_FRAMES)
        command_gripper(GRIPPER_OPEN_Q7 + a*(GRIPPER_CLOSED_Q7 - GRIPPER_OPEN_Q7))
        await next_frame()

    for _ in range(POST_CLOSE_HOLD):
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        target = cube_p.copy(); target[2] += GRASP_Z_BIAS
        q_new, ok = solve_track(target, q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm(q_seed); command_gripper(GRIPPER_CLOSED_Q7); await next_frame()

    print("    gripper after close:", np.round(read_gripper()*1000, 2), "mm")

    # (j) QUICK LIFT + verify rise
    print("\n[9] QUICK LIFT")
    ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
    lift_start = ee_p.copy(); lift_end = lift_start + np.array([0, 0, QUICK_LIFT_HEIGHT])
    for f in range(QUICK_LIFT_FRAMES):
        a = smoothstep((f + 1) / QUICK_LIFT_FRAMES)
        q_new, ok = solve_track(lift_start + a*(lift_end - lift_start), q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm(q_seed); command_gripper(GRIPPER_CLOSED_Q7); await next_frame()

    cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
    rise = float(cube_p[2] - cube_z0)
    print("    cube rise after quick lift: %.1f mm" % (rise*1000))
    if rise < CUBE_RISE_GATE:
        print("\n[STOP] cube did NOT rise (%.1f < %.1f mm gate). Grasp failed — "
              "not transferring." % (rise*1000, CUBE_RISE_GATE*1000))
        for _ in range(20): command_gripper(GRIPPER_OPEN_Q7); command_arm(q_seed); await next_frame()
        q0 = as64(robot.get_joint_positions())[:6]
        await interp(command_arm, command_gripper, q0, ready_q, READY_MOVE_FRAMES, GRIPPER_OPEN_Q7)
        print("\n[NEXT] try, in order:")
        print("   1. sweep GRASP_Z_BIAS a few mm (fingers slightly low/high)")
        print("   2. if the close bottoms out with no squeeze, resize cube 20->25 mm")
        print("   3. raise FINGER_MAX_FORCE (grip too weak)")
        return
    print("    grasp CONFIRMED (cube is being carried).")

    # (k) TRANSFER up-and-over (joint-space, solved once)
    print("\n[10] TRANSFER to pedestal")
    carry_z = cube_z0 + CARRY_HEIGHT

    def solve_key(pos, warm, label):
        q, ok = solve(pos, R_grasp, warm)
        if not ok:
            q, ok = solve(pos, None, warm)
        if not ok:
            print("    [WARN] IK failed for %s at %s" % (label, np.round(pos, 3)))
            return warm, False
        return q, True

    ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
    q_up,   _ = solve_key(np.array([ee_p[0], ee_p[1], carry_z]), q_seed, "lift-high")
    q_over, _ = solve_key(np.array([place_pos[0], place_pos[1], carry_z]), q_up, "over-pedestal")
    q_place, _ = solve_key(place_pos, q_over, "place")

    await interp(command_arm, command_gripper, q_seed, q_up,   TRANSFER_FRAMES // 2, GRIPPER_CLOSED_Q7)
    await interp(command_arm, command_gripper, q_up,   q_over, TRANSFER_FRAMES,      GRIPPER_CLOSED_Q7)
    await interp(command_arm, command_gripper, q_over, q_place, PLACE_FRAMES,        GRIPPER_CLOSED_Q7)

    # (l) RELEASE + RETREAT
    print("\n[11] RELEASE + RETREAT")
    for _ in range(25): command_arm(q_place); command_gripper(GRIPPER_OPEN_Q7); await next_frame()
    await interp(command_arm, command_gripper, q_place, q_over, RETREAT_FRAMES, GRIPPER_OPEN_Q7)
    q_end = as64(robot.get_joint_positions())[:6]
    await interp(command_arm, command_gripper, q_end, ready_q, READY_MOVE_FRAMES, GRIPPER_OPEN_Q7)

    # (m) report
    cube_f, _ = cube_prim.get_world_pose(); cube_f = as64(cube_f)
    xy_err = float(np.linalg.norm(cube_f[:2] - place_pos[:2]))
    z_err = float(abs(cube_f[2] - place_pos[2]))
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print("  cube final     :", np.round(cube_f, 4))
    print("  pedestal target:", np.round(place_pos, 4))
    print("  place XY error : %.1f mm   Z error: %.1f mm" % (xy_err*1000, z_err*1000))
    print("  quick-lift rise: %.1f mm (clean-pick gate %.0f mm)"
          % (rise*1000, FINAL_RISE_GATE*1000))
    print("\n  OVERALL: %s" %
          ("PICK & PLACE SUCCESS" if (xy_err < 0.03 and z_err < 0.03)
           else "placed, but check pedestal error above"))
    print("=" * 70)


# ======================================================================
# 6. LAUNCH (Ctrl+Enter; surfaces async exceptions with traceback)
# ======================================================================

def _done(task):
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        print("[LYNX] task cancelled"); return
    if exc:
        print("\n[LYNX] !!! task raised an exception:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    else:
        print("[LYNX] task finished cleanly")

try:
    if "_mpp_task" in globals() and not _mpp_task.done():
        _mpp_task.cancel()
except Exception:
    pass

_mpp_task = asyncio.ensure_future(moving_pick_place())
_mpp_task.add_done_callback(_done)
print("[LYNX] task scheduled — running now")
