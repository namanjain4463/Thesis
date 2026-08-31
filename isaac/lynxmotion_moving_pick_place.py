# ======================================================================
# LYNXMOTION SES-PRO 550 + CGE-10-10  —  ON-THE-FLY MOVING PICK & PLACE
# Isaac Sim 5.1  |  Script Editor  |  run with Ctrl+Enter
# ======================================================================
#
# WHAT THIS SCRIPT DOES (deterministic, no random motion):
#
#   1. Stop -> reconfigure the 3-finger gripper drives -> Play.
#      (fixes the mimic-vs-drive fight that made the fingers desync)
#   2. Gripper SELF-TEST: open/close and prove q7/q8/q9 stay symmetric.
#      If the gripper cannot close symmetrically, we ABORT here and tell
#      you exactly what to change — no wasted grasp attempt.
#   3. Move to a FIXED known-good READY pose (no random IK search).
#   4. Catch up to the moving cube: EE deliberately runs faster than the
#      cube, then matches the cube's velocity so the close happens as if
#      the cube were standing still (feed-forward on measured cube speed).
#   5. Close the 3 fingers around the cube while still tracking it.
#   6. QUICK LIFT and VERIFY the cube actually rose. If it did not rise,
#      we STOP (open, return to READY) — we never "transfer" a cube we
#      did not grasp.
#   7. Only if the cube rose: up-and-over transfer to the pedestal using
#      joint-space interpolation between poses solved ONCE (no per-frame
#      free IK -> no snapping, no branch changes, stays clear of the belt
#      and rails), lower, release, retreat.
#
# ANTI-SNAP / ANTI-COLLISION PRINCIPLES:
#   * One FIXED tool orientation for the whole pick phase.
#   * Warm-started full-pose IK during tracking (deterministic branch).
#   * Transfer = joint-space interpolation between a handful of solved
#     key poses, each lifted high above obstacles.
#   * The only intended contact is fingers <-> cube.
#
# The script reads the cube / pedestal / base poses LIVE, so you do not
# have to hard-code coordinates.
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

# ---- Scene prim paths (from factory_cell_grasp_lift_v05) --------------
ROBOT_PATH = "/World/Lynxmotion/root_joint"     # articulation root
BASE_PATH  = "/World/Lynxmotion"                # robot base xform
EE_PATH    = "/World/Lynxmotion/gripper/pro_arm_ee"
CUBE_PATH  = "/World/Cube"
PEDESTAL_PATH = "/World/Pedestal"
EE_FRAME   = "pro_arm_ee"                        # Lula frame name

# ---- Articulation layout ---------------------------------------------
ARM_INDICES     = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
GRIPPER_INDICES = np.array([6, 7, 8], dtype=np.int32)   # joint_7,8,9
FINGER_JOINT_NAMES = ["joint_7", "joint_8", "joint_9"]  # 7 = master

# ---- Known-good READY branch (deg) — from your commissioning ----------
READY_Q_DEG = [-75.455, -36.334, 11.833, 3.131, 84.152, -97.696]

# ---- Gripper aperture (joint_7 stroke is 0 .. +0.010 m) --------------
#   OPEN  : q7 = +0.010  -> q8 = q9 = -0.010
#   CLOSED: q7 =  0.000  -> q8 = q9 =  0.000
GRIPPER_OPEN_Q7   = 0.010
GRIPPER_CLOSED_Q7 = 0.000

# ---- Finger drive gains applied at startup (the real fix) ------------
#   Stiff position drive so the fingers actually hold their target, with
#   a capped force so a fully-commanded close squeezes with ~FINGER_MAX_
#   FORCE newtons instead of ejecting the cube.
FINGER_DRIVE_STIFFNESS = 5.0e4     # N/m
FINGER_DRIVE_DAMPING   = 8.0e2     # N/(m/s)
FINGER_MAX_FORCE       = 20.0      # N   (grip force cap; raise if it slips)

# ---- Heights (m) ------------------------------------------------------
READY_HEIGHT = 0.220               # EE above cube at READY
TRACK_HEIGHT = 0.100               # EE above cube while catching up
GRASP_Z_BIAS = -0.008              # final TCP Z offset at grasp (tune +/- few mm)

# ---- Catch-up / capture (the "faster than the cube" behaviour) --------
TAU_CATCHUP  = 0.30                # s  aim this far AHEAD of the cube -> EE outruns it
TAU_CAPTURE  = 0.03                # s  near grasp: aim just ahead -> match cube speed
LEAD_X       = 0.030              # m  extra fixed lead during catch-up

# ---- Pick-zone geometry, RELATIVE to robot-base X (m) -----------------
TRACK_START_OFFSET   = -0.270      # begin catching up when cube reaches here
CAPTURE_START_OFFSET = -0.090      # allowed to start the final close after here
DEADLINE_OFFSET      = +0.090      # give up (do NOT chase downstream) past here

# ---- Capture gate: must be this accurate for N frames before closing --
CAP_XY_TOL      = 0.006            # m
CAP_Z_TOL       = 0.006            # m
CAP_RELV_TOL    = 0.020            # m/s  relative EE-cube speed at capture
REQUIRED_GOOD_FRAMES = 5

# ---- Closing ----------------------------------------------------------
CLOSE_FRAMES   = 18                # fast close (cube keeps moving during it)
POST_CLOSE_HOLD = 12

# ---- Quick lift + success gate ---------------------------------------
QUICK_LIFT_HEIGHT = 0.080          # m  immediate lift off the belt
QUICK_LIFT_FRAMES = 45
CUBE_RISE_GATE    = 0.015          # m  minimum rise to accept the grasp
FINAL_RISE_GATE   = 0.040          # m  reported "clean pick" threshold

# ---- Transfer to pedestal --------------------------------------------
CARRY_HEIGHT    = 0.230            # m  clearance height above cube start Z
PLACE_CLEARANCE = 0.004            # m  gap left under the cube when placing
TRANSFER_FRAMES = 150
PLACE_FRAMES    = 90
RETREAT_FRAMES  = 70

# ---- Motion timing ----------------------------------------------------
READY_MOVE_FRAMES = 120
IK_POS_TOL   = 0.002
IK_ORI_TOL   = 0.10                # rad (loose -> keeps solutions findable)
MAX_IK_FAILS = 15
MIN_UPSTREAM = 0.30                # cube must start at least this far upstream


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

def rot_to_quat(R):
    """3x3 rotation matrix -> (w, x, y, z) quaternion.
    Lula compute_forward_kinematics returns a matrix, but
    compute_inverse_kinematics expects a wxyz quaternion."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)

def gripper_targets(q7):
    """Master + mirrored followers: q7 -> [q7, -q7, -q7]."""
    return np.array([+q7, -q7, -q7], dtype=np.float32)


# ======================================================================
# 3. GRIPPER DRIVE / MIMIC CONFIGURATION  (run while sim is STOPPED)
# ======================================================================

def find_finger_joint_prims(stage):
    """Locate joint_7/8/9 prims anywhere under the robot base."""
    found = {}
    root = stage.GetPrimAtPath(BASE_PATH)
    if not root.IsValid():
        return found
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name in FINGER_JOINT_NAMES and prim.IsA(UsdPhysics.PrismaticJoint):
            found[name] = prim
    # fallback: match by name only if type check missed
    if len(found) < 3:
        for prim in Usd.PrimRange(root):
            n = prim.GetName()
            if n in FINGER_JOINT_NAMES and n not in found:
                found[n] = prim
    return found


def try_remove_mimic(prim):
    """Best-effort removal of PhysxMimicJointAPI so drives fully own the
    finger joints. Returns list of (schema, removed?)."""
    out = []
    try:
        for s in list(prim.GetAppliedSchemas()):
            if "MimicJoint" not in s:
                continue
            inst = s.split(":", 1)[1] if ":" in s else ""
            ok = False
            try:
                if inst:
                    ok = bool(prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, inst))
                else:
                    ok = bool(prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI))
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


def set_linear_drive(prim, stiffness, damping, max_force):
    """Force a stiff position drive on a prismatic joint."""
    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))


def configure_gripper(stage):
    """Reconcile mimic + drives so all three fingers move as one stiff,
    symmetric unit. Must be called while the timeline is STOPPED so the
    changes are baked when physics starts."""
    print("\n[GRIPPER] configuring finger drives (sim stopped)...")
    prims = find_finger_joint_prims(stage)
    if len(prims) < 3:
        print("[GRIPPER]   WARNING: found only", list(prims.keys()),
              "- check FINGER_JOINT_NAMES / paths.")
    for name in FINGER_JOINT_NAMES:
        prim = prims.get(name)
        if prim is None:
            continue
        removed = try_remove_mimic(prim)
        if removed:
            print(f"[GRIPPER]   {name}: mimic ->", removed)
        try:
            set_linear_drive(prim, FINGER_DRIVE_STIFFNESS,
                             FINGER_DRIVE_DAMPING, FINGER_MAX_FORCE)
            print(f"[GRIPPER]   {name}: drive set "
                  f"(k={FINGER_DRIVE_STIFFNESS:.0f}, "
                  f"d={FINGER_DRIVE_DAMPING:.0f}, "
                  f"Fmax={FINGER_MAX_FORCE:.0f})")
        except Exception as exc:
            print(f"[GRIPPER]   {name}: drive set FAILED:", exc)


# ======================================================================
# 4. GEOMETRY HELPERS
# ======================================================================

def world_bbox(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError("Invalid prim: " + prim_path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    dims = np.array([hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]], dtype=np.float64)
    center = np.array([(hi[0]+lo[0])/2, (hi[1]+lo[1])/2, (hi[2]+lo[2])/2],
                      dtype=np.float64)
    top_z = float(hi[2])
    return dims, center, top_z


# ======================================================================
# 5. MAIN
# ======================================================================

async def moving_pick_place():
    print("\n" + "=" * 70)
    print("LYNXMOTION — ON-THE-FLY MOVING PICK & PLACE")
    print("=" * 70)

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()

    # ---- (a) STOP -> configure gripper -> PLAY -------------------------
    if not timeline.is_stopped():
        print("\n[1] stopping timeline to configure gripper...")
        timeline.stop()
        await wait_frames(5)

    configure_gripper(stage)

    print("\n[1] starting simulation...")
    timeline.play()
    await wait_frames(30)

    # ---- (b) init robot + prims ---------------------------------------
    robot = SingleArticulation(prim_path=ROBOT_PATH, name="lynx_mpp")
    robot.initialize()
    base_prim = SingleXFormPrim(prim_path=BASE_PATH, name="lynx_base")
    ee_prim   = SingleXFormPrim(prim_path=EE_PATH,   name="lynx_ee")
    cube_prim = SingleXFormPrim(prim_path=CUBE_PATH, name="lynx_cube")

    dof_names = list(robot.dof_names)
    print("\n[2] DOF order:", dof_names)

    base_pos, base_quat = base_prim.get_world_pose()
    base_pos, base_quat = as64(base_pos), as64(base_quat)
    cube_start, _ = cube_prim.get_world_pose()
    cube_start = as64(cube_start)
    cube_dims, _, _ = world_bbox(CUBE_PATH)
    cube_z0 = float(cube_start[2])
    lane_y  = float(cube_start[1])

    print("    base   :", np.round(base_pos, 4))
    print("    cube   :", np.round(cube_start, 4),
          " dims(mm):", np.round(cube_dims * 1000, 1))

    # pedestal top (live)
    _, ped_center, ped_top_z = world_bbox(PEDESTAL_PATH)
    place_pos = np.array([ped_center[0], ped_center[1],
                          ped_top_z + cube_dims[2] / 2.0 + PLACE_CLEARANCE],
                         dtype=np.float64)
    print("    pedestal top Z:", round(ped_top_z, 4),
          " -> place target:", np.round(place_pos, 4))

    # pick-zone X coordinates
    pick_x   = float(base_pos[0])
    track_x  = pick_x + TRACK_START_OFFSET
    capture_x = pick_x + CAPTURE_START_OFFSET
    dead_x   = pick_x + DEADLINE_OFFSET
    print("    pick center X:", round(pick_x, 4),
          " track@", round(track_x, 4),
          " capture@", round(capture_x, 4),
          " deadline@", round(dead_x, 4))

    upstream = pick_x - float(cube_start[0])
    print("    cube upstream distance:", round(upstream, 4), "m")
    if upstream < MIN_UPSTREAM:
        print("\n[ABORT] cube not far enough upstream. Stop the sim so the "
              "cube resets upstream, then re-run.")
        return

    # ---- (c) Lula solver ----------------------------------------------
    solver = LulaKinematicsSolver(robot_description_path=ROBOT_DESCRIPTION,
                                  urdf_path=URDF)
    solver.set_robot_base_pose(base_pos, base_quat)
    lower, upper = solver.get_cspace_position_limits()
    lower, upper = as64(lower), as64(upper)

    def command_arm(q):
        robot.apply_action(ArticulationAction(
            joint_positions=np.asarray(q, dtype=np.float32),
            joint_indices=ARM_INDICES))

    def command_gripper(q7):
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets(q7),
            joint_indices=GRIPPER_INDICES))

    def read_gripper():
        q = as64(robot.get_joint_positions())[GRIPPER_INDICES]
        return q  # [q7, q8, q9]

    def solve(pos, R, warm):
        if R is None:
            q, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME,
                target_position=as64(pos),
                target_orientation=None,
                warm_start=as64(warm),
                position_tolerance=IK_POS_TOL)
        else:
            q, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME,
                target_position=as64(pos),
                target_orientation=as64(R),
                warm_start=as64(warm),
                position_tolerance=IK_POS_TOL,
                orientation_tolerance=IK_ORI_TOL)
        return as64(q), bool(ok)

    # ---- (d) GRIPPER SELF-TEST (Gate 1) --------------------------------
    print("\n[3] GRIPPER SELF-TEST")
    for _ in range(25):
        command_gripper(GRIPPER_OPEN_Q7)
        await next_frame()
    q_open = read_gripper()
    for _ in range(25):
        command_gripper(GRIPPER_CLOSED_Q7)
        await next_frame()
    q_closed = read_gripper()
    for _ in range(20):
        command_gripper(GRIPPER_OPEN_Q7)
        await next_frame()
    q_reopen = read_gripper()

    print("    OPEN   q7,q8,q9 [mm]:", np.round(q_open * 1000, 2))
    print("    CLOSED q7,q8,q9 [mm]:", np.round(q_closed * 1000, 2))
    print("    REOPEN q7,q8,q9 [mm]:", np.round(q_reopen * 1000, 2))

    def symmetric(q, tol=0.0015):
        return abs(q[1] + q[0]) < tol and abs(q[2] + q[0]) < tol

    open_ok  = symmetric(q_open)  and q_open[0]  > 0.006
    close_ok = symmetric(q_closed) and q_closed[0] < 0.004
    if not (open_ok and close_ok):
        print("\n[ABORT] gripper did NOT open/close symmetrically.")
        print("        The mimic constraint is probably still fighting the")
        print("        drives. Fix once in the USD, then re-run:")
        print("          * select joint_8 and joint_9,")
        print("          * remove 'Physx Mimic Joint' API from each,")
        print("          * save, stop, and run this script again.")
        print("        (This script tries to remove it automatically, but")
        print("         some scenes lock it on a stronger session layer.)")
        return
    print("    --> gripper OK (symmetric open & close)")

    # ---- (e) fixed grasp orientation (anti-snap) ----------------------
    ready_q = np.deg2rad(READY_Q_DEG)
    fk_pos, R_grasp_mat = solver.compute_forward_kinematics(EE_FRAME, ready_q)
    R_grasp = rot_to_quat(R_grasp_mat)      # Lula IK expects a wxyz quaternion
    print("\n[4] fixed grasp orientation locked from READY branch")
    print("    grasp quat (wxyz):", np.round(R_grasp, 4))

    # verify the fixed orientation is usable at the pick point
    pick_probe = np.array([pick_x, lane_y, cube_z0 + GRASP_Z_BIAS])
    _, ok = solve(pick_probe, R_grasp, ready_q)
    if not ok:
        print("    fixed-orientation IK failed at pick point; will fall back")
        print("    to position-only tracking with a joint-continuity filter.")
        R_track = None
    else:
        R_track = R_grasp

    # ---- (f) open + move to READY -------------------------------------
    print("\n[5] opening gripper + moving to READY (smooth)")
    q_now = as64(robot.get_joint_positions())[:6]
    for f in range(READY_MOVE_FRAMES):
        a = smoothstep((f + 1) / READY_MOVE_FRAMES)
        command_arm(q_now + a * (ready_q - q_now))
        command_gripper(GRIPPER_OPEN_Q7)
        await next_frame()

    cube_p, _ = cube_prim.get_world_pose()
    if float(as64(cube_p)[0]) >= capture_x:
        print("\n[ABORT] cube already reached the capture zone before READY "
              "finished. Stop the sim (cube resets) and re-run.")
        return

    # ---- (g) WAIT for cube to reach track zone ------------------------
    print("\n[6] waiting for cube to reach track zone...")
    q_seed = ready_q.copy()
    while True:
        cube_p, _ = cube_prim.get_world_pose()
        command_arm(q_seed)
        command_gripper(GRIPPER_OPEN_Q7)
        if float(as64(cube_p)[0]) >= track_x:
            break
        await next_frame()

    # ---- (h0) ENGAGE: smoothly move from READY to a first tracking pose
    #          so the first tracking step is never rejected as a jump ----
    cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
    engage_tgt = np.array([cube_p[0] + LEAD_X, cube_p[1],
                           cube_p[2] + TRACK_HEIGHT])
    q_engage, ok = solve(engage_tgt, R_track, ready_q)
    if ok:
        await interp(command_arm, command_gripper, ready_q, q_engage,
                     25, GRIPPER_OPEN_Q7)
        q_seed = q_engage

    # ---- (h) CATCH-UP -> ALIGN -> CLOSE gate --------------------------
    print("\n[7] CATCH-UP + TRACK  (EE runs faster, then matches cube speed)")
    cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
    prev_cube = as64(cube_p)
    prev_ee, _ = ee_prim.get_world_pose()
    prev_ee = as64(prev_ee)
    prev_t = time.monotonic()
    v_cube = np.zeros(3)
    v_ee = np.zeros(3)
    ik_fails = 0
    good = 0
    frame = 0

    def bad_jump(qa, qb, lim=0.35):
        return float(np.max(np.abs(np.asarray(qa) - np.asarray(qb)))) > lim

    while True:
        cube_p, _ = cube_prim.get_world_pose()
        cube_p = as64(cube_p)
        cube_x = float(cube_p[0])

        now = time.monotonic()
        dt = max(now - prev_t, 1e-3)
        v_cube = 0.7 * v_cube + 0.3 * ((cube_p - prev_cube) / dt)
        prev_cube, prev_t = cube_p.copy(), now

        # phase: catch up (aim ahead -> EE outruns cube) until the cube is
        # in the capture zone, then shrink the lead so EE matches cube speed
        tau = TAU_CAPTURE if cube_x >= capture_x else TAU_CATCHUP
        lead = 0.0 if cube_x >= capture_x else LEAD_X

        # height schedule: descend from TRACK_HEIGHT toward grasp bias
        if cube_x <= track_x:
            hz = TRACK_HEIGHT
        else:
            prog = np.clip((cube_x - track_x) / (capture_x - track_x), 0, 1)
            hz = TRACK_HEIGHT + smoothstep(prog) * (GRASP_Z_BIAS - TRACK_HEIGHT)

        target = cube_p.copy()
        target[0] = cube_p[0] + v_cube[0] * tau + lead
        target[1] = cube_p[1] + v_cube[1] * tau
        target[2] = cube_p[2] + hz

        q_new, ok = solve(target, R_track, q_seed)
        if ok and not bad_jump(q_new, q_seed):
            q_seed = q_new
            ik_fails = 0
        else:
            ik_fails += 1
        command_arm(q_seed)
        command_gripper(GRIPPER_OPEN_Q7)
        await next_frame()

        ee_p, _ = ee_prim.get_world_pose()
        ee_p = as64(ee_p)
        v_ee = 0.7 * v_ee + 0.3 * ((ee_p - prev_ee) / dt)
        prev_ee = ee_p.copy()

        grasp_pt = cube_p.copy()
        grasp_pt[2] = cube_p[2] + GRASP_Z_BIAS
        err = ee_p - grasp_pt
        exy = float(np.linalg.norm(err[:2]))
        ez = float(err[2])
        relv = float(np.linalg.norm(v_ee - v_cube))

        if frame % 8 == 0:
            print(f"    x={cube_x:.3f} h={hz*1000:+5.1f}mm "
                  f"exy={exy*1000:5.1f} ez={ez*1000:+5.1f}mm "
                  f"vcx={v_cube[0]:+.3f} vex={v_ee[0]:+.3f} "
                  f"relv={relv*1000:5.1f}mm/s good={good}")

        aligned = (exy < CAP_XY_TOL and abs(ez) < CAP_Z_TOL
                   and relv < CAP_RELV_TOL)
        good = good + 1 if (cube_x >= capture_x and aligned) else 0

        if good >= REQUIRED_GOOD_FRAMES:
            print("\n    >>> CAPTURE ALIGNED: exy=%.1fmm ez=%.1fmm relv=%.1fmm/s"
                  % (exy*1000, ez*1000, relv*1000))
            break
        if ik_fails > MAX_IK_FAILS:
            print("\n[ABORT] repeated IK failure during tracking.")
            return
        if cube_x >= dead_x:
            print("\n[ABORT] cube passed the grasp deadline without alignment "
                  "(exy=%.1fmm, relv=%.1fmm/s). Not chasing downstream."
                  % (exy*1000, relv*1000))
            return
        frame += 1

    # ---- (i) CLOSE while tracking (velocity-matched) ------------------
    print("\n[8] CLOSING (still tracking cube)")
    for f in range(CLOSE_FRAMES):
        cube_p, _ = cube_prim.get_world_pose()
        cube_p = as64(cube_p)
        now = time.monotonic(); dt = max(now - prev_t, 1e-3)
        v_cube = 0.7 * v_cube + 0.3 * ((cube_p - prev_cube) / dt)
        prev_cube, prev_t = cube_p.copy(), now

        target = cube_p.copy()
        target[0] += v_cube[0] * TAU_CAPTURE
        target[1] += v_cube[1] * TAU_CAPTURE
        target[2] += GRASP_Z_BIAS
        q_new, ok = solve(target, R_track, q_seed)
        if ok and not bad_jump(q_new, q_seed):
            q_seed = q_new
        command_arm(q_seed)
        a = smoothstep((f + 1) / CLOSE_FRAMES)
        command_gripper(GRIPPER_OPEN_Q7 + a * (GRIPPER_CLOSED_Q7 - GRIPPER_OPEN_Q7))
        await next_frame()

    for _ in range(POST_CLOSE_HOLD):
        cube_p, _ = cube_prim.get_world_pose()
        cube_p = as64(cube_p)
        target = cube_p.copy(); target[2] += GRASP_Z_BIAS
        q_new, ok = solve(target, R_track, q_seed)
        if ok and not bad_jump(q_new, q_seed):
            q_seed = q_new
        command_arm(q_seed)
        command_gripper(GRIPPER_CLOSED_Q7)
        await next_frame()

    qg = read_gripper()
    print("    gripper after close q7,q8,q9 [mm]:", np.round(qg * 1000, 2))

    # ---- (j) QUICK LIFT + verify the cube actually rose ---------------
    print("\n[9] QUICK LIFT")
    ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
    lift_start = ee_p.copy()
    lift_end = lift_start + np.array([0, 0, QUICK_LIFT_HEIGHT])
    for f in range(QUICK_LIFT_FRAMES):
        a = smoothstep((f + 1) / QUICK_LIFT_FRAMES)
        tgt = lift_start + a * (lift_end - lift_start)
        q_new, ok = solve(tgt, R_track, q_seed)
        if ok and not bad_jump(q_new, q_seed):
            q_seed = q_new
        command_arm(q_seed)
        command_gripper(GRIPPER_CLOSED_Q7)
        await next_frame()

    cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
    rise = float(cube_p[2] - cube_z0)
    print("    cube rise after quick lift: %.1f mm" % (rise * 1000))

    if rise < CUBE_RISE_GATE:
        print("\n[STOP] cube did NOT rise (%.1f mm < %.1f mm gate)."
              % (rise * 1000, CUBE_RISE_GATE * 1000))
        print("       Grasp failed — NOT transferring. Opening + returning "
              "to READY.")
        for _ in range(20):
            command_gripper(GRIPPER_OPEN_Q7); command_arm(q_seed); await next_frame()
        q0 = as64(robot.get_joint_positions())[:6]
        for f in range(READY_MOVE_FRAMES):
            a = smoothstep((f + 1) / READY_MOVE_FRAMES)
            command_arm(q0 + a * (ready_q - q0))
            command_gripper(GRIPPER_OPEN_Q7)
            await next_frame()
        print("\n[NEXT] Most likely fixes, in order:")
        print("   1. sweep GRASP_Z_BIAS by a few mm (fingers slightly low/high)")
        print("   2. if the close bottoms out with no squeeze, resize cube 20->25mm")
        print("   3. raise FINGER_MAX_FORCE (grip too weak) ")
        return

    print("    grasp CONFIRMED (cube is being carried).")

    # ---- (k) TRANSFER up-and-over to pedestal (joint-space) -----------
    print("\n[10] TRANSFER to pedestal (solved once, joint-space interp)")
    carry_z = cube_z0 + CARRY_HEIGHT

    def solve_key(pos, warm, label):
        q, ok = solve(pos, R_grasp, warm)          # try fixed orientation
        if not ok or bad_jump(q, warm, 1.2):
            q, ok = solve(pos, None, warm)          # fall back position-only
        if not ok:
            print("    [WARN] IK failed for %s at %s" % (label, np.round(pos, 3)))
            return warm, False
        return q, True

    ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
    carry_over_cube = np.array([ee_p[0], ee_p[1], carry_z])
    carry_over_ped  = np.array([place_pos[0], place_pos[1], carry_z])

    q_up,   _ = solve_key(carry_over_cube, q_seed, "lift-high")
    q_over, _ = solve_key(carry_over_ped,  q_up,   "over-pedestal")
    q_place, _ = solve_key(place_pos, q_over, "place")

    # segment 1: straight up over the belt
    await interp(command_arm, command_gripper, q_seed, q_up,
                 TRANSFER_FRAMES // 2, GRIPPER_CLOSED_Q7)
    # segment 2: swing across to above the pedestal (well clear of belt)
    await interp(command_arm, command_gripper, q_up, q_over,
                 TRANSFER_FRAMES, GRIPPER_CLOSED_Q7)
    # segment 3: lower onto pedestal
    await interp(command_arm, command_gripper, q_over, q_place,
                 PLACE_FRAMES, GRIPPER_CLOSED_Q7)

    # ---- (l) RELEASE + RETREAT ----------------------------------------
    print("\n[11] RELEASE + RETREAT")
    for _ in range(25):
        command_arm(q_place); command_gripper(GRIPPER_OPEN_Q7); await next_frame()
    await interp(command_arm, command_gripper, q_place, q_over,
                 RETREAT_FRAMES, GRIPPER_OPEN_Q7)
    q_end = as64(robot.get_joint_positions())[:6]
    await interp(command_arm, command_gripper, q_end, ready_q,
                 READY_MOVE_FRAMES, GRIPPER_OPEN_Q7)

    # ---- (m) report ----------------------------------------------------
    cube_f, _ = cube_prim.get_world_pose(); cube_f = as64(cube_f)
    place_err = float(np.linalg.norm(cube_f[:2] - place_pos[:2]))
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print("  cube final     :", np.round(cube_f, 4))
    print("  pedestal target:", np.round(place_pos, 4))
    place_z_err = float(abs(cube_f[2] - place_pos[2]))
    print("  place XY error : %.1f mm" % (place_err * 1000))
    print("  place Z  error : %.1f mm" % (place_z_err * 1000))
    print("  quick-lift rise: %.1f mm (gate %.0f mm)"
          % (rise * 1000, FINAL_RISE_GATE * 1000))
    if place_err < 0.03 and place_z_err < 0.03:
        print("\n  OVERALL: PICK & PLACE SUCCESS")
    else:
        print("\n  OVERALL: placed, but check pedestal error above")
    print("=" * 70)


async def interp(cmd_arm, cmd_grip, q0, q1, frames, grip_q7):
    """Smooth joint-space interpolation between two solved configs."""
    q0 = as64(q0); q1 = as64(q1)
    for f in range(frames):
        a = smoothstep((f + 1) / frames)
        cmd_arm(q0 + a * (q1 - q0))
        cmd_grip(grip_q7)
        await next_frame()


# ======================================================================
# 6. LAUNCH  (Ctrl+Enter; surfaces async exceptions with traceback)
# ======================================================================

def _done(task):
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        print("[LYNX] task cancelled")
        return
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
