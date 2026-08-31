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

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdShade, Sdf

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
WRIST_PATH = "/World/Lynxmotion/gripper"        # gripper body / wrist (collision monitor)
ORIG_CUBE_PATH = "/World/Cube"                  # original scene cube
PEDESTAL_PATH = "/World/Pedestal"
CONVEYOR_PATH = "/World/ConveyorBelt_A09"       # read live to find the rail tops
EE_FRAME   = "pro_arm_ee"

# ---- Run mode + grasp object -----------------------------------------
#   "single" : the moving pick-and-place (one run, watch it)
#   "sweep"  : a STATIC grasp parameter sweep (object size x friction x grasp
#              offset x force) to map what actually grips -- see grasp_sweep()
RUN_MODE = "single"

#   A 3-jaw gripper self-centers on a ROUND object, so we grasp a cylinder
#   instead of a cube (a cube gets ejected on its corners). The script builds
#   the cylinder itself and hides the original cube.
OBJECT_SHAPE    = "cylinder"       # "cylinder" or "cube" (cube = use the scene cube)
OBJECT_PATH     = "/World/GraspObject"
#   A short WIDE puck (height < diameter) so it does NOT topple/roll when the
#   conveyor yanks it.  Round -> self-centers in the 3 fingers and (having no
#   corners) fits the open jaws up to ~36 mm dia.
OBJECT_DIAMETER = 0.030            # m
OBJECT_HEIGHT   = 0.022            # m  (aspect < 1 -> stable on the belt)
OBJECT_MASS     = 0.05             # kg
CUBE_PATH = OBJECT_PATH if OBJECT_SHAPE == "cylinder" else ORIG_CUBE_PATH

# ---- Parameter sweep (RUN_MODE = "sweep") ----------------------------
#   A STATIC grasp test on the pedestal (no belt) that grasps a cylinder and
#   measures how much it lifts, across the grid below.  Fast + isolates the
#   grasp mechanics from the moving tracking.  Results printed + saved to CSV.
SWEEP_DIAMETERS = [0.018, 0.022, 0.026, 0.030]   # m  cylinder diameter
SWEEP_FRICTIONS = [0.5, 1.0, 1.5]                # contact friction
SWEEP_ZOFFSETS  = [-0.005, 0.000, 0.005]         # m  grasp height vs object center
SWEEP_FORCES    = [20.0, 40.0]                    # N  finger force cap
SWEEP_OBJ_HEIGHT = 0.028                          # m  cylinder height (kept low so it doesn't tip)
SWEEP_CSV = CFG_ROOT + "/grasp_sweep_results.csv"
SWEEP_RISE_OK = 0.015                             # m  rise counted as a success

# ---- Articulation layout ---------------------------------------------
ARM_INDICES     = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
GRIPPER_INDICES = np.array([6, 7, 8], dtype=np.int32)   # joint_7,8,9
FINGER_JOINT_NAMES = ["joint_7", "joint_8", "joint_9"]  # 7 = master

# ---- Known-good READY branch (deg) -----------------------------------
READY_Q_DEG = [-75.455, -36.334, 11.833, 3.131, 84.152, -97.696]

# ---- Gripper aperture (joint_7 stroke 0 .. +0.010 m) -----------------
#   VERIFIED from the URDF + live calibration that the direction is the
#   OPPOSITE of what earlier notes assumed:
#     q7 = 0.000 -> fingertips ~37 mm from center = WIDE  (OPEN)
#     q7 = 0.010 -> fingertips ~27 mm from center = NARROW (CLOSED / clamps)
#   With it backwards, "open" narrowed the jaws (blocking the descent) and
#   "close" spread them (never gripping).  This is the corrected mapping.
GRIPPER_OPEN_Q7   = 0.000          # wide  -> clears the cube on the way down
GRIPPER_CLOSED_Q7 = 0.010          # narrow -> clamps the cube

# ---- Cube size --------------------------------------------------------
#   This CGE-10-10 is a LARGE 3-jaw gripper: even fully closed the fingertips
#   sit ~37 mm from the grasp center (measured from the URDF + Lula collision
#   spheres).  A 20-25 mm cube is far too small -- the fingers close through
#   empty air and never clamp it (every "no rise" run).  The gripper needs an
#   IMPORTANT (measured from the finger STL meshes): for a TOP-DOWN grasp the
#   open fingers only clear an ~18 mm-radius opening, so the cube's corners
#   must stay inside that -> cube must be <= ~24 mm to fit between the fingers
#   on the way down.  Bigger cubes (the 45 mm) simply ram the fingers.  The
#   gripper's 20-60 mm rating is for objects presented differently, not for
#   top-down insertion around a cube on a belt.  ~20 mm gives a solid grip.
CUBE_TARGET_SIZE = 0.020           # m  (None = don't touch the cube)

# ---- Gripper finger geometry (for calibration) -----------------------
#   Fingertip offsets in each finger's LOCAL frame, from the Lula collision
#   spheres, used to measure the real aperture live at startup.
FINGER_PRIMS = {
    "finger_1": "/World/Lynxmotion/finger_1",
    "finger_2": "/World/Lynxmotion/finger_2",
    "finger_3": "/World/Lynxmotion/finger_3",
}
FINGER_TIP_LOCAL = {   # real STL tips (Lula spheres were ~20 mm short of these)
    "finger_1": np.array([-0.0181, -0.0170, -0.0446]),
    "finger_2": np.array([ 0.0300, -0.0170,  0.0085]),
    "finger_3": np.array([-0.0180, -0.0170,  0.0447]),
}

# ---- Finger drive gains applied at startup (the real grasp fix) -------
FINGER_DRIVE_STIFFNESS = 5.0e4     # N/m
FINGER_DRIVE_DAMPING   = 8.0e2     # N/(m/s)
FINGER_MAX_FORCE       = 28.0      # N  (grip force cap; enough for the puck, less bounce)

# ---- Contact friction (the main fix for a slipping grasp) -------------
#   The default ~0.3 friction lets the small cube slip out of the 3-finger
#   corner grip.  Boost the cube + finger contact friction at startup.
FRICTION_BOOST   = True
GRIP_FRICTION    = 1.3             # static = dynamic friction for cube & fingers

# ---- Heights (m) ------------------------------------------------------
READY_HEIGHT = 0.220               # EE above cube at READY
HOVER_HEIGHT = 0.060               # EE hovers this far above cube at the intercept

# The gripper's GRASP CENTER (where the 3 fingers converge) sits this far
# from the pro_arm_ee frame, TOWARD the gripper along the approach axis
# (measured from the URDF: ~19.5 mm; re-measured live during calibration).
# We place the grasp center at the cube center, so the EE goes ~19.5 mm the
# other side of the cube center.  This makes the grasp height correct for any
# cube size and any (top-down/tilted) orientation automatically.
GRASP_CENTER_FROM_EE   = 0.005     # m  ee -> grasp-center (real STL tips ~= pro_arm_ee)
GRASP_CENTER_ABOVE_CUBE = 0.000    # m  put grasp center this far above cube center
FINGERTIP_BELT_CLEAR   = 0.004     # m  keep the fingertips above the belt
DESCEND_GATE_XY        = 0.010     # don't drop to grasp level until aligned within this

# ---- Collision avoidance (conveyor side rails) ------------------------
#   The robot is on the near side of the belt; the cube lane is only ~165 mm
#   past the near rail.  A tilted gripper leans its wrist INTO that rail, so
#   we grasp TOP-DOWN (wrist directly above the cube) and keep all travel
#   above the rail tops, which are read live from the conveyor bbox.
RAIL_CLEAR_MARGIN = 0.050          # m  travel this far above the rail tops
COLLISION_MONITOR = True           # warn when the wrist enters the rail zone
TOPDOWN_YAW_STEP  = 15             # deg  granularity of the top-down yaw search

# ---- Intercept / capture ---------------------------------------------
#   The arm hovers over the lane at the robot's own X (best reach) and
#   grabs the cube as it arrives, matching the cube's velocity.  We do NOT
#   chase far upstream -- that is what over-extended the arm before.
TAU_CAPTURE    = 0.05              # s  aim this far ahead of the cube (velocity match)
LEAD_X         = 0.020             # m  small +X lead early in approach, decays to 0
MAX_JOINT_STEP = 0.035             # rad/frame command cap (smaller = smoother, less jitter)
CMD_SMOOTH     = 0.55              # arm-command low-pass (0=none, higher=smoother/slower)
VEL_SMOOTH     = 0.85             # cube-velocity EMA weight (higher = less noisy feed-forward)

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
SETTLE_OPEN_FRAMES = 12            # hold OPEN around the centered cube before closing
CLOSE_FRAMES    = 18
POST_CLOSE_HOLD = 25               # hold the close longer so the grip settles

# ---- Quick lift + success gate ---------------------------------------
QUICK_LIFT_HEIGHT = 0.080
QUICK_LIFT_FRAMES = 90             # slower lift so inertia doesn't break the grip
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

def quat_to_R(q):
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], dtype=np.float64)

def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def grasp_quat(yaw_deg, tilt_deg=0.0):
    """EE orientation whose approach axis (the EE-frame -Z, per the URDF)
    points DOWN, spun by `yaw_deg` about vertical and leaned by `tilt_deg`
    along the belt's X axis.  tilt=0 is pure top-down (wrist directly above
    the cube).  A small tilt along X keeps the wrist over the lane (does NOT
    move it toward the near rail) while relaxing the IK if pure top-down is
    just out of reach."""
    R = _Ry(np.radians(tilt_deg)) @ _Rz(np.radians(yaw_deg))
    return rot_to_quat(R)

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


def stabilize_physics(stage):
    """TGS solver + stabilization on the physics scene -> steadier contacts,
    less bounce/jitter. Best-effort."""
    try:
        sc = stage.GetPrimAtPath("/World/PhysicsScene")
        if not sc.IsValid():
            return
        api = PhysxSchema.PhysxSceneAPI.Apply(sc)
        api.CreateSolverTypeAttr().Set("TGS")
        api.CreateEnableStabilizationAttr().Set(True)
        print("[PHYSICS] TGS solver + stabilization enabled")
    except Exception as exc:
        print("[PHYSICS] scene tune skipped (%s)" % exc)


def boost_friction(stage, mu):
    """Bind a high-friction physics material to the cube + fingers so the
    grasp doesn't slip (run while STOPPED). Best-effort."""
    try:
        mat_path = "/World/GripHighFriction"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr().Set(float(mu))
            api.CreateDynamicFrictionAttr().Set(float(mu))
            api.CreateRestitutionAttr().Set(0.0)
        mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        targets = [CUBE_PATH, "/World/Lynxmotion/finger_1",
                   "/World/Lynxmotion/finger_2", "/World/Lynxmotion/finger_3"]
        done = []
        for p in targets:
            prim = stage.GetPrimAtPath(p)
            if not prim.IsValid():
                continue
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(
                mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics")
            done.append(prim.GetName())
        print("[FRICTION] mu=%.1f bound to: %s" % (mu, done))
    except Exception as exc:
        print("[FRICTION] failed (%s) — grasp may still slip; can raise "
              "friction in the USD instead." % exc)


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


def make_cylinder(stage, path, diameter, height, pos, mass):
    """Create (or replace) an upright cylinder rigid body at `pos` (center),
    run while STOPPED.  Returns True on success."""
    r = diameter / 2.0
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(r))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([(-r, -r, -height/2.0), (r, r, height/2.0)])
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    prim = cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    # stability: damping + extra solver iterations so it doesn't bounce/jitter
    try:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr().Set(0.5)
        rb.CreateAngularDampingAttr().Set(0.5)
        rb.CreateSolverPositionIterationCountAttr().Set(32)
        rb.CreateSolverVelocityIterationCountAttr().Set(4)
        rb.CreateSleepThresholdAttr().Set(0.0)
    except Exception:
        pass
    return True


def prepare_object(stage):
    """Build the grasp object (cylinder) on the belt and hide the scene cube.
    Returns the object's belt-top Z so callers can place things.  No-op for
    OBJECT_SHAPE == 'cube'."""
    if OBJECT_SHAPE != "cylinder":
        return
    try:
        dims, center, _ = world_bbox(ORIG_CUBE_PATH)
        belt = float(center[2] - dims[2] / 2.0)
        cx, cy = float(center[0]), float(center[1])
    except Exception:
        cx, cy, belt = 0.1815, 3.947, 1.7843
        print("[OBJECT] could not read scene cube; using default belt pose")
    orig = stage.GetPrimAtPath(ORIG_CUBE_PATH)
    if orig.IsValid():
        orig.SetActive(False)                       # remove the cube from the sim
    try:
        make_cylinder(stage, OBJECT_PATH, OBJECT_DIAMETER, OBJECT_HEIGHT,
                      (cx, cy, belt + OBJECT_HEIGHT/2.0), OBJECT_MASS)
        print("[OBJECT] cylinder d=%.0f mm h=%.0f mm at [%.3f, %.3f, %.4f]"
              % (OBJECT_DIAMETER*1000, OBJECT_HEIGHT*1000, cx, cy, belt+OBJECT_HEIGHT/2.0))
    except Exception as exc:
        print("[OBJECT] cylinder build FAILED (%s). Re-activate the cube or add a "
              "cylinder named %s manually." % (exc, OBJECT_PATH))


def resize_cube(stage, target_m):
    """Scale /World/Cube to ~target_m per side (run while STOPPED).  Best
    effort + self-verifying: the main loop re-measures and prints the size."""
    try:
        prim = stage.GetPrimAtPath(CUBE_PATH)
        if not prim.IsValid():
            print("[CUBE] not found — skip resize"); return
        dims, _, _ = world_bbox(CUBE_PATH)
        cur = float(np.mean(dims))
        if cur <= 1e-6:
            print("[CUBE] bad dims — skip resize"); return
        if abs(cur - target_m) < 5e-4:
            print("[CUBE] already ~%.1f mm" % (cur*1000)); return
        ratio = target_m / cur
        xf = UsdGeom.Xformable(prim)
        scale_op = trans_op = None
        for op in xf.GetOrderedXformOps():
            t = op.GetOpType()
            if t == UsdGeom.XformOp.TypeScale and scale_op is None: scale_op = op
            if t == UsdGeom.XformOp.TypeTranslate and trans_op is None: trans_op = op
        base = (scale_op.Get() if scale_op is not None else None) or Gf.Vec3f(1, 1, 1)
        if scale_op is None:
            scale_op = xf.AddScaleOp()
        scale_op.Set(Gf.Vec3f(float(base[0])*ratio, float(base[1])*ratio,
                              float(base[2])*ratio))
        # lift so the enlarged cube doesn't start penetrating the belt
        grow_half = (target_m - cur) / 2.0
        if trans_op is not None:
            tv = trans_op.Get()
            if tv is not None:
                trans_op.Set(type(tv)(tv[0], tv[1], tv[2] + grow_half))
        print("[CUBE] resized ~%.1f -> ~%.1f mm (scale x%.3f)"
              % (cur*1000, target_m*1000, ratio))
    except Exception as exc:
        print("[CUBE] resize failed (%s) — resize /World/Cube to %.0f mm "
              "manually and re-run." % (exc, target_m*1000))


# ======================================================================
# 5. MAIN
# ======================================================================

async def moving_pick_place():
    print("\n" + "=" * 70)
    print("LYNXMOTION — ON-THE-FLY MOVING PICK & PLACE")
    print("=" * 70)

    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()

    # (a) STOP -> configure gripper (+ resize cube) -> PLAY
    if not timeline.is_stopped():
        print("\n[1] stopping timeline to configure gripper...")
        timeline.stop(); await wait_frames(5)
    configure_gripper(stage)
    if OBJECT_SHAPE == "cylinder":
        prepare_object(stage)                       # build cylinder, hide cube
    else:
        oc = stage.GetPrimAtPath(ORIG_CUBE_PATH)    # re-activate cube if a prior
        if oc.IsValid() and not oc.IsActive():      # cylinder run hid it
            oc.SetActive(True)
        if CUBE_TARGET_SIZE is not None:
            resize_cube(stage, CUBE_TARGET_SIZE)
    if FRICTION_BOOST:
        boost_friction(stage, GRIP_FRICTION)
    stabilize_physics(stage)
    print("\n[1] starting simulation...")
    timeline.play(); await wait_frames(30)

    # (b) init robot + prims
    robot = SingleArticulation(prim_path=ROBOT_PATH, name="lynx_mpp")
    robot.initialize()
    base_prim  = SingleXFormPrim(prim_path=BASE_PATH,  name="lynx_base")
    ee_prim    = SingleXFormPrim(prim_path=EE_PATH,    name="lynx_ee")
    wrist_prim = SingleXFormPrim(prim_path=WRIST_PATH, name="lynx_wrist")
    cube_prim  = SingleXFormPrim(prim_path=CUBE_PATH,  name="lynx_cube")
    finger_prims = {n: SingleXFormPrim(prim_path=p, name="lynx_"+n)
                    for n, p in FINGER_PRIMS.items()}

    def fingertips_world():
        pts = {}
        for n, fp in finger_prims.items():
            p, q = fp.get_world_pose()
            pts[n] = as64(p) + quat_to_R(q) @ FINGER_TIP_LOCAL[n]
        return pts

    print("\n[2] DOF order:", list(robot.dof_names))
    base_pos, base_quat = base_prim.get_world_pose()
    base_pos, base_quat = as64(base_pos), as64(base_quat)
    cube_start, _ = cube_prim.get_world_pose(); cube_start = as64(cube_start)
    cube_dims, _, _ = world_bbox(CUBE_PATH)
    cube_z0 = float(cube_start[2]); lane_y = float(cube_start[1])
    print("    base :", np.round(base_pos, 4))
    print("    cube :", np.round(cube_start, 4), " dims(mm):", np.round(cube_dims*1000, 1))

    # cube rests on the belt; grasp geometry (below) is derived once the
    # grasp orientation is chosen, since the vertical tip-drop depends on it
    belt_top_z = cube_z0 - float(cube_dims[2]) / 2.0

    _, ped_center, ped_top_z = world_bbox(PEDESTAL_PATH)
    place_pos = np.array([ped_center[0], ped_center[1],
                          ped_top_z + cube_dims[2]/2.0 + PLACE_CLEARANCE])
    print("    pedestal top Z:", round(ped_top_z, 4), " -> place:", np.round(place_pos, 4))

    # conveyor + side rails (read live) -> rail-top Z and the near-rail Y band
    try:
        conv_dims, conv_center, conv_top_z = world_bbox(CONVEYOR_PATH)
        conv_min_y = conv_center[1] - conv_dims[1] / 2.0
        conv_max_y = conv_center[1] + conv_dims[1] / 2.0
        conv_min_x = conv_center[0] - conv_dims[0] / 2.0
        conv_max_x = conv_center[0] + conv_dims[0] / 2.0
        rail_top_z = float(conv_top_z)
    except Exception as exc:
        print("    [WARN] could not read conveyor bbox (%s); using belt top" % exc)
        rail_top_z = cube_z0 - float(cube_dims[2]) / 2.0
        conv_min_y, conv_max_y = lane_y - 0.2, lane_y + 0.9
        conv_min_x, conv_max_x = -2.0, 2.5
    # The conveyor bbox top (rail_top_z) includes the tall belt superstructure
    # (~2.3 m), which is well beyond the 550 mm arm's reach.  Straining the EE
    # toward it made the IK fail and over-yank the arm (the 100+ mm over-lift and
    # the bounce).  Cap every "clear" height to a KNOWN-reachable ceiling: a lift
    # of ~READY_HEIGHT above the belt (the READY pose already sits there).
    reach_ceiling_z = cube_z0 + READY_HEIGHT + 0.02
    clear_z = min(rail_top_z + RAIL_CLEAR_MARGIN, reach_ceiling_z)
    print("    conveyor rail-top Z: %.4f  reach ceiling: %.4f  -> travel/clear Z: %.4f"
          % (rail_top_z, reach_ceiling_z, clear_z))
    print("    conveyor Y band: [%.3f, %.3f]  near rail ~%.3f"
          % (conv_min_y, conv_max_y, conv_min_y))

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

    # low-pass filtered arm command: the per-frame IK solution can wobble
    # frame-to-frame (redundant DOF), which shows up as gripper JITTER.  An EMA
    # on the commanded joints (init to the current seed, no start-up jump) makes
    # the wrist glide instead of buzz.  Only used in the per-frame tracking
    # loops; the joint-space transfers are already smooth via interp().
    _q_cmd = [None]
    def command_arm_smooth(q_target, reset=False):
        q_target = as64(q_target)
        if reset or _q_cmd[0] is None:
            _q_cmd[0] = q_target.copy()
        else:
            _q_cmd[0] = CMD_SMOOTH * _q_cmd[0] + (1.0 - CMD_SMOOTH) * q_target
        command_arm(_q_cmd[0])
        return _q_cmd[0]

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

    # (d) GRIPPER SELF-TEST + APERTURE CALIBRATION
    print("\n[3] GRIPPER SELF-TEST")
    for _ in range(25): command_gripper(GRIPPER_OPEN_Q7); await next_frame()
    q_open = read_gripper(); tips_open = fingertips_world()
    for _ in range(25): command_gripper(GRIPPER_CLOSED_Q7); await next_frame()
    q_closed = read_gripper(); tips_closed = fingertips_world()
    for _ in range(20): command_gripper(GRIPPER_OPEN_Q7); await next_frame()
    q_reopen = read_gripper()
    print("    OPEN  :", np.round(q_open*1000, 2), "mm")
    print("    CLOSED:", np.round(q_closed*1000, 2), "mm")
    print("    REOPEN:", np.round(q_reopen*1000, 2), "mm")

    def symmetric(q, tol=0.0015):
        return abs(q[1]+q[0]) < tol and abs(q[2]+q[0]) < tol
    # OPEN command -> q7 ~ 0 (wide); CLOSED command -> q7 ~ 10 mm (narrow)
    if not (symmetric(q_open) and q_open[0] < 0.004 and
            symmetric(q_closed) and q_closed[0] > 0.006):
        print("\n[ABORT] gripper not symmetric — remove the Physx Mimic Joint API")
        print("        from joint_8 & joint_9 in the USD, save, stop, re-run.")
        return
    print("    --> gripper OK (symmetric open & close)")

    # aperture calibration: measure the fingertip spread + grasp center
    def radii(tips):
        pts = np.array([tips[n] for n in FINGER_PRIMS])
        c = pts.mean(axis=0)
        return c, np.linalg.norm(pts - c, axis=1)      # centroid, per-finger radius
    c_open, r_open = radii(tips_open)
    c_closed, r_closed = radii(tips_closed)
    ee_p0, _ = ee_prim.get_world_pose()
    gc_from_ee = float(np.linalg.norm(c_closed - as64(ee_p0)))
    # Top-down insertion is limited by the finger INNER clearance (from the STL
    # meshes ~18 mm radius open).  A CUBE is limited by its corners (~24 mm), but
    # a ROUND object has no corners, so it fits up to ~2*18 = ~36 mm dia.
    TOPDOWN_MAX_CUBE = 0.036 if OBJECT_SHAPE == "cylinder" else 0.024
    print("\n[3b] APERTURE CALIBRATION (measured from the live gripper)")
    print("    fingertip radius from grasp-center: closed ~%.0f mm, open ~%.0f mm"
          % (r_closed.mean()*1000, r_open.mean()*1000))
    print("    grasp-center %.1f mm from pro_arm_ee (using STL const %.1f mm)"
          % (gc_from_ee*1000, GRASP_CENTER_FROM_EE*1000))
    # object horizontal size: cylinder diameter, else cube face/target
    if OBJECT_SHAPE == "cylinder":
        tgt = OBJECT_DIAMETER
    else:
        tgt = CUBE_TARGET_SIZE if CUBE_TARGET_SIZE else float(np.mean(cube_dims[:2]))
    print("    TOP-DOWN insertion limit (finger inner clearance): object <= ~%.0f mm"
          % (TOPDOWN_MAX_CUBE*1000))
    if tgt > TOPDOWN_MAX_CUBE:
        print("    [WARN] object %.0f mm is too big for top-down insertion (<= ~%.0f mm)."
              % (tgt*1000, TOPDOWN_MAX_CUBE*1000))
    else:
        print("    object %.0f mm fits between the open fingers (<= ~%.0f mm)."
              % (tgt*1000, TOPDOWN_MAX_CUBE*1000))

    # (e) TOP-DOWN grasp orientation (wrist rides ABOVE the cube -> clears the
    #     near rail) + grasp geometry + tracking helpers + collision monitor
    ready_q = np.deg2rad(READY_Q_DEG)
    # grasp-center -> ee distance.  The Lula collision spheres sit ~20 mm above
    # the real STL fingertips, so the live tip-centroid measurement is NOT
    # reliable here; use the STL-derived constant (~5 mm).
    gc = GRASP_CENTER_FROM_EE
    # top-down EE sits ~gc BELOW the cube center; probe there for reachability
    probe = np.array([pick_x, lane_y, cube_z0 + GRASP_CENTER_ABOVE_CUBE - gc])
    best = None
    for tilt in (0.0, 10.0, -10.0, 20.0, -20.0):     # prefer most-vertical
        for deg in range(0, 360, TOPDOWN_YAW_STEP):
            R = _Ry(np.radians(tilt)) @ _Rz(np.radians(deg))
            q, ok = solve(probe, rot_to_quat(R), ready_q)
            if not ok:
                continue
            # penalise tilt heavily, then prefer the smoothest (elbow-up) one
            score = abs(tilt) * 100.0 + float(np.linalg.norm(q - ready_q))
            if best is None or score < best[0]:
                best = (score, R, deg, tilt)
    if best is not None:
        R_track_mat = best[1]
        R_track = rot_to_quat(R_track_mat)
        print("\n[4] TOP-DOWN grasp orientation selected: yaw=%d deg tilt=%d deg "
              "(reachable; wrist above the cube, clear of the rail)"
              % (best[2], best[3]))
    else:
        _, R_track_mat = solver.compute_forward_kinematics(EE_FRAME, ready_q)
        R_track_mat = np.asarray(R_track_mat, dtype=np.float64)
        R_track = rot_to_quat(R_track_mat)
        print("\n[4] [WARN] no top-down IK solution at the cube — using the tilted")
        print("    READY orientation, which may clip the near rail. If the wrist")
        print("    collides, move the robot a few cm closer to the conveyor.")
    R_grasp = R_track

    # grasp-center model: we command the GRASP CENTER (where the fingers
    # converge, gc from the EE toward the gripper) to sit at the cube center,
    # then convert to the EE target that IK needs.
    approach_world = R_track_mat @ np.array([0.0, 0.0, -1.0])
    offset_ee = np.array([0.0, 0.0, gc])                 # grasp center in the EE frame
    def ee_for_center(center):                            # EE target to put center at `center`
        return as64(center) - R_track_mat @ offset_ee
    def center_from_ee(ee_world):                         # current grasp center from the EE
        return as64(ee_world) + R_track_mat @ offset_ee
    # fingertips are ~gc below the grasp center; keep them above the belt and
    # (until aligned) above the cube top
    center_floor_z  = belt_top_z + FINGERTIP_BELT_CLEAR + gc
    safe_center_off = float(cube_dims[2]) / 2.0 + gc + 0.003
    print("    approach axis (world): %s" % np.round(approach_world, 3))
    print("    grasp-center offset from ee: %.1f mm (STL const)" % (gc*1000))
    print("    grasp center -> cube center %+0.0f mm; belt floor Z %.4f"
          % (GRASP_CENTER_ABOVE_CUBE*1000, center_floor_z))

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

    # collision monitor: warn if the wrist dips into the near-rail zone
    coll_warns = [0]
    near_lo, near_hi = conv_min_y - 0.03, conv_min_y + 0.10
    def collision_check(tag):
        if not COLLISION_MONITOR:
            return
        wp, _ = wrist_prim.get_world_pose(); wp = as64(wp)
        if (conv_min_x <= wp[0] <= conv_max_x and near_lo <= wp[1] <= near_hi
                and wp[2] < rail_top_z + 0.005):
            coll_warns[0] += 1
            if coll_warns[0] <= 6 or coll_warns[0] % 10 == 0:
                print("    [COLLISION] wrist in near-rail zone Y=%.3f Z=%.4f "
                      "(rail top %.4f) [%s] #%d"
                      % (wp[1], wp[2], rail_top_z, tag, coll_warns[0]))

    # (f) open + move to READY
    print("\n[5] opening gripper + moving to READY")
    q_now = as64(robot.get_joint_positions())[:6]
    await interp(command_arm, command_gripper, q_now, ready_q, READY_MOVE_FRAMES,
                 GRIPPER_OPEN_Q7)

    # (f2) move to HOVER over the intercept.  With a TOP-DOWN grasp the wrist
    #      rides directly above the cube (at the lane, PAST the near rail), so
    #      the gripper clears the rail at any height -- a moderate hover is fine.
    hover_center = np.array([pick_x, lane_y, cube_z0 + HOVER_HEIGHT])
    intercept = ee_for_center(hover_center)
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

    # (g) WAIT until cube reaches the approach zone (with a no-motion failsafe)
    wait_x0 = float(as64(cube_p)[0]); waited = 0
    while True:
        cube_p, _ = cube_prim.get_world_pose()
        command_arm(q_seed); command_gripper(GRIPPER_OPEN_Q7)
        if float(as64(cube_p)[0]) >= approach_x:
            break
        waited += 1
        if waited > 1200 and float(as64(cube_p)[0]) < wait_x0 + 0.02:
            print("\n[ABORT] object isn't moving on the belt — the conveyor may not be "
                  "conveying the created cylinder. Check /World/ConveyorBelt_A09, or set "
                  "OBJECT_SHAPE='cube' to use the scene cube.")
            return
        await next_frame()

    # (h) APPROACH: descend + track + match velocity -> capture gate
    print("\n[7] APPROACH  (descend, track, match cube speed)")
    prev_cube = as64(cube_p); prev_ee, _ = ee_prim.get_world_pose(); prev_ee = as64(prev_ee)
    prev_t = time.monotonic(); v_cube = np.zeros(3); v_ee = np.zeros(3)
    ik_fails = 0; good = 0; frame = 0; last_exy = 1.0
    cube_x_max = float(cube_p[0]); knocked = False

    while True:
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        cube_x = float(cube_p[0])
        # the belt only moves the cube +X; a backward jump means we hit it
        if cube_x < cube_x_max - 0.02 and not knocked:
            knocked = True
            print("    [CONTACT] cube shoved back %.0f mm — the gripper is hitting it "
                  "(not clearing). Lower CUBE_TARGET_SIZE / check GRASP_CENTER_FROM_EE."
                  % ((cube_x_max - cube_x) * 1000))
        cube_x_max = max(cube_x_max, cube_x)
        now = time.monotonic(); dt = max(now - prev_t, 1e-3)
        v_cube = VEL_SMOOTH*v_cube + (1-VEL_SMOOTH)*((cube_p - prev_cube)/dt)
        prev_cube, prev_t = cube_p.copy(), now

        # descend the GRASP CENTER from hover height to the cube center as
        # the cube crosses approach->capture
        prog = np.clip((cube_x - approach_x) / max(capture_x - approach_x, 1e-6), 0, 1)
        c_off = HOVER_HEIGHT + smoothstep(prog) * (GRASP_CENTER_ABOVE_CUBE - HOVER_HEIGHT)
        lead = LEAD_X * (1.0 - prog)                 # small +X lead, decays to 0

        # keep the fingers above the cube top until laterally aligned
        if last_exy > DESCEND_GATE_XY:
            c_off = max(c_off, safe_center_off)

        center_tgt = cube_p.copy()
        center_tgt[0] = cube_p[0] + v_cube[0]*TAU_CAPTURE + lead
        center_tgt[1] = cube_p[1] + v_cube[1]*TAU_CAPTURE
        center_tgt[2] = max(cube_p[2] + c_off, center_floor_z)   # belt floor
        target = ee_for_center(center_tgt)           # EE target for IK

        q_new, ok = solve_track(target, q_seed)
        if ok:
            q_seed = step_toward(q_seed, q_new); ik_fails = 0
        else:
            ik_fails += 1
        command_arm_smooth(q_seed); command_gripper(GRIPPER_OPEN_Q7)
        await next_frame()
        collision_check("approach")

        ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
        v_ee = VEL_SMOOTH*v_ee + (1-VEL_SMOOTH)*((ee_p - prev_ee)/dt); prev_ee = ee_p.copy()
        # error of the GRASP CENTER vs the cube center
        cur_center = center_from_ee(ee_p)
        desired_center = cube_p.copy(); desired_center[2] = cube_p[2] + GRASP_CENTER_ABOVE_CUBE
        err = cur_center - desired_center
        exy = float(np.linalg.norm(err[:2])); ez = float(err[2])
        relv = float(np.linalg.norm(v_ee - v_cube))
        last_exy = exy

        if frame % 6 == 0:
            print(f"    x={cube_x:.3f} c_off={c_off*1000:+5.1f}mm exy={exy*1000:5.1f} "
                  f"ez={ez*1000:+5.1f}mm relv={relv*1000:5.1f}mm/s "
                  f"grip=OPEN(q7={read_gripper()[0]*1000:.1f}mm) good={good}")

        aligned = exy < CAP_XY_TOL and abs(ez) < CAP_Z_TOL and relv < CAP_RELV_TOL
        good = good + 1 if (cube_x >= capture_x and aligned) else 0
        if good >= REQUIRED_GOOD_FRAMES:
            print("\n    >>> CAPTURE ALIGNED  exy=%.1fmm ez=%.1fmm relv=%.1fmm/s"
                  % (exy*1000, ez*1000, relv*1000))
            print("        (gripper held OPEN the whole approach: q7=%.1f mm)"
                  % (read_gripper()[0]*1000))
            break
        if ik_fails > MAX_IK_FAILS:
            print("\n[ABORT] repeated IK failure — pick point may be out of reach.")
            return
        if cube_x >= dead_x:
            print("\n[ABORT] cube passed deadline without alignment "
                  "(exy=%.1fmm relv=%.1fmm/s). Not chasing downstream."
                  % (exy*1000, relv*1000))
            if coll_warns[0]:
                print("        NOTE: %d wrist/near-rail collisions were detected — "
                      "the arm was knocked off the cube. If top-down still clips "
                      "the rail, move the robot a few cm toward the conveyor."
                      % coll_warns[0])
            return
        frame += 1

    # (h2) SETTLE the cube inside the OPEN fingers before closing — keep the
    #      gripper fully OPEN and track the cube so it is definitely between
    #      the jaws, then (and only then) close.
    print("\n[7b] SETTLE (gripper OPEN around the centered cube)")
    for _ in range(SETTLE_OPEN_FRAMES):
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        now = time.monotonic(); dt = max(now - prev_t, 1e-3)
        v_cube = VEL_SMOOTH*v_cube + (1-VEL_SMOOTH)*((cube_p - prev_cube)/dt); prev_cube, prev_t = cube_p.copy(), now
        center_tgt = cube_p.copy()
        center_tgt[0] += v_cube[0]*TAU_CAPTURE; center_tgt[1] += v_cube[1]*TAU_CAPTURE
        center_tgt[2] = max(cube_p[2] + GRASP_CENTER_ABOVE_CUBE, center_floor_z)
        q_new, ok = solve_track(ee_for_center(center_tgt), q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm_smooth(q_seed); command_gripper(GRIPPER_OPEN_Q7)   # STILL OPEN
        await next_frame()
    print("    cube seated between OPEN fingers (q7=%.1f mm) — closing now"
          % (read_gripper()[0]*1000))

    # (i) CLOSE while tracking (velocity matched)
    print("\n[8] CLOSING (still tracking)")
    for f in range(CLOSE_FRAMES):
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        now = time.monotonic(); dt = max(now - prev_t, 1e-3)
        v_cube = VEL_SMOOTH*v_cube + (1-VEL_SMOOTH)*((cube_p - prev_cube)/dt); prev_cube, prev_t = cube_p.copy(), now
        center_tgt = cube_p.copy()
        center_tgt[0] += v_cube[0]*TAU_CAPTURE; center_tgt[1] += v_cube[1]*TAU_CAPTURE
        center_tgt[2] = max(cube_p[2] + GRASP_CENTER_ABOVE_CUBE, center_floor_z)
        target = ee_for_center(center_tgt)
        q_new, ok = solve_track(target, q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm_smooth(q_seed)
        a = smoothstep((f + 1) / CLOSE_FRAMES)
        command_gripper(GRIPPER_OPEN_Q7 + a*(GRIPPER_CLOSED_Q7 - GRIPPER_OPEN_Q7))
        await next_frame()
        collision_check("close")

    for _ in range(POST_CLOSE_HOLD):
        cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
        center_tgt = cube_p.copy()
        center_tgt[2] = max(cube_p[2] + GRASP_CENTER_ABOVE_CUBE, center_floor_z)
        target = ee_for_center(center_tgt)
        q_new, ok = solve_track(target, q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm_smooth(q_seed); command_gripper(GRIPPER_CLOSED_Q7); await next_frame()

    print("    gripper after close:", np.round(read_gripper()*1000, 2), "mm")

    # (j) QUICK LIFT (straight up) + verify rise.  A short, gentle lift to just
    #     pull the object off the belt -- NOT a strain up to the tall conveyor
    #     bbox top (that over-yanked the arm and bounced the object).  The top-
    #     down grasp keeps the wrist in the lane past the near rail, so a small
    #     vertical lift needs no rail clearance here.
    print("\n[9] QUICK LIFT")
    ee_p, _ = ee_prim.get_world_pose(); ee_p = as64(ee_p)
    lift_start = ee_p.copy()
    lift_end = lift_start.copy()
    lift_end[2] = lift_start[2] + QUICK_LIFT_HEIGHT
    for f in range(QUICK_LIFT_FRAMES):
        a = smoothstep((f + 1) / QUICK_LIFT_FRAMES)
        q_new, ok = solve_track(lift_start + a*(lift_end - lift_start), q_seed)
        if ok: q_seed = step_toward(q_seed, q_new)
        command_arm_smooth(q_seed); command_gripper(GRIPPER_CLOSED_Q7); await next_frame()

    cube_p, _ = cube_prim.get_world_pose(); cube_p = as64(cube_p)
    rise = float(cube_p[2] - cube_z0)
    print("    cube rise after quick lift: %.1f mm" % (rise*1000))
    if rise < CUBE_RISE_GATE:
        print("\n[STOP] cube did NOT rise (%.1f < %.1f mm gate). Grasp failed — "
              "not transferring." % (rise*1000, CUBE_RISE_GATE*1000))
        for _ in range(20): command_gripper(GRIPPER_OPEN_Q7); command_arm(q_seed); await next_frame()
        q0 = as64(robot.get_joint_positions())[:6]
        await interp(command_arm, command_gripper, q0, ready_q, READY_MOVE_FRAMES, GRIPPER_OPEN_Q7)
        qg = read_gripper()
        print("\n[NEXT] read the 'gripper after close' q7 above vs the [3b] calibration:")
        print("   * q7 stalled > ~3 mm (~%.1f mm now) -> gripped but SLIPPED:" % (qg[0]*1000))
        print("       raise GRIP_FRICTION (e.g. 1.6) and/or FINGER_MAX_FORCE; try a")
        print("       slightly larger cube (0.022) for a broader contact")
        print("   * q7 near 0 -> no contact: lower CUBE_TARGET_SIZE (fingers over-closed)")
        print("       or nudge GRASP_CENTER_ABOVE_CUBE so the jaws sit on the cube body")
        return
    print("    grasp CONFIRMED (cube is being carried).")

    # (k) TRANSFER up-and-over (joint-space, solved once), all above rail tops
    print("\n[10] TRANSFER to pedestal")
    # keep the carry height reachable (see reach_ceiling_z above) but always
    # above the place point so the final descent onto the pedestal is downward.
    carry_z = max(min(cube_z0 + CARRY_HEIGHT, reach_ceiling_z), float(place_pos[2]) + 0.05)
    print("    carry Z: %.4f  (reach ceiling %.4f, place Z %.4f)"
          % (carry_z, reach_ceiling_z, float(place_pos[2])))

    # solve_key targets a GRASP-CENTER (cube) position; converts to the EE
    def solve_key(center, warm, label):
        pos = ee_for_center(center)
        q, ok = solve(pos, R_grasp, warm)
        if not ok:
            q, ok = solve(pos, None, warm)
        if not ok:
            print("    [WARN] IK failed for %s at %s" % (label, np.round(center, 3)))
            return warm, False
        return q, True

    cur_center = center_from_ee(as64(ee_prim.get_world_pose()[0]))
    q_up,   _ = solve_key(np.array([cur_center[0], cur_center[1], carry_z]), q_seed, "lift-high")
    q_over, _ = solve_key(np.array([place_pos[0], place_pos[1], carry_z]), q_up, "over-pedestal")
    q_place, _ = solve_key(place_pos, q_over, "place")   # cube center -> place_pos

    # start the transfer from the ACTUAL last commanded pose (the low-pass state
    # lags q_seed a touch) so the hand-off from tracking to transfer has no tick
    q_cmd_now = _q_cmd[0].copy() if _q_cmd[0] is not None else q_seed
    await interp(command_arm, command_gripper, q_cmd_now, q_up, TRANSFER_FRAMES // 2, GRIPPER_CLOSED_Q7)
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
    print("  rail collisions: %d" % coll_warns[0])
    print("\n  OVERALL: %s" %
          ("PICK & PLACE SUCCESS" if (xy_err < 0.03 and z_err < 0.03)
           else "placed, but check pedestal error above"))
    print("=" * 70)


# ======================================================================
# 5b. STATIC GRASP PARAMETER SWEEP  (RUN_MODE = "sweep")
# ======================================================================

async def static_grasp(stage, timeline, solver, d, mu, zoff, force, ready_q,
                       stand_xy, ped_top):
    """One static top-down grasp of a cylinder resting on the pedestal.
    Returns the object's rise (m); negative sentinels for unreachable."""
    if not timeline.is_stopped():
        timeline.stop(); await wait_frames(3)
    # finger force for this combo
    try:
        fp = find_finger_joint_prims(stage)
        for n in FINGER_JOINT_NAMES:
            if fp.get(n) is not None:
                set_linear_drive(fp[n], FINGER_DRIVE_STIFFNESS, FINGER_DRIVE_DAMPING, force)
    except Exception:
        pass
    # (re)build the cylinder on the pedestal; hide the scene cube
    h = SWEEP_OBJ_HEIGHT
    make_cylinder(stage, OBJECT_PATH, d, h,
                  (stand_xy[0], stand_xy[1], ped_top + h/2.0 + 0.001), OBJECT_MASS)
    c = stage.GetPrimAtPath(ORIG_CUBE_PATH)
    if c.IsValid():
        c.SetActive(False)
    boost_friction(stage, mu)
    timeline.play(); await wait_frames(20)

    robot = SingleArticulation(prim_path=ROBOT_PATH, name="sweep_robot")
    robot.initialize()
    ee_prim  = SingleXFormPrim(prim_path=EE_PATH,     name="sweep_ee")
    obj_prim = SingleXFormPrim(prim_path=OBJECT_PATH, name="sweep_obj")

    def cmd_arm(q):
        robot.apply_action(ArticulationAction(
            joint_positions=np.asarray(q, dtype=np.float32), joint_indices=ARM_INDICES))
    def cmd_grip(q7):
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets(q7), joint_indices=GRIPPER_INDICES))
    def solve(pos, R, warm):
        if R is None:
            q, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME, target_position=as64(pos), target_orientation=None,
                warm_start=as64(warm), position_tolerance=IK_POS_TOL)
        else:
            q, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME, target_position=as64(pos), target_orientation=as64(R),
                warm_start=as64(warm), position_tolerance=IK_POS_TOL,
                orientation_tolerance=IK_ORI_TOL)
        return as64(q), bool(ok)

    obj_p, _ = obj_prim.get_world_pose(); obj_p = as64(obj_p); obj_z0 = float(obj_p[2])
    gc = GRASP_CENTER_FROM_EE
    offset = np.array([0.0, 0.0, gc])

    # pick a reachable top-down orientation at the grasp point
    probe = np.array([obj_p[0], obj_p[1], obj_z0 + zoff - gc])
    R = None
    for tilt in (0.0, 10.0, -10.0, 20.0, -20.0):
        for deg in range(0, 360, 30):
            Rq = grasp_quat(deg, tilt)
            _, ok = solve(probe, Rq, ready_q)
            if ok:
                R = Rq; break
        if R is not None:
            break
    if R is None:
        timeline.stop(); return -0.999
    R_mat = quat_to_R(R)
    def ee_for_center(cw):
        return as64(cw) - R_mat @ offset

    # to READY (open), then hover over the object, then descend
    q0 = as64(robot.get_joint_positions())[:6]
    for f in range(50):
        a = smoothstep((f+1)/50); cmd_arm(q0 + a*(ready_q-q0)); cmd_grip(GRIPPER_OPEN_Q7)
        await next_frame()
    q_hover, ok = solve(ee_for_center(np.array([obj_p[0], obj_p[1], obj_z0+0.06])), R, ready_q)
    if not ok:
        timeline.stop(); return -0.998
    qc = as64(robot.get_joint_positions())[:6]
    for f in range(50):
        a = smoothstep((f+1)/50); cmd_arm(qc + a*(q_hover-qc)); cmd_grip(GRIPPER_OPEN_Q7)
        await next_frame()
    q_seed = q_hover
    q_grasp, ok = solve(ee_for_center(np.array([obj_p[0], obj_p[1], obj_z0+zoff])), R, q_seed)
    if ok:
        for f in range(45):
            a = smoothstep((f+1)/45); cmd_arm(q_seed + a*(q_grasp-q_seed)); cmd_grip(GRIPPER_OPEN_Q7)
            await next_frame()
        q_seed = q_grasp
    # close, hold
    for f in range(CLOSE_FRAMES):
        a = smoothstep((f+1)/CLOSE_FRAMES)
        cmd_arm(q_seed); cmd_grip(GRIPPER_OPEN_Q7 + a*(GRIPPER_CLOSED_Q7-GRIPPER_OPEN_Q7))
        await next_frame()
    for _ in range(POST_CLOSE_HOLD):
        cmd_arm(q_seed); cmd_grip(GRIPPER_CLOSED_Q7); await next_frame()
    # lift straight up
    q_lift, ok = solve(ee_for_center(np.array([obj_p[0], obj_p[1], obj_z0+zoff+0.08])), R, q_seed)
    if ok:
        for f in range(60):
            a = smoothstep((f+1)/60); cmd_arm(q_seed + a*(q_lift-q_seed)); cmd_grip(GRIPPER_CLOSED_Q7)
            await next_frame()
    for _ in range(20):
        cmd_arm(q_seed); cmd_grip(GRIPPER_CLOSED_Q7); await next_frame()

    obj_f, _ = obj_prim.get_world_pose()
    rise = float(as64(obj_f)[2] - obj_z0)
    timeline.stop(); await wait_frames(2)
    return rise


async def grasp_sweep():
    print("\n" + "=" * 70)
    print("STATIC GRASP PARAMETER SWEEP (cylinder on the pedestal)")
    print("=" * 70)
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    if not timeline.is_stopped():
        timeline.stop(); await wait_frames(3)

    _, ped_c, ped_top = world_bbox(PEDESTAL_PATH)
    solver = LulaKinematicsSolver(robot_description_path=ROBOT_DESCRIPTION, urdf_path=URDF)
    base_prim = SingleXFormPrim(prim_path=BASE_PATH, name="sweep_base")
    bpos, bq = base_prim.get_world_pose()
    solver.set_robot_base_pose(as64(bpos), as64(bq))
    ready_q = np.deg2rad(READY_Q_DEG)
    stand_xy = (float(ped_c[0]), float(ped_c[1]))

    total = (len(SWEEP_DIAMETERS)*len(SWEEP_FRICTIONS)*len(SWEEP_ZOFFSETS)*len(SWEEP_FORCES))
    print("    %d combos; object h=%.0f mm on pedestal top Z=%.3f\n"
          % (total, SWEEP_OBJ_HEIGHT*1000, float(ped_top)))
    rows = []; i = 0
    for d in SWEEP_DIAMETERS:
        for mu in SWEEP_FRICTIONS:
            for zo in SWEEP_ZOFFSETS:
                for fN in SWEEP_FORCES:
                    i += 1
                    try:
                        rise = await static_grasp(stage, timeline, solver, d, mu, zo, fN,
                                                   ready_q, stand_xy, float(ped_top))
                    except Exception as exc:
                        rise = None
                        print("    combo %d raised: %s" % (i, exc))
                    ok = (rise is not None and rise > SWEEP_RISE_OK)
                    tag = ("OK " if ok else ("unreach" if (rise is not None and rise < -0.5) else " x "))
                    rows.append((d, mu, zo, fN, rise))
                    rr = ("%.1f" % (rise*1000)) if rise is not None else "ERR"
                    print("  [%2d/%2d] d=%2.0f mu=%.1f zoff=%+3.0f F=%2.0f -> rise=%6s mm  %s"
                          % (i, total, d*1000, mu, zo*1000, fN, rr, tag))

    # save CSV
    try:
        with open(SWEEP_CSV, "w") as f:
            f.write("diameter_mm,friction,z_offset_mm,force_N,rise_mm,success\n")
            for d, mu, zo, fN, rise in rows:
                rmm = "" if rise is None else "%.2f" % (rise*1000)
                succ = 1 if (rise is not None and rise > SWEEP_RISE_OK) else 0
                f.write("%.1f,%.2f,%.1f,%.1f,%s,%d\n"
                        % (d*1000, mu, zo*1000, fN, rmm, succ))
        print("\n    results saved: %s" % SWEEP_CSV)
    except Exception as exc:
        print("\n    could not save CSV (%s)" % exc)

    goods = [r for r in rows if r[4] is not None and r[4] > SWEEP_RISE_OK]
    goods.sort(key=lambda r: -r[4])
    print("\n  TOP GRASPS (largest lift):")
    for d, mu, zo, fN, rise in goods[:8]:
        print("    d=%2.0fmm mu=%.1f zoff=%+3.0fmm F=%2.0fN -> %.1f mm"
              % (d*1000, mu, zo*1000, fN, rise*1000))
    if not goods:
        print("    none lifted — widen the grid (bigger diameter / friction / force)")
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

_entry = grasp_sweep if RUN_MODE == "sweep" else moving_pick_place
_mpp_task = asyncio.ensure_future(_entry())
_mpp_task.add_done_callback(_done)
print("[LYNX] task scheduled (RUN_MODE=%s) — running now" % RUN_MODE)
