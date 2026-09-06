"""
gp_core.py — grasp-and-place benchmark CORE (Tier-2 integrated experiment).

Task (fixed BEFORE any model is evaluated, per the 3rd/4th review):
  approach -> close(command) -> lift -> transport(px 0 -> X_FIX at transport speed)
  -> lower into a fixture POCKET -> RELEASE -> WITHDRAW the gripper -> SETTLE (fixed
  period after release) -> label.

Success (measured ONLY after release + withdrawal + settle — a part still held over the
fixture is NOT a success):
  * PLACED: object center within POS_TOL of the pocket center (xy), resting on the pocket
    floor (z within Z_TOL), tilt < TILT_TOL at the END, and it never CATASTROPHICALLY
    tipped (> TIP_FAIL) or dropped during the trajectory.
Failure modes are separated:
  * DROP  — object lost / fell below the pocket floor during transport or after release.
  * TIP   — object catastrophically rotated (> TIP_FAIL deg) at any point (out of grasp / fixture).
  * PLACE — survived transport but final seated pose out of tolerance (missed pocket /
            left leaning / fingers could not seat it).

Tolerances are RESEARCH-benchmark values, NOT industrial-assembly precision (stated plainly):
  POS_TOL=0.006 m, TILT_TOL=12 deg, Z_TOL=0.010 m; catastrophic TIP_FAIL=45 deg.

Embodiments here are GRIPPER CONFIGURATIONS (source S vs a held-out target T with different
finger length / mass / gains / force cap / palm). This is GRIPPER-CONFIGURATION transfer,
NOT yet articulated-arm cross-embodiment — labeled honestly. An articulated arm (Panda) is
the stated next target; the scene/rollout API is written so the body can be swapped.

Action a = (gy grasp-y location, yaw grasp orientation, close finger command, tspeed transport speed).

Object families:
  A uniform  — centered cylinder, roomy pocket; a central grasp works (yaw irrelevant). EASY control.
  B offcom   — bar with a heavy off-center end; a central grasp tips/slips in transit, a
               CoM-ward grasp holds. Discriminates a CoM-aware selector from a naive one.
  C placerest— cylinder + snug-in-x / roomy-in-y pocket: fingers along x (yaw=0) COLLIDE with
               the pocket wall and cannot seat it; fingers along y (yaw≈90 deg) clear.
               Discriminates a placement/orientation-aware selector.
"""
import numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True)

G = 9.81
DT = 0.002
PALM_HOME = 0.6
PED_TOP = 0.20
X_FIX = 0.30
POS_TOL = 0.006
TILT_TOL = 12.0
Z_TOL = 0.010
Z_RELEASE_TOL = 0.018     # part must be lowered to within this of seated BEFORE release (controlled place)
TIP_FAIL = 45.0
SETTLE_STEPS = 200

# A POPULATION of source gripper configurations (train the shared model on these) + one
# held-out TARGET configuration T (different finger length / mass / gains / force-cap / palm).
# This is gripper-CONFIGURATION transfer, NOT yet articulated-arm cross-embodiment.
BODIES = {
    "S1": dict(kp=320.0, kv=8.0, fcap=13.0, fbase=0.040, fhalf=0.006, flen=0.030,
               fmass=0.030, palm=(0.05, 0.03, 0.02), frange=0.045),
    "S2": dict(kp=260.0, kv=7.0, fcap=10.0, fbase=0.044, fhalf=0.006, flen=0.034,
               fmass=0.040, palm=(0.052, 0.03, 0.022), frange=0.048),
    "S3": dict(kp=380.0, kv=9.0, fcap=16.0, fbase=0.038, fhalf=0.005, flen=0.028,
               fmass=0.026, palm=(0.048, 0.028, 0.02), frange=0.043),
    "T":  dict(kp=180.0, kv=6.0, fcap=7.0,  fbase=0.050, fhalf=0.008, flen=0.045,
               fmass=0.060, palm=(0.06, 0.035, 0.025), frange=0.055),
}
SOURCE_BODIES = ("S1", "S2", "S3")
TARGET_BODY = "T"


def _fixture_xml(pocket_hx, pocket_hy, wall_h, wall_t=0.006, mu=1.0):
    t = wall_t; h = wall_h; zf = PED_TOP; base_h = 0.10
    sx, sy = pocket_hx, pocket_hy
    stiff = 'solref="0.003 1" solimp="0.98 0.995 0.001"'   # stiff walls: fingers cannot punch through
    parts = [f'<geom name="fix_base" type="box" pos="{X_FIX} 0 {zf-base_h}" '
             f'size="{sx+t} {sy+t} {base_h}" friction="{mu} 0.01 0.001" {stiff}/>']
    for nm, dx, dy, hx, hy in [("wpx", sx+t/2, 0, t/2, sy+t), ("wnx", -(sx+t/2), 0, t/2, sy+t),
                                ("wpy", 0, sy+t/2, sx+t, t/2), ("wny", 0, -(sy+t/2), sx+t, t/2)]:
        parts.append(f'<geom name="fix_{nm}" type="box" pos="{X_FIX+dx} {dy} {zf+h/2}" '
                     f'size="{hx} {hy} {h/2}" friction="{mu} 0.01 0.001" {stiff}/>')
    return "\n    ".join(parts), zf


def _object_xml(family, mu=1.0, inst=None):
    """Returns (body_xml, fphalf, hz, comy_true, mass_true, pocket_hx, pocket_hy, wall_h).
    inst optionally carries per-instance HIDDEN structural variation:
      'shift' (B CoM offset), 'me' (B end mass). mu is the hidden friction (all families)."""
    inst = inst or {}
    if family == "A":
        r = 0.018; hz = 0.030; m = 0.12
        oz = PED_TOP + hz + 0.001
        body = (f'<body name="object" pos="0 0 {oz}"><freejoint name="obj"/>'
                f'<geom name="obj_main" type="cylinder" size="{r} {hz}" mass="{m}" condim="3" '
                f'friction="{mu} 0.005 0.0001"/></body>')
        return body, r, hz, 0.0, m, r + 0.020, r + 0.020, 0.05      # roomy square pocket
    if family == "B":
        hy = 0.06; hx = 0.014; hz = 0.014; mb = 0.035
        me = float(inst.get("me", 0.20)); shift = float(inst.get("shift", 0.062))
        oz = PED_TOP + hz + 0.001
        comy = me * shift / (mb + me)
        body = (f'<body name="object" pos="0 0 {oz}"><freejoint name="obj"/>'
                f'<geom name="obj_main" type="box" size="{hx} {hy} {hz}" mass="{mb}" condim="3" '
                f'friction="{mu} 0.005 0.0001"/>'
                f'<geom name="obj_end" type="box" pos="0 {shift} 0" size="{hx} 0.014 {hz}" '
                f'mass="{me}" condim="3" friction="{mu} 0.005 0.0001"/></body>')
        return body, hx, hz, comy, mb + me, hx + 0.045, hy + 0.030, 0.05   # roomy both axes
    if family == "C":
        # SHORT puck + TALL pocket walls: fingers must descend BELOW the rim to seat it, so
        # the pocket footprint (snug in x, roomy in y) forces a finger ORIENTATION (yaw~90).
        r = 0.020; hz = 0.016; m = 0.12
        oz = PED_TOP + hz + 0.001
        body = (f'<body name="object" pos="0 0 {oz}"><freejoint name="obj"/>'
                f'<geom name="obj_main" type="cylinder" size="{r} {hz}" mass="{m}" condim="3" '
                f'friction="{mu} 0.005 0.0001"/></body>')
        # snug in x (fingers-along-x collide with wall), roomy in y (fingers-along-y clear)
        return body, r, hz, 0.0, m, r + 0.002, r + 0.030, 0.075
    raise ValueError(family)


def scene(family, body, mu=1.0, inst=None, ts=None):
    b = BODIES[body]
    obj_xml, fphalf, hz, comy, mass, phx, phy, wall_h = _object_xml(family, mu, inst)
    fix_xml, zf = _fixture_xml(phx, phy, wall_h, mu=mu)
    px, py, pz = b["palm"]
    dt = DT if ts is None else ts
    xml = f"""
<mujoco model="gp">
  <option timestep="{dt}" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <default><geom solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <geom name="pedestal" type="box" pos="0 0 0.1" size="0.09 0.09 0.1" friction="{mu} 0.01 0.001"/>
    {fix_xml}
    <body name="palm" pos="0 0 {PALM_HOME}">
      <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/><joint name="pyaw" type="hinge" axis="0 0 1"/>
      <geom name="palm" type="box" size="{px} {py} {pz}" mass="0.3" contype="2" conaffinity="0"/>
      <body name="lfinger" pos="{b['fbase']} 0 -0.05">
        <joint name="lf" type="slide" axis="-1 0 0" range="0 {b['frange']}"/>
        <geom name="lfinger" type="box" size="{b['fhalf']} 0.012 {b['flen']}" mass="{b['fmass']}"
              condim="3" friction="{mu} 0.02 0.002"/>
      </body>
      <body name="rfinger" pos="-{b['fbase']} 0 -0.05">
        <joint name="rf" type="slide" axis="1 0 0" range="0 {b['frange']}"/>
        <geom name="rfinger" type="box" size="{b['fhalf']} 0.012 {b['flen']}" mass="{b['fmass']}"
              condim="3" friction="{mu} 0.02 0.002"/>
      </body>
    </body>
    {obj_xml}
  </worldbody>
  <actuator>
    <position name="apx" joint="px" kp="900" kv="45"/><position name="apy" joint="py" kp="900" kv="45"/>
    <position name="apz" joint="pz" kp="900" kv="45"/><position name="apyaw" joint="pyaw" kp="60" kv="6"/>
    <position name="alf" joint="lf" kp="{b['kp']}" kv="{b['kv']}" ctrlrange="0 {b['frange']}" forcerange="-{b['fcap']} {b['fcap']}"/>
    <position name="arf" joint="rf" kp="{b['kp']}" kv="{b['kv']}" ctrlrange="0 {b['frange']}" forcerange="-{b['fcap']} {b['fcap']}"/>
  </actuator>
</mujoco>"""
    meta = dict(family=family, body=body, fphalf=fphalf, hz=hz, comy=comy, mass=mass, mu=mu,
                pocket_hx=phx, pocket_hy=phy, zf=zf, fbase=b["fbase"], fhalf=b["fhalf"],
                kp=b["kp"], fcap=b["fcap"], frange=b["frange"])
    return xml, meta


def run_episode(family, body, action, mu=1.0, inst=None, record=False, ts=None):
    gy = action["gy"]; yaw = action.get("yaw", 0.0); close = action["close"]; tspeed = action["tspeed"]
    xml, meta = scene(family, body, mu, inst, ts=ts)
    dt = DT if ts is None else ts
    sc = max(1, int(round(DT / dt)))   # scale step COUNTS so physical durations are dt-invariant
    def S(n): return int(n * sc)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    A = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("apx","apy","apz","apyaw","alf","arf")}
    obid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    hz = meta["hz"]; zf = meta["zf"]
    oz0 = float(d.xpos[obid][2])
    gz = PED_TOP + 2*hz + 0.035
    hov = gz + 0.15; lift_z = gz + 0.22
    tip_flag = [False]; frames = []

    def C(px, py, pz, lf, rf):
        d.ctrl[A["apx"]]=px; d.ctrl[A["apy"]]=py; d.ctrl[A["apz"]]=pz-PALM_HOME
        d.ctrl[A["apyaw"]]=yaw; d.ctrl[A["alf"]]=lf; d.ctrl[A["arf"]]=rf
    def tilt():
        R=np.array(d.xmat[obid]).reshape(3,3); return float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))))
    def step(px, py, pz, lf, rf, n):
        for _ in range(n):
            C(px, py, pz, lf, rf); mujoco.mj_step(m,d)
            if tilt() > TIP_FAIL: tip_flag[0]=True
            if record: frames.append((float(d.xpos[obid][0]), float(d.xpos[obid][1]),
                                       float(d.xpos[obid][2]), tilt()))

    # approach at grasp offset gy (palm_y = gy so the grasp point aligns to the object)
    step(0.0, gy, hov, 0, 0, S(60))
    for k in range(S(80)): a=(k+1)/S(80); step(0.0, gy, hov+a*(gz-hov), 0, 0, 1)
    for k in range(S(80)): a=(k+1)/S(80); step(0.0, gy, gz, a*close, a*close, 1)
    step(0.0, gy, gz, close, close, S(40))
    for k in range(S(120)): a=(k+1)/S(120); step(0.0, gy, gz+a*(lift_z-gz), close, close, 1)
    dz_lift = float(d.xpos[obid][2]) - oz0
    # transport: move palm x 0->X_FIX, HOLD py=gy so the OBJECT CENTER lands at pocket center
    dist = X_FIX; nT = max(S(40), int(dist / max(tspeed,1e-3) / dt))
    for k in range(nT): a=(k+1)/nT; step(a*X_FIX, gy, lift_z, close, close, 1)
    # lower into pocket
    place_z = gz + (zf - PED_TOP)
    for k in range(S(120)): a=(k+1)/S(120); step(X_FIX, gy, lift_z+a*(place_z-lift_z), close, close, 1)
    step(X_FIX, gy, place_z, close, close, S(40))
    # object height at the moment of release: a controlled placement must have LOWERED the
    # part to near its seated height. A gripper jammed on the pocket rim that then DROPS the
    # part in is NOT a successful controlled place (mirrors "still held over the fixture != placed").
    z_at_release = float(d.xpos[obid][2])
    # RELEASE
    for k in range(S(60)): a=(k+1)/S(60); step(X_FIX, gy, place_z, close*(1-a), close*(1-a), 1)
    # WITHDRAW
    for k in range(S(120)): a=(k+1)/S(120); step(X_FIX, gy, place_z+a*(hov-place_z), 0, 0, 1)
    # SETTLE (object free)
    step(X_FIX, gy, hov, 0, 0, S(SETTLE_STEPS))

    ox, oy, ozf = float(d.xpos[obid][0]), float(d.xpos[obid][1]), float(d.xpos[obid][2])
    final_tilt = tilt()
    pos_err = float(np.hypot(ox - X_FIX, oy - 0.0))
    z_rest = zf + hz
    z_err = abs(ozf - z_rest)
    rel_z_err = abs(z_at_release - z_rest)     # how far above seated the part was at release
    dropped = (ozf < PED_TOP + 0.4*hz)
    if dropped:
        label = "DROP"
    elif tip_flag[0]:
        label = "TIP"
    elif pos_err <= POS_TOL and final_tilt <= TILT_TOL and z_err <= Z_TOL and rel_z_err <= Z_RELEASE_TOL:
        label = "PLACED"
    else:
        label = "PLACE"
    out = dict(family=family, body=body, label=label, placed=(label == "PLACED"),
               pos_err=pos_err, final_tilt=final_tilt, z_err=z_err, rel_z_err=rel_z_err,
               dz_lift=dz_lift, ox=ox, oy=oy, ozf=ozf, action=action)
    if record: out["frames"] = frames
    return out


if __name__ == "__main__":
    print("SMOKE TEST — nominal grasp per family, source body S1")
    for fam in ("A", "B", "C"):
        _, meta = scene(fam, "S1")
        gy0 = meta["comy"]; yaw0 = (np.pi/2 if fam == "C" else 0.0)
        act = dict(gy=gy0, yaw=yaw0, close=meta["frange"], tspeed=0.4)
        r = run_episode(fam, "S", act)
        print(f"  family {fam}: {r['label']:6s}  pos_err={r['pos_err']*1000:5.1f}mm "
              f"tilt={r['final_tilt']:5.1f}deg z_err={r['z_err']*1000:5.1f}mm dz_lift={r['dz_lift']*1000:5.1f}mm "
              f"(gy={gy0*1000:.0f}mm yaw={np.degrees(yaw0):.0f}deg)")
