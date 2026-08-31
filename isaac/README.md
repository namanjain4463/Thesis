# Lynxmotion moving pick & place — controller

`lynxmotion_moving_pick_place.py` — a deterministic on-the-fly pick of a cube
moving on the conveyor, then place on the pedestal, for the Lynxmotion SES-Pro
550 mm 6-DOF arm + DH CGE-10-10 3-finger gripper in Isaac Sim 5.1.

Run it in the **Script Editor** with **Ctrl+Enter**. It prints immediately,
starts the timeline itself, and surfaces async errors with a traceback.

---

## A. Diagnosis (why the previous attempts failed)

**1. The grasp failed because of a mimic-vs-drive conflict — not arm speed.**
In the URDF, `joint_8` and `joint_9` are *mimic* joints of `joint_7`
(`multiplier = -1`), so the mechanism is `q8 = q9 = -q7`. When Isaac imports
those mimic joints as PhysX coupling constraints and the controller *also*
sends independent position targets to `joint_8`/`joint_9` (as v2 did:
`[+g, -g, -g]` on all three), the constraint and the drives fight each other.
That is exactly the desync in your logs:

```
commanded OPEN : q7=+10.00  q8=-10.00  q9=-10.00
before CLOSE   : q7=+0.00   q8=-9.37   q9=-5.58     <-- fingers no longer coupled
after CLOSE    : q7=+0.00   q8=-0.14   q9=-4.49     <-- asymmetric, no real grip
quick lift     : cube rise = -1.8 mm                <-- nothing grasped
```

The fingers never reach a symmetric aperture, so they never close *on the cube*.

**2. The "random movements" are redundancy snapping.** Position-only IK on a
6-DOF arm leaves 3 redundant DOF, so the solver is free to jump between very
different joint configurations frame to frame, and again at every transfer
frame. That is the snapping / branch-changing you saw.

---

## B. What this controller does differently

**Gripper (the real fix)**
- On startup it **stops the sim, reconfigures the finger joints, then plays** so
  the changes are baked into physics:
  - best-effort **removes the PhysX mimic API** from `joint_8`/`joint_9`, and
  - sets a **stiff, force-capped position drive** on all three fingers
    (`FINGER_DRIVE_STIFFNESS`, `FINGER_DRIVE_DAMPING`, `FINGER_MAX_FORCE`).
  - The three fingers are then commanded as one coordinated unit
    (`q7 -> [+q7, -q7, -q7]`).
- A **gripper self-test** opens and closes before any grasp and **aborts with
  instructions if the fingers are not symmetric** — so you never waste a run on
  a broken gripper. If it aborts, remove the *Physx Mimic Joint* API from
  `joint_8`/`joint_9` in the USD once (some scenes lock it on a session layer
  the script can't edit live), save, stop, re-run.

**Arm (no snapping, no collisions)**
- One **fixed tool orientation** locked from your known-good READY branch is
  used for the whole pick — deterministic IK branch, no snapping.
- **Warm-started full-pose IK** during tracking + a per-frame joint-jump reject
  filter.
- **Transfer is joint-space interpolation** between key poses solved **once**
  (lift-high → over-pedestal → down), each high above the belt and rails, so the
  arm goes up-and-over and the only intended contact is fingers ↔ cube.

**On-the-fly capture (belt never stops)**
- **Catch-up:** the EE aims *ahead* of the cube (`cube + v_cube*TAU_CATCHUP +
  LEAD_X`), so it deliberately runs faster than the cube and closes the gap.
- **Capture:** the lead shrinks to `TAU_CAPTURE`, so the EE **matches the cube's
  velocity** — the close then happens as if the cube were standing still.
- Closes only after `REQUIRED_GOOD_FRAMES` of small XY error **and** small
  relative EE↔cube velocity.

**Never place a cube you didn't grasp**
- After the quick lift it checks the **cube actually rose** (`CUBE_RISE_GATE`).
  If not, it opens, returns to READY, prints the one next thing to try, and
  stops — it never continues to the pedestal on a failed grasp.

---

## C. State machine

```
STOP -> configure gripper -> PLAY
  -> GRIPPER SELF-TEST (abort if asymmetric)
  -> READY (fixed known-good pose)
  -> WAIT for cube to reach track zone
  -> CATCH-UP     (v_ee > v_cube; descend from TRACK_HEIGHT)
  -> CAPTURE      (v_ee -> v_cube; align in XY, low relative speed)
  -> CLOSE        (close 3 fingers while still tracking cube)
  -> QUICK LIFT   (verify cube rose;  fail -> open, return, STOP)
  -> TRANSFER     (up -> over pedestal, joint-space, solved once)
  -> PLACE -> RELEASE -> RETREAT -> report
```

---

## D. The knobs that matter (top of the file)

| Constant | Meaning | If it misses |
|---|---|---|
| `GRASP_Z_BIAS` | final TCP Z at grasp (`pro_arm_ee` isn't exactly the finger center) | sweep ±few mm if fingers sit high/low on the cube |
| `FINGER_MAX_FORCE` | grip force cap (N) | raise if the cube slips; lower if it gets ejected |
| `FINGER_DRIVE_STIFFNESS`/`DAMPING` | finger position-drive gains | raise stiffness if fingers don't hold their target |
| `TAU_CATCHUP` / `LEAD_X` | how aggressively the EE outruns the cube | raise if it never catches up; lower if it overshoots |
| `TAU_CAPTURE` | velocity-match lead near grasp | raise slightly if the cube slides through the close |
| `CAP_XY_TOL`, `CAP_RELV_TOL`, `REQUIRED_GOOD_FRAMES` | how strict the capture gate is | loosen if it hits the deadline; tighten if it closes off-center |
| `CLOSE_FRAMES` | close speed | fewer = faster close = less cube travel during close |
| `CUBE_RISE_GATE` | min rise to accept the grasp | — |
| paths + `READY_Q_DEG` | scene paths and READY branch | edit if your scene/paths differ |

**Cube size:** you approved bumping to ~25 mm if a centered 20 mm close bottoms
out with no preload. If the self-test passes but the cube still slips on a
well-centered close, resize the `/World/Cube` to ~25 mm and re-run before
chasing other knobs.

---

## E. One-run diagnostics to read in the console

- `GRIPPER SELF-TEST` — are `q7/q8/q9` symmetric on open **and** close?
- catch-up lines — `vcx` (cube speed) vs `vex` (EE speed): EE should exceed the
  cube during catch-up, then converge; `relv` should fall below `CAP_RELV_TOL`.
- `CAPTURE ALIGNED` — `exy`, `ez`, `relv` at the instant of closing.
- `gripper after close` — did the fingers close symmetrically on the cube?
- `cube rise after quick lift` — the single number that says grasp / no grasp.
