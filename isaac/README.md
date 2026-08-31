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
- **Top-down grasp.** The robot sits on the near side of the belt and the cube
  lane is only ~165 mm past the near rail. A *tilted* gripper (the old READY
  orientation leaned 36° off vertical) put the wrist into that rail. The script
  now searches for a **top-down** orientation (fingers straight down → wrist
  rides directly over the cube, 165 mm clear of the rail), preferring the most
  vertical reachable one; it falls back to small belt-aligned tilts, and only
  to the tilted READY orientation if nothing vertical is reachable (with a
  warning to move the robot closer).
- **Grasp height is computed, not hard-coded** — the vertical tip-drop depends
  on the chosen orientation (≈31 mm top-down vs ≈25 mm tilted), so switching
  orientation can't drive the fingers into the belt.
- **Rail-aware travel.** The conveyor bbox is read live for the rail-top height;
  the quick-lift rises above it and the transfer swings across the belt above
  it. A **collision monitor** watches the wrist and prints `[COLLISION]` if it
  ever dips into the near-rail zone (and reports the count at the end).
- **Warm-started full-pose IK** + a per-frame **rate limiter** (smooth, never
  freezes). **Transfer is joint-space interpolation** between key poses solved
  **once**, so the only intended contact is fingers ↔ cube.

**On-the-fly capture (belt never stops), sized to the arm's reach**
- The cube lane is ~0.45 m out in Y — near the edge of the 550 mm arm's
  workspace. So the arm does **not** chase the cube far upstream (that
  over-extends it and the IK fails). Instead it **hovers over the lane at the
  robot's own X** (the minimum-reach intercept) and lets the cube come to it.
- **Match:** as the cube arrives it descends and aims slightly ahead
  (`cube + v_cube*TAU_CAPTURE + LEAD_X`, lead decaying to 0), so the EE
  **matches the cube's velocity** and the close happens as if the cube were
  standing still.
- Closes only after `REQUIRED_GOOD_FRAMES` of small XY error **and** small
  relative EE↔cube velocity.
- **Rate limiter (`MAX_JOINT_STEP`)** caps the per-frame joint change instead
  of rejecting big jumps, so the arm is smooth *and* never freezes/aborts.
- If the hover/intercept itself is unreachable, it aborts and tells you to move
  the robot closer to the conveyor — the one thing a controller can't fix.

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
  -> HOVER over the lane at the robot's X   (best-reach intercept)
  -> WAIT for cube to reach the approach zone
  -> APPROACH     (descend + track + match cube velocity)
  -> CAPTURE      (align in XY, low relative speed, near intercept X)
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
| `HOVER_HEIGHT` | how high the EE waits above the lane at the intercept | lower a bit if the descend is too rushed |
| `FINGER_MAX_FORCE` | grip force cap (N) | raise if the cube slips; lower if it gets ejected |
| `FINGER_DRIVE_STIFFNESS`/`DAMPING` | finger position-drive gains | raise stiffness if fingers don't hold their target |
| `APPROACH_START_OFFSET` | how far upstream the descend begins (relative to base X) | more negative = more time to align, but more reach |
| `TAU_CAPTURE` / `LEAD_X` | velocity-match lead near grasp | raise `TAU_CAPTURE` slightly if the cube slides through the close |
| `MAX_JOINT_STEP` | per-frame joint rate limit (anti-snap) | lower for gentler motion; raise if it lags the cube |
| `CMD_SMOOTH` | low-pass on the arm command (kills gripper jitter) | raise toward 0.7 if the wrist still buzzes; lower if it lags |
| `VEL_SMOOTH` | cube-velocity EMA weight for the feed-forward | raise if the feed-forward is noisy; lower if it lags a speed change |
| `FINGER_MAX_FORCE` | grip force cap (N) | lower to reduce object bounce on close; raise if it slips |
| `QUICK_LIFT_FRAMES`/`QUICK_LIFT_HEIGHT` | gentle straight-up lift off the belt | more frames = gentler (less inertia bounce) |
| `CAP_XY_TOL`, `CAP_RELV_TOL`, `REQUIRED_GOOD_FRAMES` | how strict the capture gate is | loosen if it hits the deadline; tighten if it closes off-center |
| `CLOSE_FRAMES` | close speed | fewer = faster close = less cube travel during close |
| `CUBE_RISE_GATE` | min rise to accept the grasp | — |
| paths + `READY_Q_DEG` | scene paths and READY branch | edit if your scene/paths differ |

**Reach note:** the pick is near the arm's outer workspace. If the log shows
`cannot reach the hover/intercept pose` or repeated IK failures during
approach, the fix is physical, not a parameter: move the robot ~5–10 cm closer
to the conveyor (or the conveyor toward the robot) so the cube lane sits well
inside reach, then re-run.

**Gripper is large; the cube must match it (`CUBE_TARGET_SIZE`, default 0.080 m).**
Computed from your URDF + Lula collision spheres: this CGE-10-10 is a big 3-jaw
gripper whose fingertips sit ~37 mm from the grasp center even fully closed, and
the fingers only travel ~7 mm radially. So it grips objects roughly **75–88 mm**
across (face-on) — a 20–25 mm cube closes through empty air and never touches
(every "no rise" run). The script:
- runs an **aperture calibration** at startup (`[3b]` in the log) that measures
  the real fingertip spread live and prints the graspable size range + whether
  your `CUBE_TARGET_SIZE` is inside it;
- resizes `/World/Cube` to `CUBE_TARGET_SIZE` (80 mm) so it's graspable;
- positions the grasp by the gripper's **grasp center** (≈19.5 mm toward the
  gripper from `pro_arm_ee`, measured live), placing that center at the cube
  center — so the grasp height is correct for any cube size or orientation, and
  the place puts the *cube* (not the EE) on the pedestal.

The tell after a close: `gripper after close` q7 **near 0** = cube still too
small (raise `CUBE_TARGET_SIZE` toward the calibrated range); q7 **stalled >
~2 mm** = gripped (if it still slips, raise `FINGER_MAX_FORCE`). Set
`CUBE_TARGET_SIZE = None` to keep your own cube and resize it yourself.

---

## E. One-run diagnostics to read in the console

- `GRIPPER SELF-TEST` — are `q7/q8/q9` symmetric on open **and** close?
- catch-up lines — `vcx` (cube speed) vs `vex` (EE speed): EE should exceed the
  cube during catch-up, then converge; `relv` should fall below `CAP_RELV_TOL`.
- `CAPTURE ALIGNED` — `exy`, `ez`, `relv` at the instant of closing.
- `gripper after close` — did the fingers close symmetrically on the cube?
- `cube rise after quick lift` — the single number that says grasp / no grasp.

---

## F. Motion quality — killing the bounce & the jitter

The pick succeeded but the object bounced and the gripper buzzed. Both came
from the same two root causes, now fixed:

**1. Gripper jitter = frame-to-frame IK wobble.** Position IK on a 6-DOF arm
has redundant DOF, so consecutive solves can pick slightly different joint
branches — the rate limiter caps the *size* of each step but not its
*direction*, so the wrist buzzes. Fix: a low-pass filter (`CMD_SMOOTH`) on the
commanded joints in every per-frame tracking loop (approach / settle / close /
hold / lift). The joint-space transfers already interpolate solved key-poses,
so they stay untouched. `MAX_JOINT_STEP` was also lowered (0.060 → 0.035).

**2. Object bounce = over-yank + hard contacts.** The quick-lift and the
transfer were straining the EE toward the conveyor bbox top (~2.3 m — the tall
belt superstructure, far past the arm's reach). The IK failed and the arm
snapped upward, launching the object (the 100 mm+ over-lift). Fixes:
- every "clear/carry" height is capped to a **known-reachable ceiling**
  (~`READY_HEIGHT` above the belt); the quick-lift is now a short, gentle
  straight-up pull off the belt (`QUICK_LIFT_HEIGHT`, no rail-clearance strain);
- **physics stabilization** — TGS solver + stabilization on the scene;
- **lower grip force** (`FINGER_MAX_FORCE` 40 → 28 N) and a smoothed velocity
  feed-forward (`VEL_SMOOTH`) so the close doesn't punch the object.

**3. The bounce also drifted the object OUT OF REACH and made the arm hit the
rail.** On the moving belt the light (50 g) cylinder was popping on spawn and
being spun by the belt's friction — its bbox read `40×35×33 mm` (it had tipped
onto its side) and it drifted ~25 mm further out in Y. That extra 25 mm pushed
the lane just past the arm's **top-down reach envelope**, so the top-down IK
search found nothing and the code fell back to the strongly-tilted READY
orientation — which leans the wrist straight onto the near blue rail (the crash
in the video). Fixes:
- **Object can't bounce or tumble any more.** On the belt it is now heavier
  (`OBJECT_MASS` 50 → 150 g) with a **capped depenetration velocity** (no spawn
  pop), **capped linear/angular velocity** (the belt can't fling or spin it) and
  heavy damping — so it rides the belt upright at its rest Y and stays in reach.
- **The arm never uses the rail-clipping orientation.** The grasp orientation is
  now restricted to **near-vertical only** (`|tilt| ≤ 20°`). If no such
  orientation is reachable at the cube, the script **aborts cleanly and stays
  safe** instead of lunging with a tilted wrist — the tilted READY fallback is
  gone. (The later "dynamic gripper height / retreat when the object is too
  large" pass builds on this safe-abort behaviour.)

What to watch on the next run: the object should stay upright (bbox ≈
`30×30×22 mm`) and NOT drift; `[4]` should print `TOP-DOWN grasp orientation
selected … near-vertical` (no `[WARN] no top-down IK solution`); the console
prints a `reach ceiling` and a `carry Z` line — both comfortably reachable (no
`[WARN] IK failed for lift-high`); and `cube rise after quick lift` should be
~80 mm, not 100 mm+. If the wrist still buzzes, raise `CMD_SMOOTH` toward 0.7;
if the object still hops on close, drop `FINGER_MAX_FORCE` a few N. If `[4]`
still aborts as unreachable, the lane is genuinely at the arm's limit — move the
robot a few cm toward the conveyor.
