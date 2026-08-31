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
| `CAP_XY_TOL`, `CAP_RELV_TOL`, `REQUIRED_GOOD_FRAMES` | how strict the capture gate is | loosen if it hits the deadline; tighten if it closes off-center |
| `CLOSE_FRAMES` | close speed | fewer = faster close = less cube travel during close |
| `CUBE_RISE_GATE` | min rise to accept the grasp | — |
| paths + `READY_Q_DEG` | scene paths and READY branch | edit if your scene/paths differ |

**Reach note:** the pick is near the arm's outer workspace. If the log shows
`cannot reach the hover/intercept pose` or repeated IK failures during
approach, the fix is physical, not a parameter: move the robot ~5–10 cm closer
to the conveyor (or the conveyor toward the robot) so the cube lane sits well
inside reach, then re-run.

**Cube size (`CUBE_TARGET_SIZE`, default 0.025 m):** a 20 mm cube is at/below
this gripper's fully-closed aperture — a perfectly-centered close reaches
q7≈0 with **no preload** (measured: `gripper after close ~0.7 mm`) and the cube
is never gripped. The script resizes `/World/Cube` to 25 mm at startup (while
stopped) so the fingers stall **on** the cube faces (q7 > 0 = real squeeze) and
prints the measured size to confirm. Set `CUBE_TARGET_SIZE = None` to leave your
cube alone and resize it yourself. The tell in the log after a close:
`gripper after close` q7 **near 0** = still too small (raise the size);
q7 **> ~2 mm** = gripped (if it still slips, raise `FINGER_MAX_FORCE`).

---

## E. One-run diagnostics to read in the console

- `GRIPPER SELF-TEST` — are `q7/q8/q9` symmetric on open **and** close?
- catch-up lines — `vcx` (cube speed) vs `vex` (EE speed): EE should exceed the
  cube during catch-up, then converge; `relv` should fall below `CAP_RELV_TOL`.
- `CAPTURE ALIGNED` — `exy`, `ez`, `relv` at the instant of closing.
- `gripper after close` — did the fingers close symmetrically on the cube?
- `cube rise after quick lift` — the single number that says grasp / no grasp.
