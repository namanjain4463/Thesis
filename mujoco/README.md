# Factorized Interaction World Model — MuJoCo science core

The thesis science core moved from Isaac Sim to **MuJoCo** for one reason: native
access to the **Delassus operator** `W = J_c M⁻¹ J_cᵀ` (contact-space inverse
inertia), the analytical global coupling the factorization is built on.

## The idea in one line

A contact force is the output of a **convex solve** whose data splits into a
**local, embodiment-free constitutive law** `C_θ(z_local)` and an **analytical,
embodiment-carrying global coupling** `W = J_c M⁻¹ J_cᵀ`. Learn only the local
law; compute `W` per embodiment; transfer is **certified**.

```
force  =  Solve(  W(embodiment)  ,  C_θ(z_local)  )         # z_local ⟂ embodiment
```

## Layout

Core pipeline
- `contact_probe.py`      — validated MuJoCo physics probe (assembles `W`, per-contact records, object motion). `W` matches `mj_solveM` to 0.0, PSD, ΣFn = weight.
- `z_local_schema.py`     — **frozen** 27-D per-contact local feature schema (`z_local.v1.1`), object/contact-frame only. `ContactTracker` accumulates slip/age. `W_ii` logged separately for the RQ1 ablation.
- `m2_floating_gripper_grasp.py` — embodiment #1: floating parallel gripper grasp generator + dataset builder hook.
- `panda_embodiment.py`   — embodiment #2: **real Franka Panda** (mujoco_menagerie) + DLS IK. Reuses `contact_probe` + `z_local_schema` **unchanged**.
- `build_zlocal_dataset.py` — randomized grasps → consolidated `z_local` training set.

Validation (run any of these; each prints its own verdict)
- `verify_solve.py`       — implicit gradient through the convex contact solve vs finite diff (err ~1e-12).
- `verify_math.py`        — port factorization invariance (V1), internal-force nullspace + proprioception (V2), transfer certificate + conditioning (V3).
- `verify_arch.py`        — robot-model-error leakage (2A) and the zero-velocity/compliance argument (2B).
- `validate_premises.py`  — P1/P2/P3 on **real** grasp trajectories.
- `premises_final.py`     — the consolidated **pre-embodiment GATE** (prints GO/NO-GO).
- `p2_frictionless.py`    — the exact contact-solve KKT in the clean frictionless case (`jar = R·f`, machine precision), confirming the local/global split against ground-truth physics.
- `panda_zlocal_check.py` — proves the Panda emits **schema-identical** `z_local` (0 NaN on primitive grasp contacts).
- `cross_embodiment_v2.py`— exact port split `W = Y_object + Y_robot`, `Y_object` embodiment-invariance (to 1e-15) at matched geometry, and the transfer certificate on the two real arms.
- `surface_field_covering.py` — **new-math headline**: drops the matched-geometry assumption. Models the contact law as a **field on the object surface** and the embodiment as a **sampling measure**; the transfer certificate becomes a **surface-covering** condition `error ≤ ε + C·L·dist(ξ*, trained support)`. Validated on real contact supports from both arms: error `∝ L` (slope/L = 1.389 ± 0.0% over a 10× range), collapses to one universal law (R²=0.95), and the **covering distance predicts cross-embodiment transfer error (228×)**. `probe_supports.py` extracts the `(θ,z)` material-coordinate supports.
- `hetero_covering.py` — closes the "learned vs analytical" gap: a cylinder with **height-varying friction** `μ(z)` baked into physics. The field is *learned from physics* — `μ_obs = |Ft|/Fn` measured at **sliding** contacts (2339 samples). Covering law holds on the learned material field: in-support error 0.02, out-of-support error grows at slope 10 vs the field's true `L=9`. `probe_hetero.py` confirms `μ_obs` tracks the segment friction (corr 0.90) at slip.
- `graspability.py` — **graspability = another certificate**. For a battery of shapes (sphere, box, cylinder, ellipsoid, wide box, weak-grip cases) computes interpretable margins — `γ_kin` (finger span − object half-width), `γ_fric = μ − μ_required` with `μ_required = mg/ΣFn` — and checks each against the real lift outcome. **100% accurate (8/8)**; every rejection is a *named margin going negative* (WIDE box → kinematic; slick/heavy weak-grip → friction cone).

Rendering / analysis
- `panda_render.py`, `render_mujoco.py` — OSMesa 3D renders. `render_filmstrip.py` — side-view filmstrip. `analyze_batch.py` — batch outcome figure.

## What is validated (all reproducible here)

| Check | Result |
|---|---|
| **P1** coupling identity `M(q̈−q̈_smooth)=Jᵀf` ⇔ `efc_J(q̈−q̈_s)=W·efc_force` | residual ~1e-16 (exact) |
| **P2** local/global split — frictionless KKT `jar = R·f` from `W`,`R`,`aref` | resid ~1e-15; weight exact |
| P2 corollary: contact force is **not** a local function of kinematics | `Fn~[pen,vn]` R²≈0.06 (expected — that is why the `W`-solve is needed) |
| **P3** robot-model-error budget (structured ΔM → contact-force contamination) | ~linear: 2%→1.5%, 5%→4%, 10%→8% |
| z_local **schema identity** across the two embodiments | 27-D, 0 NaN on primitive grasp contacts; `κ_obj = 1/r` exact |
| Exact **port split** `W = Y_object + Y_robot` (object/robot dofs inertially decoupled) | ‖W−(Yo+Yr)‖/‖W‖ ~1e-17 |
| `Y_object` **embodiment-invariance** at matched object-frame contact | `Y_object = 1/m = 20.0` for both arms, Δ ~1e-15 |
| Transfer **certificate** `‖ΔH‖ ≤ ε/(m(m−ε))` on the two real arms | holds; `Y_robot` = 8.98 (Panda) vs 33.33 (floating) |

Known TODO flagged in the code: **mesh-geom curvature** in `z_local` is NaN by
design (needs a mesh curvature estimator); primitive grasp contacts are clean.

## Running

```bash
pip install mujoco numpy matplotlib
# Panda embodiment needs mujoco_menagerie:
git clone https://github.com/google-deepmind/mujoco_menagerie
export MENAGERIE_DIR=$PWD/mujoco_menagerie      # or FRANKA_DIR=<.../franka_emika_panda>
export MUJOCO_GL=osmesa                          # headless rendering only

python premises_final.py        # the GATE
python cross_embodiment_v2.py   # the cross-embodiment certificate
python panda_render.py          # 3D grasp filmstrip -> panda_grasp_strip.png
```
