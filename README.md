# Factorized Interaction World Model — Master Document (PRD + Research Log)

> **Purpose of this file.** This is the single entry point for anyone — human or agent —
> picking up this thesis. It states the problem, the mathematics we derived, what we
> built and validated (with numbers), every major decision and *why* we made it, the
> open questions we are investigating, and the roadmap. If you are an agent starting
> fresh: read this top-to-bottom, then read `CLAUDE.md` (operating rules), then
> `mujoco/README.md` (how to run each experiment). Do not start from scratch.

---

## 0. Thirty-second orientation (for a new agent)

- **Thesis in one line:** contact-skill transfer across different robot bodies is a
  *certifiable* operation. Freeze a local contact law `C_θ`, recompute each robot's
  analytical port `Y_G`, and a set of **computable margins** decides — before acting —
  whether a skill transfers, needs a probe, or must be rejected, and *names the reason*.
- **Where the science lives:** `mujoco/` (MuJoCo 3.12.0). Isaac Sim harness in `isaac/`.
- **Hard rules:** only touch the `namanjain4463/Thesis` repo; develop on branch
  `claude/lynxmotion-cube-picker-controller-9umls9`; commit + push as work lands
  (containers are ephemeral); Isaac cylinder mass ≤ 0.1 kg. (Full rules in `CLAUDE.md`.)
- **Status:** all core math validated *within simulator* against MuJoCo ground truth.
  The certificate now carries both hardware error sources (`ε_Y`, `ε_C`). Next build:
  a genuine sim-to-real stress test (see §10).

---

## 1. Problem statement & motivation

A robot skill learned on one arm does not transfer to another: a policy trained on a
Franka does not run on a Lynxmotion, a bimanual rig, or a humanoid, because the *body*
is baked into what was learned. The field's usual answers — retrain per robot, or learn
a giant embodiment-conditioned foundation model — are expensive and, crucially,
**uncertified**: they cannot tell you *in advance* whether a given skill will transfer to
a given body on a given object, or *why* it will fail.

The target hardware is a **Lynxmotion SES-Pro 550 mm 6-DOF arm + 3-finger gripper** with
a **D455 camera** (no contact-force sensing), deployed on a **Jetson Orin Nano 8 GB**. This
hardware *cannot observe internal grasp forces*. A defensible system must therefore be
**sensing-aware**: it must know what it cannot know and reject tasks that require it.

**The scientific question:** can contact-skill transfer be made *certified and
sensing-aware* — reduced to computable margins that predict success, failure, and cause?

---

## 2. Core idea — the factorization

A contact force is the output of a **convex solve** whose data splits into:

- a **local, embodiment-free constitutive law** `C_θ(z_local)` (the physics *at* the contact:
  stiffness, damping, friction, restitution — a function of a local, contact/object-frame
  feature vector `z_local`), and
- an **analytical, embodiment-carrying global coupling** `Y_G` (how the robot + object +
  contact graph respond), computed per embodiment, never learned.

$$
v = v^0 - Y_G[f], \qquad f \in C_\theta[v,h,z], \qquad
H_G = \left(Y_G + C_\theta^{-1}\right)^{-1}
$$

Changing the robot (Lynxmotion → Franka → humanoid) changes **only** `Y_G`; the learned
local law `C_θ` is frozen. This is the whole bet, and every result below is a test of it.

**Where "world model" enters:** the analytical global solve *is* the world model. We do not
learn a video/foundation world model; we learn the *local constitutive law* and compose it
with an analytical, differentiable global dynamics operator. This is a **factorized**
interaction world model — learned-local ⊕ analytical-global.

---

## 3. Mathematical formulation (everything we derived)

Notation: `M` = system mass matrix, `J_c` = contact Jacobian, `q̈` = `qacc`,
`q̈_s` = `qacc_smooth` (constraint-free acceleration), `f` = contact impulse/force in the
constraint (`efc`) basis, `n̂` = contact normal, `r` = contact point relative to object COM.

### 3.1 The Delassus operator (global coupling)

$$
W \;=\; J_c\, M^{-1} J_c^{\top} \quad\in\mathbb{R}^{n_{efc}\times n_{efc}}
$$

the **contact-space inverse inertia** (maps contact impulse → contact-space velocity
change). It is the discrete, single-port special case of the continuous multiport
admittance `Y_G`. **Validated:** `W` from dense `inv(M)` equals `W` from `mj_solveM` to
`0.0`; symmetric PSD; per-contact normal forces sum to object weight.

### 3.2 The coupling identity (the load-bearing fact — "P1")

Newton's law with contact forces, verified on real grasp trajectories to `~1e-16`:

$$
M\,(q̈ - q̈_s) = J_c^{\top} f \quad\Longleftrightarrow\quad J_c\,(q̈-q̈_s) = W f
$$

The embodiment enters the object's equations of motion **only** through `W` (hence `M`)
and `J_c^{\top}`. This is what licenses "recompute `Y_G`, freeze `C_θ`."

### 3.3 Exact port split (embodiment isolation)

The free object's DOFs and the robot's DOFs have **no inertial coupling** in `M` (they
couple only through contact), so `M^{-1}` is block-diagonal and the Delassus splits
**exactly** (validated to `~1e-17`):

$$
W \;=\; \underbrace{J_{obj} M_{obj}^{-1} J_{obj}^{\top}}_{Y_{object}\ (\text{embodiment-invariant})}
\;+\; \underbrace{J_{rob} M_{rob}^{-1} J_{rob}^{\top}}_{Y_{robot}\ (\text{embodiment})}
$$

- `Y_object` depends only on object inertia + contact geometry (all in `z_local`). Its
  per-contact normal admittance is analytic:

$$
Y_{object}^{nn} = \tfrac{1}{m} + (r\times n̂)^{\top} I_{obj}^{-1} (r\times n̂)
$$

  At a mid-height side contact (`r×n̂ = 0`) this is `1/m`. **Validated:** `= 20.0` for the
  *same object* under **both** the floating gripper and the Panda, to `3.5e-15` — i.e.
  embodiment-invariant at matched geometry.
- `Y_robot` is the arm's reflected inverse inertia (the port). Measured `8.98` (Panda) vs
  `33.33` (floating) at the same contact — this is where the embodiment lives.

### 3.4 The convex contact solve (constitutive law ↔ force)

MuJoCo's contact force is the stationary point of a strictly-convex program. In the clean
frictionless case we verified the exact KKT to `3.2e-14`:

$$
(R - W)\,f \;=\; J_c\,q̈_s + a_{ref}, \qquad \text{equivalently}\quad \underbrace{J_c q̈ + a_{ref}}_{\text{jar}} = R f
$$

where `R = diag(efc_R)` is the **local compliance** and `a_{ref} = efc_aref` the **local
reference** — both per-contact functions of `{penetration, v_n, solref, solimp}`, i.e. of
`z_local`. Our thesis solve uses the strictly-convex SAP-style form
`λ* = argmin_{λ∈FC} ½λᵀ(W+R)λ + qᵀλ` with `A = W+R ≻ 0` (unique). **The force is not a
local function of kinematics** — it is the solve's output; that non-locality is *why* the
analytical `W`-solve is needed. We verified the **implicit gradient** through this solve
vs finite differences to `1e-12` (so `C_θ` is trainable end-to-end).

### 3.5 Composed interface response & the port quotient

$$
H_e = \left(Y_e + C^{-1}\right)^{-1}, \qquad H_e^{-1} - Y_e = C^{-1}\ \text{(embodiment-independent)}
$$

Verified in the frequency domain across two embodiments to `1e-17`: raw response `|H_e|`
differs wildly across bodies, but `H_e^{-1} - Y_e` recovers the **same** interface law.
This is the falsifiable pre-training invariance.

### 3.6 Transfer certificate (now two-source)

Perturbing the interface `A = Y_G + C^{-1}` (spectral norm) gives a computable bound:

$$
\|\hat H - H\| \;\le\; \frac{\varepsilon_Y + \varepsilon_C}{m\,(m - (\varepsilon_Y+\varepsilon_C))},
\qquad m = \sigma_{\min}(Y_G + C^{-1})
$$

- `ε_C` = contact-law error (learning / sim-to-real of `C_θ`).
- `ε_Y` = port-identification error (how well `Y_G` is known on hardware).
- `m` = interface conditioning (a stiff, well-conditioned grasp tolerates more error; a
  soft/near-singular one amplifies it).

**Validated:** same `ε` gives ~20–200× larger outcome error on an ill-conditioned
embodiment than a well-conditioned one; the two error sources **add** inside one bound.

### 3.7 Internal-force nullspace (sensing limit)

For a multi-contact grasp with grasp map `G`:

$$
f = G^{\dagger} w_o + N_G\,\xi
$$

Object motion reveals only `G f`; the internal squeeze `N_G ξ` lives in `ker G` and is
**invisible to object-motion (camera-only) observation**. Verified: object-motion
supervision is genuinely blind to internal force. Consequence: the Lynxmotion + D455
**cannot** claim exact internal-force regulation — the certificate must *reject*
force-critical tasks, not fail silently. (Self-correction: an earlier claim that the
convex solve "resolves" this was wrong — forward-uniqueness ≠ training-identifiability.)

### 3.8 Observation sufficiency / task quotient (contribution B)

Transfer to a task `τ` (projection `P_τ`) with observation operator `O` is admissible iff

$$
\ker O \subseteq \ker P_\tau, \qquad
\beta_\tau = \min_L \big[\, R\,\|P_\tau - L O\| + \varepsilon_y \|L\| \,\big]
$$

small `β_τ` → sensors suffice (compile); moderate → probe; large → reject. This is a
computable rank/defect condition, not a heuristic. (Self-correction: underactuation is
*not* automatically unidentifiable — identifiability = injectivity of the *stacked*
observation operator `[G; J_1; J_2; …]`, a computable rank test.)

### 3.9 Surface-field reformulation & the covering law (the new-math headline)

The earlier invariance proof *assumed matched contact geometry*. Real bodies make
**different contact sets** (the Panda made 16 finger-object contacts, the floating gripper
9, for the "same" grasp). So we reframed:

> The contact law is a **field** `g(ξ)` on the object's material surface `ξ=(θ,z)`; an
> embodiment is a **sampling measure** `μ_E` on that surface. The dynamics become an
> integral equation; discretizing `μ_E` recovers the finite `(Y+C^{-1})^{-1}`.

Transfer from body A (support `S_A`) to body B then obeys a **surface-covering** bound:

$$
|\hat g(\xi^*) - g(\xi^*)| \;\lesssim\; \varepsilon_{learn} + C\cdot L\cdot \mathrm{dist}(\xi^*, S_A)
$$

`L` = field Lipschitz constant, `dist` = geodesic fill-distance on the surface, `C` = a
**field-independent geometry constant** (the sampling Lebesgue constant). Matched geometry
(`dist=0`) recovers the exact case. **Validated on real supports from both arms:** error
`∝ L` (`slope/L = 1.389 ± 0.0%` over a 10× range of `L`), `err/L` collapses onto one
universal law (`R²=0.95`), and **covering distance predicts cross-embodiment transfer error
(228× worse when body B contacts a surface band body A never trained on)**. Re-validated on
a *genuinely learned* material field `μ(z)` measured from sliding physics.

### 3.10 Graspability as a certificate

Graspability uses the *same* friction-cone / force-closure operators. Interpretable margins:

$$
\gamma_{kin} = \text{finger span} - \text{object half-width}, \qquad
\gamma_{fric} = \mu - \mu_{req}, \quad \mu_{req} = \frac{m g}{\sum_i F_n^{\,i}}
$$

Graspable iff all margins positive; the negative margin **names the reason**. **Validated:
100% (8/8)** over a shape battery (cylinder, sphere, box, ellipsoid, wide box, weak-grip
cases) — wide box → kinematic; slick/heavy weak-grip → friction cone.

### 3.11 Free-space separation principle (port identification)

Because the contact force vanishes in free space (`df ≡ 0`), exciting the endpoint there
identifies `Y_robot` **uncontaminated by `C_θ`**, making the blind Y/C deconvolution
well-posed. Two facts:

- From contact `H = (Y+C^{-1})^{-1}` **alone**, `(Y,C)` is a **gauge family** (any `Y'` fits
  with a compensating `C'` — verified: `‖H'−H‖ ~ 3e-19`). Free-space `Y` makes `C` unique.
- The naive rigid-body port `J M^{-1} J^{\top}` can be badly wrong: for the Panda it is
  **94% off** because the finger **tendon** is a real structural coupling. Free-space ID
  recovers the true (constrained) port. **You identify `Y`, you do not assume it.**
  `ε_Y` scales ~linearly with endpoint-sensor noise.

---

## 4. Contributions & novelty

- **(A) A computable transfer certificate** `‖Ĥ−H‖ ≤ (ε_Y+ε_C)/(m(m−ε))` with
  `m = σ_min(Y_G+C^{-1})`. *Lit-checked as genuinely novel* — prior port-Hamiltonian
  learning (van der Schaft; Neary–Topcu arXiv:2412.11215) and NeRD (arXiv:2508.15755) do
  not give a computable, embodiment-transfer error certificate.
- **(B) An observation-nullspace injectivity theorem** for internal forces + a
  proprioception-resolution condition (`ker O ⊆ ker P_τ`, defect `β_τ`). The facts are
  classical grasp mechanics; the *learning-theoretic packaging as a transfer admissibility
  test* is novel.
- **(Quotient) Admittance-quotient invariance** `H_e^{-1} - Y_e = C^{-1}`: the falsifiable,
  pre-training embodiment quotient. Novel in specifics; must be distinguished from
  compositional port-Hamiltonian learning in the writeup.
- **(New this line of work) The surface-covering law**: recasting embodiment transfer as
  scattered-data approximation on the object surface, with a certificate computable from
  contact geometry alone. This dissolves the matched-geometry assumption and handles
  different finger counts / patch shapes / bimanual / humanoid as different measures on one
  field.

**Falsifiable prediction (RQ1):** `C_θ` should **not** need the self-inertia block `W_ii`
(the constitutive law is inertia-independent). `W_ii` is logged separately from the strict
27-D `z_local` so this can be tested by ablation without regenerating data.

---

## 5. What we built & validated (with reasoning)

| # | Artifact | What it establishes | Key numbers |
|---|---|---|---|
| 1 | `contact_probe.py` | `W` assembly = ground truth | `W` vs `mj_solveM` = 0.0; ΣFn = weight |
| 2 | `z_local_schema.py` | frozen 27-D embodiment-agnostic feature schema | `κ_obj = 1/r` exact |
| 3 | `premises_final.py` | pre-embodiment GATE (P1/P2/P3) | P1 `1e-16`, P2 frictionless KKT `1e-15`, P3 contamination ~linear |
| 4 | `verify_solve.py` | implicit gradient through the solve | err `1e-12` |
| 5 | `verify_math.py` | quotient invariance / nullspace / certificate | `1e-17` / rank test / conditioning 210× |
| 6 | `panda_embodiment.py` | 2nd real embodiment (Franka) grasps + lifts | schema-identical `z_local`, 0 NaN on pads |
| 7 | `cross_embodiment_v2.py` | exact port split + `Y_object` invariance + certificate | split `1e-17`, `Y_object=20.0` Δ`3.5e-15` |
| 8 | `surface_field_covering.py` | **covering law** on real supports | slope/L `1.389±0%`, R² `0.95`, cross-embodiment `228×` |
| 9 | `hetero_covering.py` | covering law on a **learned** material field | in-support `0.02`, slope `10` vs L `9` |
| 10 | `graspability.py` | graspability = named margins | **100% (8/8)** |
| 11 | `port_identification.py` | free-space `ε_Y` + two-source certificate | Panda naive port 94% wrong; bound holds |

**Why this order.** We validated the *load-bearing identity* (P1) before anything built on
it; got *one* embodiment fully working and visually verified (rendered filmstrips) before
adding a second; matched geometry first (clean invariance) before removing that assumption
(covering law); and only then added the hardware-honesty layer (`ε_Y`). Each step is a
falsification opportunity, not a demo.

**Outputs (figures + raw numbers): [`outputs/RESULTS.md`](outputs/RESULTS.md).** Every
figure and the verbatim console output of each experiment is collected there; regenerate
any of them by running the matching script in `mujoco/`
(`MUJOCO_GL=osmesa`, `MENAGERIE_DIR` set for the Panda).

---

## 6. Decision log (what we decided and why)

- **Simulator = MuJoCo (not Isaac) for the science core.** MuJoCo exposes the Delassus
  operator natively (`mj_solveM`, `efc_J`, `efc_R/aref`), which is the object the whole
  factorization is built on. Isaac is kept for the arm/pick-place harness.
- **`C_θ` outputs physical constitutive params `(k_n, d_n, μ, e, uncertainty)`**, not an
  abstract latent `η`. Physical params keep the solve strictly convex (`k_n>0`) and make
  the law interpretable and calibratable against reality.
- **Supervision = per-contact force + object motion**, not object motion alone (object
  motion is blind to internal force — §3.7).
- **Architecture = one runtime stack, not a network of "AI agents."** We rejected the
  agent-network / LLM / graph-planner framing: multi-arm coordination is already carried
  by the multiport math `Y_G = Bᵀ blkdiag(Y_1,…,Y_n,Y_obj) B`; adding agents adds
  consistency problems and no capability. Deployment is a thin deterministic adapter per
  robot (a mechanical port), a frozen `C_θ`, an analytical compiler, a convex solver, and a
  certificate/supervisor (compile / probe / reject).
- **Certificate carries two explicit error sources (`ε_Y`, `ε_C`).** A single-ε certificate
  hid the hardware identification problem; the honest bound budgets both.
- **`Y_G` is identified in free space, not assumed.** The Panda tendon (94% port error)
  proved the naive analytical port is unsafe.

---

## 7. Reasoning / design rationale (why I did what I did)

- **Why a factorization at all:** the only embodiment-invariant object in contact dynamics
  is the *local constitutive law*; everything body-specific funnels through `M`/`J`. So the
  maximally-transferable thing to learn is the local law, and the maximally-reusable thing
  to compute is `Y_G`. The factorization is the unique split that respects this.
- **Why the surface-field reframe:** our own clean results *cheated* by matching geometry.
  The honest object is not a fixed operator but a field sampled by whatever body shows up.
  This is the "out-of-the-box" move that makes the certificate computable from contact
  geometry and extends to bimanual/humanoid without new machinery.
- **Why free-space identification:** it is the *only* place `Y` is observable without `C`
  contaminating it (the gauge argument, §3.11). It also matches how you would calibrate a
  real arm: wave it in open space before touching anything.
- **Why report everything with named error terms:** the thesis is "certified" transfer. A
  claim without a computable, budgeted error term is a demo, not a certificate.

---

## 8. `z_local` schema (frozen contract)

`z_local.v1.1`, 27-D, object/contact-frame only (never joint/world): gap(1), contact point
in object frame(3), normal in object frame(3), `v_n`(1), `v_t`(2), spin `w_n`(1), object
curvature(2), other-geom curvature(2), friction(3), `solref`(2), `solimp`(5), slip_accum(1),
contact_age(1). The self-inertia block `W_ii`(9) is logged **separately** (for the RQ1
ablation). Mesh-geom curvature is NaN by design (TODO: a mesh curvature estimator);
primitive grasp contacts are clean.

---

## 9. Deployment architecture (target: Jetson Orin Nano 8 GB)

Offline (workstation): parallel data generation, train `C_θ`, certificate stress-tests,
export frozen weights. Online (Jetson): D455 perception → contact/object estimator →
`z_local` → task quotient (`β_τ`) → frozen `C_θ` → multiport compiler assembles `Y_G` →
robust convex solve → certificate `Γ = min{γ_kin, γ_fric, γ_cond, γ_obs, γ_cover}` →
compile / probe / reject → thin per-robot adapter → vendor controller. A new robot needs
only: an adapter, a free-space `Y_G` identification (with `ε_Y`), and its observation
operator `O` — **`C_θ` is never retrained**. Plug-and-**certify**, not plug-and-play.

---

## 10. Open problems / what we are investigating & why

- **Sim-to-real of `C_θ` (`ε_C`) — the next build.** Everything so far is *within-simulator*
  (errors measured against MuJoCo's own ground truth). Plan: learn a law/port under one
  contact model, evaluate under a **deliberately mismatched** contact model (different
  solver / stiffness / friction regularization) standing in for reality, and confirm the
  certificate's `ε_C` predicts the degradation. This is what turns "self-consistent" into
  "tracks reality."
- **Closed-loop, controller-shaped `Y_G`.** The real Lynxmotion port is dominated by its
  position servo + latency, not rigid-body inertia. Identify the *closed-loop* port in free
  space; fold its `ε_Y`.
- **Mesh curvature estimator** to remove the `z_local` NaN on non-primitive geoms.
- **Probe design for path-observability of internal force** — can a slip-modulating probe
  make `N_G ξ` partially observable up to a certified residual, turning some "reject" cases
  into "probe"?
- **Bimanual + humanoid** as larger `Y_G` graphs feeding compiled contact objectives to an
  existing whole-body controller — same frozen `C_θ`.

---

## 11. Roadmap

1. **Now:** sim-to-real stress test (`ε_C` predicts mismatched-model degradation).
2. **Next:** closed-loop `Y_G` identification + Lynxmotion adapter; mesh curvature estimator.
3. **Then:** train the actual `C_θ` network on the `z_local` dataset; first *learned*
   cross-embodiment transfer (empirical certificate with a real learned law).
4. **Then:** bimanual coalition selection + probe-design theorem for internal force.
5. **Deploy:** Jetson runtime, shadow-mode, low-speed execution with live certificate gating.

---

## 12. Repository layout

```
CLAUDE.md                     operating rules + preferences (agents read this)
README.md                     ← this master document
mujoco/                       science core (see mujoco/README.md for run instructions)
  contact_probe.py            validated W / Delassus probe
  z_local_schema.py           frozen 27-D feature schema
  m2_floating_gripper_grasp.py  embodiment #1 (floating gripper)
  panda_embodiment.py         embodiment #2 (Franka, needs $MENAGERIE_DIR)
  premises_final.py           pre-embodiment GATE
  cross_embodiment_v2.py      port split + Y_object invariance + certificate
  surface_field_covering.py   covering law (headline)
  hetero_covering.py          covering law on a learned material field
  graspability.py             graspability certificate (shape battery)
  port_identification.py      free-space eps_Y + two-source certificate
  verify_solve.py / verify_math.py / verify_arch.py   math checks
isaac/                        Isaac Sim arm + pick-place harness
```

Runtime notes: MuJoCo 3.12.0; headless render `MUJOCO_GL=osmesa`; Panda needs
`export MENAGERIE_DIR=<mujoco_menagerie checkout>` (menagerie is gitignored).

---

## 13. Notation & glossary

`W` Delassus / contact-space inverse inertia · `Y_G` global port admittance ·
`Y_object`/`Y_robot` object/robot port blocks · `C_θ` learned local contact law ·
`z_local` frozen per-contact feature vector · `H` composed interface response ·
`G` grasp map, `N_G` its nullspace (internal forces) · `μ_E` embodiment sampling measure ·
`L` field Lipschitz constant · `C` sampling geometry (Lebesgue) constant ·
`ε_Y` port-ID error · `ε_C` contact-law error · `m = σ_min(Y_G+C^{-1})` interface conditioning ·
`β_τ` observation-sufficiency defect · `Γ` composite deployment certificate.

---

## 14. Honest limitations (state these plainly)

1. All validation is **within-simulator**; sim-to-real (`ε_C`) is not yet tested against a
   mismatched model.
2. `Y_G` is currently the rigid-body/constrained port; the **closed-loop controller-shaped**
   port on real hardware is not yet identified.
3. `C_θ` is **not yet trained** — the covering law and certificate are validated on the
   analytical/measured fields and controlled stand-ins, not a learned network (that is a
   roadmap item, and the pipeline + implicit gradient are ready for it).
4. The **camera-only Lynxmotion cannot observe internal force** — this is a fundamental
   limit the certificate must respect by rejecting force-critical tasks.
