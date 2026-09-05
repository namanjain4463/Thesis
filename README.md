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
- **Status (within-simulator):** the load-bearing identities hold — P1 coupling (`1e-16`), the
  corrected P2 KKT (settled *and* transient), the exact port split (`1e-17`), the covering
  *geometry* and the *physical*-field bound (0%, gated). Honest boundaries: the covering law on
  a *learned* field does **not** yet hold (RBF-KRR extrapolation), `C_θ` is **not trained**, and
  `Y_G` is the instantaneous (not closed-loop) port. Read **§14** before quoting any result.
  Next build: a genuine sim-to-real stress test (see §10).

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

  At a mid-height side contact (`r×n̂ = 0`) this is `1/m`. **Confirmed** `= 20.0` for the
  *same object* under **both** the floating gripper and the Panda, to `3.5e-15`. Note this
  invariance is **exact by construction** — `Y_object` is computed from the object body's own
  Jacobian, which carries no arm DOFs — so the numeric match confirms the *implementation*, not
  a learned transfer. The physically substantive fact is the exact **split** below (MuJoCo's
  real `M` is block-diagonal between object and arm DOFs to `~1e-17`), which is a genuine
  numerical identity, not a tautology.
- `Y_robot` is the arm's reflected inverse inertia (the port). Measured `8.98` (Panda) vs
  `33.33` (floating) at the same contact — this is where the embodiment lives.

### 3.4 The convex contact solve (constitutive law ↔ force)

MuJoCo's contact force is the stationary point of a strictly-convex program. The exact
active-constraint KKT stationarity in the clean frictionless case is

$$
(W + R)\,f \;=\; a_{ref} - J_c\,q̈_s, \qquad \text{equivalently}\quad R f = a_{ref} - J_c q̈
$$

(`a_u := J_c q̈_s` is the unconstrained contact acceleration.) Verified to machine precision at
a **settled** *and* a **transient (accelerating-contact)** state — settled `1.6e-15`, transient
`1.8e-16`. The transient is essential: the wrong-sign form `(R−W)f = a_u+a_ref` has residual
exactly `−2·a_c` (`a_c = J_c q̈`), so it passes for *any* sign at a settled state where `a_c≈0`.
An earlier version wrote that wrong sign and tested only the settled box, so P2 passed
vacuously; the corrected test falsifies the sign (old sign → transient residual `5e-2`). Here
`R = diag(efc_R)` is the **local compliance** and `a_{ref} = efc_aref` the **local reference** —
both per-contact functions of `{penetration, v_n, solref, solimp}`, i.e. of `z_local`. Our thesis solve uses the strictly-convex SAP-style form
`λ* = argmin_{λ∈FC} ½λᵀ(W+R)λ + qᵀλ` with `A = W+R ≻ 0` (unique). **The force is not a
local function of kinematics** — it is the solve's output; that non-locality is *why* the
analytical `W`-solve is needed. We verified the **implicit gradient** through this solve
vs finite differences to `1e-12` (so `C_θ` is trainable end-to-end).

### 3.5 Composed interface response & the port quotient

$$
H_e = \left(Y_e + C^{-1}\right)^{-1}, \qquad H_e^{-1} - Y_e = C^{-1}\ \text{(embodiment-independent)}
$$

In the frequency-domain check (`verify_math.py` V1) this holds to `1e-17` — but that check
**constructs** `H_e = (Y_e + Y_c)^{-1}` and then recovers `Y_c`, so the `1e-17` is an *exact
algebraic identity / implementation check*, not independent evidence. The **falsifiable**
content is the same quotient applied to `H` and `Y` that are *independently* obtained: the
exact port split (§3.3) and the free-space port ID (§3.11). Stated plainly: the algebra is
correct; the empirical claim rests on the split and the ID, not on this synthetic recovery.

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

This is the standard resolvent perturbation inequality (a genuine theorem, not fitted). It is
demonstrated on synthetic SPD interfaces (`verify_math.py` V3, `port_identification.py` Part D)
and on the two identified real-arm ports (`cross_embodiment_v2.py`) — where the contact term is
now composed in consistent accelerance units `C^{-1}=1/(k h²)` (an earlier version added a
static compliance `1/k` to an inverse inertia, a dimensional error that made the contact term
numerically negligible). Same `ε` gives ~20–200× larger outcome error on an ill-conditioned
embodiment than a well-conditioned one; the two error sources **add** inside one bound. Note the
bound is verified on *synthetic and analytically-identified* ports, not yet on a *learned* `C_θ`
or a hardware-identified `Y_G`.

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
(`dist=0`) recovers the exact case. What is and is not established (on real supports from both
arms):

- **Bound holds when stated correctly.** The physical field `Y_object(z)` obeys
  `|ĝ−g| ≤ ε_learn + C·L·dist` with **0% violation** (`C=1.389` from the controlled family,
  `L` analytic, `ε_learn` the in-support ceiling — all fixed independently of the physical
  field). An earlier version tested it with `C=1` and an under-set `ε`, spuriously reporting
  "6.2% violated" while the verdict ignored it; the verdict now **gates** on the bound.
- **The `slope/L = 1.389 ± 0.0%` "collapse across L" is not independent evidence.** For a
  *linear* learner (KRR) on a *linearly*-scaled field `g=L·z`, error scales exactly `∝ L` by
  construction — hence the `±0.0%`. The substantive axis is the dependence on covering
  **distance** and the field-independent constant `C`, not the `L`-scaling.
- **Covering distance predicts cross-embodiment transfer (directionally):** `228×` more error
  when the Panda contacts a surface band the floating gripper never trained on.
- **On a genuinely *learned* RBF friction field `μ(z)`, the bound does NOT hold** (open item).
  With the ground truth corrected (see §14), RBF-KRR **mean-reverts under extrapolation**
  (predicting `μ̂=−0.27`, negative friction, past the training band) and violates
  `ε+C·L·dist` on ~100% of out-of-support points. The covering *geometry* is sound (a
  Lipschitz-consistent estimator obeys the bound at all distances, `0%`), so this is a
  **learner** limitation: demonstrating learned-field covering needs a Lipschitz-respecting
  estimator, not RBF-KRR extrapolation.

### 3.10 Graspability as a certificate

Graspability uses the *same* friction-cone / force-closure operators. Interpretable margins:

$$
\gamma_{kin} = \text{finger span} - \text{object half-width}, \qquad
\gamma_{fric} = \mu - \mu_{req}, \quad \mu_{req} = \frac{m g}{\sum_i F_n^{\,i}}
$$

Graspable iff all margins positive; the negative margin **names the reason**. Agrees with the
actual lift on **8/8** curated cases (cylinder, sphere, box, ellipsoid, wide box, weak-grip) —
wide box → kinematic; slick/heavy weak-grip → friction cone. **Caveat (scope):** `μ_req` uses
the *measured post-contact* normal force `ΣF_n`, so this is a **consistency check** evaluated
*after* the grasp is realized, not an *a-priori* screen. A deployable gate must instead
*predict* `ΣF_n` from the squeeze command + contact model; and 8 hand-picked cases is a spanning
demonstration, not a statistical accuracy.

### 3.11 Free-space separation principle (port identification)

Because the contact force vanishes in free space (`df ≡ 0`), exciting the endpoint there
identifies `Y_robot` **uncontaminated by `C_θ`**, making the blind Y/C deconvolution
well-posed. Two facts:

- From contact `H = (Y+C^{-1})^{-1}` **alone**, `(Y,C)` is a **gauge family** (any `Y'` fits
  with a compensating `C'` — `‖H'−H‖ ~ 3e-19`). Free-space `Y` makes `C` unique. *(This is a
  synthetic linear-algebra demonstration of the identifiability principle, not a data result.)*
- The naive rigid-body port `J M^{-1} J^{\top}` can be badly wrong: for the Panda it is
  **94% off** because the finger **tendon** is a real structural coupling. Free-space ID
  recovers the true (constrained) port. **You identify `Y`, you do not assume it.**
  `ε_Y` scales ~linearly with endpoint-sensor noise.

**Caveat (what is identified).** The identified port is the **instantaneous, open-loop**
mobility: a *known* generalized force is applied and `mj_forward`'s instantaneous `q̈` is read,
so the "ID vs true port" match (`2.6e-17`) is the *same* forward-dynamics computation and is
near-tautological — the informative result is the 94% tendon gap. On hardware you get neither a
known pure endpoint wrench (you need an instrumented/known-wrench setup) nor the instantaneous
rigid port: the deployable `Y_G` is the **closed-loop** endpoint admittance (servo + latency),
which is **not yet identified** (§10, §14).

### 3.12 First *learned* cross-embodiment transfer (de-leaked C_θ)

The results above use analytical/measured fields. Here `C_θ` is trained on a **de-leaked** dataset:
the frozen `z_local` fed the true `μ/solref/solimp` straight in (leakage). Instead the only material
handle is a **categorical material id** (you know which object you picked up), never the physical
parameters; every other input is observable (`penetration, v_n, |v_t|, κ_obj, contact height,
normal alignment`, and — for the port models — the analytical `W_nn`). Models are trained **only on
the floating gripper**, **frozen**, and evaluated on **held-out Panda** trials, on a **matched object
distribution** with **synchronized logging** (`deleak_dataset.py`, `deleak_train_eval.py`). Six
models on the identical held-out set: `A` local-only MLP, `B` factorized MLP (`+W_nn`), `C` white-box
coupled series-compliance (`+W_nn`), `D` white-box analytical (`F_n = pen/k_mat`, no port), a mean
baseline, and a Panda-retrained MLP reference.

> **This section was corrected after a second audit.** An earlier version reported that adding the
> analytical port "roughly doubles" frozen transfer (`R² 0.24→0.40`). That result **did not survive**
> three fixes the audit identified: (i) the Panda logger combined post-step velocity with pre-step
> contact Jacobians (a **median 20% error** in `v_n`); (ii) float and Panda used **different object
> distributions**; (iii) frozen and retrained models were scored on **different populations**. With
> all three fixed, the honest picture is different — and, in one respect, cleaner.

What holds after the corrections (held-out Panda, constitutive regime):

- **The local constitutive *compliance* transfers.** The white-box `F_n = pen / k_mat(float)` — per-
  material compliance learned on the floating gripper, applied to Panda penetration — reaches
  **`R²=0.72`**, *beating even the Panda-retrained MLP* (`R²=0.40`). Grip strategy scales `pen` and
  `F_n` together, so `k = F_n/pen` is ~grip-invariant and transfers. **Model structure beats brute-
  force learning here.**
- **The analytical port `W_nn` as a fitted feature does NOT help — it hurts.** `D`→`C` (adding the
  port) drops `R² 0.72 → −0.28`; the port MLP `B` is worse than local `A`. The reason: the two arms'
  `W_nn` distributions **barely overlap** (float `[47,132]`, Panda `[23,54]`), so any model using
  `W_nn` must **extrapolate**, and the linear/ReLU fits explode. Using the port properly needs the
  actual `(W+R)` **solve**, not a fitted coefficient.
- **Absolute `F_n` does not transfer.** The MLPs predict float-scale forces (~2–3 N) on the Panda's
  sub-newton contacts (median `F_n` differs **~6×** by grip strategy), so `R²` is hugely negative.
  Predict the grip-invariant *compliance*, not absolute force.

Honest bottom line (narrowed): what this establishes is **a transferable local *compliance* law** —
useful and clean — **not** that "the analytical port carries the embodiment" (the port, as a bare
feature, hurt), and **not** transferable grasp selection. The port likely still matters, but only
inside the real solve; testing that is the next step.

---

## 4. Contributions & novelty

- **(A) A computable transfer certificate** `‖Ĥ−H‖ ≤ (ε_Y+ε_C)/(m(m−ε))` with
  `m = σ_min(Y_G+C^{-1})`. *Lit-checked as genuinely novel* — prior port-Hamiltonian
  learning (van der Schaft; Neary–Topcu arXiv:2212.00893, L4DC 2023) and NeRD (arXiv:2508.15755) do
  not give a computable, embodiment-transfer error certificate.
- **(B) An observation-nullspace injectivity theorem** for internal forces + a
  proprioception-resolution condition (`ker O ⊆ ker P_τ`, defect `β_τ`). The facts are
  classical grasp mechanics; the *learning-theoretic packaging as a transfer admissibility
  test* is novel.
- **(Quotient) Admittance-quotient invariance** `H_e^{-1} - Y_e = C^{-1}`: the pre-training
  embodiment quotient. The algebra is exact by construction; its empirical weight comes from
  the exact port split (§3.3) and free-space port ID (§3.11), where `H` and `Y` are obtained
  independently. Novel in specifics; must be distinguished from compositional port-Hamiltonian
  learning in the writeup.
- **(New this line of work) The surface-covering *geometry***: recasting embodiment transfer as
  scattered-data approximation on the object surface, with a certificate computable from
  contact geometry alone. This dissolves the matched-geometry assumption and handles different
  finger counts / patch shapes / bimanual / humanoid as different measures on one field. The
  geometry and the physical-field bound are validated; the covering bound on a *learned* field
  is **not yet** established (RBF-KRR extrapolation violates it — §3.9, §14), and is the open
  learning-theoretic item.

**Falsifiable prediction (RQ1):** `C_θ` should **not** need the self-inertia block `W_ii`
(the constitutive law is inertia-independent). `W_ii` is logged separately from the strict
27-D `z_local` so this can be tested by ablation without regenerating data.

---

## 5. What we built & validated (with reasoning)

| # | Artifact | What it establishes | Key numbers |
|---|---|---|---|
| 1 | `contact_probe.py` | `W` assembly = ground truth | `W` vs `mj_solveM` = 0.0; ΣFn = weight |
| 2 | `z_local_schema.py` | frozen 27-D embodiment-agnostic feature schema | `κ_obj = 1/r` exact |
| 3 | `premises_final.py` | pre-embodiment GATE (P1/P2/P3) | P1 `1e-16`; P2 KKT `(W+R)f=aref−a_u` settled `1.6e-15` / **transient** `1.8e-16`; P3 ~linear |
| 4 | `verify_solve.py` | implicit gradient through the solve | err `1e-12` |
| 5 | `verify_math.py` | quotient / nullspace / certificate | quotient `1e-17` (exact *by construction*); nullspace rank test; certificate cond. 210× (synthetic) |
| 6 | `panda_embodiment.py` | 2nd real embodiment (Franka) grasps + lifts | schema-identical `z_local`, 0 NaN on pads |
| 7 | `cross_embodiment_v2.py` | exact port split + `Y_object` invariance + certificate | split `1e-17`, `Y_object=20.0` Δ`3.5e-15` |
| 8 | `surface_field_covering.py` | **covering geometry** on real supports | physical-field bound `0%` violated (gated); cross-embodiment `228×`; slope/L `1.389` is a linear-learner identity, not evidence |
| 9 | `hetero_covering.py` | covering on a **learned** field | **NEGATIVE**: RBF-KRR mean-reverts (`μ̂=−0.27`), violates bound `100%`; geometry OK (Lipschitz-consistent est. `0%`) |
| 10 | `graspability.py` | graspability = named margins | 8/8 curated (uses **post-contact** `ΣF_n` → consistency check, not a-priori screen) |
| 11 | `port_identification.py` | free-space `ε_Y` + two-source certificate | Panda naive port 94% wrong (real); ID is **instantaneous open-loop** (closed-loop `Y_G` open) |
| 12 | `deleak_dataset.py` + `deleak_train_eval.py` | de-leaked C_θ cross-embodiment transfer (no raw `μ/solref/solimp`), **audit-corrected** | held-out Panda `F_n`: white-box **compliance transfers R²=0.72** (> retrain MLP 0.40); the port `W_nn` as a feature **hurts** (non-overlapping); absolute `F_n` confounded by grip (~6×) |

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
3. **Done (audit-corrected, §3.12):** trained a de-leaked `C_θ`. Honest finding — the local
   **compliance** transfers (white-box `R²=0.72`); the port `W_nn` as a fitted feature does not
   help (non-overlapping distributions); absolute `F_n` is grip-confounded. **Next here:** use the
   real `(W+R)` **solve-in-the-loop** on predicted constitutive parameters (not `W_nn` as a bare
   feature), overlap the port distributions or evaluate on the grip-invariant compliance, and move
   from force regression toward **grasp decisions** (the reviewer's redirection).
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
2. `Y_G` is currently the rigid-body/constrained **instantaneous** port; the **closed-loop
   controller-shaped** port on real hardware is not yet identified, and the port ID uses a
   *known* applied wrench (an instrumented setup is needed on hardware).
3. `C_θ` is **trained** on a de-leaked dataset (§3.12). After an audit correction the honest
   result is: the local **compliance** transfers across embodiments (white-box `R²=0.72`,
   beating a Panda-retrained MLP), but the analytical port `W_nn` **as a fitted feature does not
   help — it hurts** (float/Panda `W_nn` distributions barely overlap → extrapolation), and
   absolute `F_n` does not transfer (grip strategy differs ~6×). So we have a transferable local
   compliance law, **not** a demonstration that the port carries the embodiment; using `W_nn`
   needs the real `(W+R)` **solve-in-the-loop**, which is the open item. The certificate itself
   is still verified only on synthetic + analytically-identified ports.
4. The **camera-only Lynxmotion cannot observe internal force** — a fundamental limit the
   certificate must respect by rejecting force-critical tasks.
5. **Covering law on a *learned* field is not established.** On the learned RBF friction field,
   the KRR estimator mean-reverts under extrapolation (`μ̂=−0.27`) and violates
   `ε+C·L·dist` on ~100% of out-of-support points. The covering *geometry* is sound (a
   Lipschitz-consistent estimator obeys the bound); a Lipschitz-respecting learner is the open
   item. (An earlier "hetero HOLDS 5.8%" was an artifact of a staircase-vs-ramp ground-truth
   mismatch plus a verdict that did not gate on violations — both now fixed.)
6. **`z_local` contains non-deployable inputs.** It includes the *true* `μ, solref, solimp`
   (MuJoCo constitutive parameters) as input features. These are **not observable by the D455**,
   and supplying true friction as an input is **leakage** for any task that claims to *infer*
   friction. Before training `C_θ`, the deployment observation model must move `μ/solref/solimp`
   to hidden/targets and split by trial (grouping ids exist), not by contact row.
7. **Several "exactness" results are exact by construction / synthetic**, i.e. implementation
   checks, not evidence of learned transfer: the `Y_object` cross-embodiment identity (object
   Jacobian carries no arm DOFs), the `H^{-1}−Y=C^{-1}` quotient (built from `Y+C`), and the
   `Y/C` gauge demonstration.
8. **The covering-law `slope/L` collapse across `L` is not independent evidence** — it follows
   by construction from a linear learner on a linearly-scaled field. The substantive content is
   the dependence on covering *distance* and the field-independent constant `C`.
9. **Graspability uses privileged post-contact information** (`μ_req = mg/ΣF_n` with measured
   `ΣF_n`) and a small curated battery — a consistency check, not an a-priori, statistically-
   characterized screen.
