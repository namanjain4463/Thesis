# Tier 2 — Certified, capability-aware cross-embodiment grasp selection

> **Status:** research + proposal (no experiments yet). This document synthesizes a literature
> review and proposes the next contribution (Contribution C), positioned to be novel against the
> closest prior work and grounded in what Tiers 0–1 already validated. It follows the 2nd reviewer's
> redirection: *move from contact-force regression to grasp decisions.*

## 0a. Frozen research question (3rd review) & the three-part package

> **RQ (frozen):** *Can a shared interaction model select grasp location and closing behavior on a new
> robot with **less robot-specific data** than separately calibrated or trained alternatives, **at
> comparable reliability**?*

Not assumed correct: the current factorization, any world-model architecture, or the "certificate."
Capability-aware grasp planning already exists (Chen IROS'18 `1710.11190`; King RSS'13); cross-embodiment
world models exist (`2511.01177`). The contribution must be a **measured** improvement over these, and
the conclusion **follows the results — including the calibrated analytical baseline winning.** Not
required: a ranking reversal (preserving a ranking is equally valid). The package is exactly three
additions, in order: **(2) command→contact calibration, (3) a complete grasp-and-place benchmark,
(4) an independent adaptation-data evaluation.** Progress is tracked in §8.

## 0. Provenance & verification honesty

Researched via four parallel literature-review subagents (2026‑09). **arxiv.org and publisher PDF
hosts were egress-blocked**, so every arXiv id below is cross-checked against the search-index result
URLs (often multiple independent hits) but abstract pages were **not** opened. Confidence flags:
`✓` id confirmed via search-index URL; `≈` verified via GitHub/project page or search summary only
(fresh or not-abstract-read); `⚠` from model memory, **not** confirmed — fetch before citing.
Two of the four agents were cut off by a session rate limit; the *cross-embodiment world-model*
cluster is therefore covered more thinly (from the completed briefs + prior knowledge) and is the
first place to deepen the review.

**Correction surfaced by the review:** README §4 cites Neary–Topcu as `arXiv:2412.11215` — that is
**wrong**. The correct paper is **`arXiv:2212.00893`** (L4DC 2023). Fixed in README.

## 1. Landscape (what exists)

### A. Cross-embodiment / gripper-conditioned grasp synthesis
The field conditions grasp generation on gripper **geometry/kinematics**, and generalizes across
grippers by that geometry:
- **GraspGen-X** ≈`2606.00998` (NVlabs, CVPR'26): single diffusion generator conditioned on a
  gripper **swept-volume** descriptor; ~2B grasps, 32 procedural grippers, beats AdaGrasp-TSDF ~25%
  zero-shot on novel grippers. *The single most-overlapping paper with any "gripper-conditioned" claim.*
- **D(R,O) / T(R,O) Grasp** ✓`2410.01702` / ≈`2510.12724`: unified robot–object interaction
  representation; conditions on hand **kinematic description**; kinematically-valid dexterous grasps.
- **MachaGrasp** ✓`2510.06068`: morphology-aware few-shot to unseen hands (eigengrasps/morphology
  embeddings). *Nearest existing "reduced-data on unseen embodiment" claim* — but adaptation signal is
  grasp poses + kinematic morphology, sim dexterous hands.
- **AdaGrasp** ✓`2011.14206`, **UniGrasp** ✓`1910.10900`, **RobotFingerPrint/UGCS** ✓`2409.14519`:
  gripper-aware policy / cross-gripper geometric correspondence.
- Single-gripper detectors (embodiment implicit): **Contact-GraspNet** ✓`2103.14127`,
  **AnyGrasp** ✓`2212.08333` (depth-noise-robust; relevant to D455), **6-DOF GraspNet** ✓`1905.10520`.

### B. Physics/capability/uncertainty-aware grasp selection
- **iTuP / SDG-Net** ✓`2505.01399`: selects grasps by minimizing predicted **interaction wrench**
  along the task trajectory (torque/slip penalties). *Closest "physics not geometry" selection* — but
  single-embodiment mechanics, no transferable contact model.
- **Consensus-driven uncertainty** ✓`2506.20045`: **refuses** grasps when pose-estimator uncertainty
  is high. *Closest "refusal"* — but RGB-pose uncertainty for a fixed gripper, not capability/sensing.
- **GRASPA 1.0** ✓`2002.05017`: graspability score that includes object weight vs robot **payload** —
  a rare explicit capability term, but a post-hoc score, not a selection channel.
- **Reachability-/manipulability-gated ranking** ✓`2103.10562`; multi-criteria ranking **GRaCE**
  ✓`2309.08887`.

### C. Structured / certified contact dynamics
- **ContactNets** ✓`2009.11193` + **Simultaneous learning** ✓`2310.12054` (Posa group): learn a
  physics-consistent contact model of **one object**, robot dynamics assumed known; no embodiment
  split, no transfer bound.
- **Neary–Topcu port-Hamiltonian composition** ✓`2212.00893` (L4DC'23): **Thm 1** bounds composite
  error by subsystem errors — *nearest compositional bound* — but composes subsystems into a bigger
  system (energy ports), not a frozen law onto a new embodiment; no contact, bound not used to decide.
- **Lipschitz-gated planning** ✓`2010.08993` (Knuth et al.): uses model-error Lipschitz + data
  coverage to **refuse to plan** outside a trusted region — *nearest certificate-that-acts* — but
  monolithic model, no contact factorization, no port.
- **Diff-sim contact gradients are often wrong** ✓`2207.05060`; **stiff contact wrecks NN learning**
  ✓`2103.15406` — both **externally corroborate** our decision to use the real `(W+R)` solve rather
  than a learned port feature, and structure over black-box.
- Convex contact solves: **SAP** ✓`2110.10107`, **MuJoCo/Todorov'14** (Delassus source).

### D. Cross-embodiment world models / policies (thinner coverage — deepen later)
- **HPT** ✓`2409.20537` (heterogeneous pretrained transformers), **RoboCraft** ✓`2205.02909`
  (graph-net deformable dynamics); Open‑X/RT‑X, cross-embodiment world models (reviewer's
  `2511.01177`), XIRL/CrossFormer ⚠. These learn **shared latents / policies across embodiments** —
  no physics-factored contact **port**, no certificate, no capability-driven ranking reversal.

## 2. The convergent GAP

> **CORRECTION (3rd review).** An earlier version of this section claimed "no method selects grasps
> using measured physical capability, nor rejects configurations because of grip/actuator limits."
> **That is false.** *Manipulation Planning under Changing External Forces* (Chen et al., **IROS 2018**,
> `arXiv:1710.11190`) uses **measured** gripper force/torque limits + manipulator joint-torque limits
> to check stability and plan unimanual↔bimanual regrasps; *Pregrasp Manipulation as Trajectory
> Optimization* (King et al., **RSS 2013**) plans object reconfiguration when joint-torque limits
> prevent a direct lift. So **capability-aware grasp selection is NOT a sufficient novelty claim.**
> These are model-based planners with known parameters; the remaining, *measurable* opportunity is a
> **learned** interaction model that improves cross-body grasp decisions with **less new-body data**
> than strong **calibrated analytical + learned baselines** — a claim that must be *measured*, not
> asserted (a name, a physics margin, or a ranking reversal do not establish it). The items below are
> re-scoped accordingly.

Across the clusters, what remains genuinely under-served:

1. **Learning the *uncertain* part of capability, not re-deriving the known part.** The capability-aware
   planners above assume known force/torque limits and rigid mechanics. Gripper-conditioned learners
   condition on **geometry** (swept volume, URDF, morphology). Neither *learns* the uncertain
   interface/command-response behavior (compliance, slip onset, controller realization) that decides
   whether a grasp stays feasible **during execution on a new body** — and shows a **data-efficiency**
   advantage there. (Note: the Delassus port `W=JM⁻¹Jᵀ` is a **response** operator, *not* the capability
   set — see the §3.1 correction.)
2. **No transferable, embodiment-factored contact model drives selection.** Physics-based selectors
   (iTuP, force-closure) use a **single-embodiment** mechanics model retrained per robot. The
   structured-contact line (ContactNets, Neary–Topcu, Knuth) never does the `W = Y_object + Y_robot`
   split, never freezes a local law and recomposes with a per-embodiment analytical port, and never
   turns a transfer bound into a **grasp go/no-go**.
3. **No capability-and-sensing refusal.** Only pose-uncertainty refusal exists. No method refuses a
   grasp because *this embodiment cannot deliver the required contact wrench*, or because
   *camera-only sensing cannot verify the internal squeeze the task needs* (our internal-force
   nullspace result).

## 3. Proposed Contribution C — **Port-Conditioned Certified Grasp Selection (PC‑CGS)**

**One sentence (corrected, measurable).** *A frozen interaction model that predicts **sustained**
grasp outcomes across bodies by composing an embodiment-invariant learned contact law with each body's
analytical response (port) inside the real solve, and that — evaluated on **held-out** embodiments and
objects against the **CoM** and **calibrated wrench-feasibility** baselines — **improves grasp
decisions with less new-body data** than those baselines, while abstaining only when a **calibrated
error bound** cannot certify the task margin.* (Not claimed: that a ranking must reverse — preserving a
ranking is equally valid; that `W` is capability — it is response, §3.1; that a fitted positive margin
is a "certificate" — it needs a tested error bound, §7. The value is the **measured data-efficiency /
held-out accuracy delta**, nothing else.)

### 3.1 Two distinct components — RESPONSE vs CAPABILITY (3rd-review correction)
An earlier draft said "the analytical port `Y_robot` (Delassus `W=JM⁻¹Jᵀ`) *is* the missing
measured-capability signal." **That conflates two different objects, and the corrected benchmark
(§6) shows the difference concretely** — changing the actuator force limit changed the grasp outcome
**without changing `M`, `J`, or `W` at a fixed configuration.** Keep them separate:

| Component | Question it answers | Object |
|---|---|---|
| **Mechanical + controller RESPONSE** | what happens when a command / contact force is applied? | the port `W=JM⁻¹Jᵀ` (+ the closed-loop controller) |
| **Feasible contact-wrench CAPABILITY** | which force/moment combinations can this body *sustain*? | `𝒦_E(g) = { G(g)f : f in the friction cone, required actuator/joint loads within limits, config/support constraints hold }` |

A static task is feasible iff its balancing wrench lies in `𝒦_E(g)`; dynamic execution additionally
needs a valid trajectory with achievable commands. `W` does **not** by itself bound deliverable force.
So the corrected job for the **learned** model is narrow and testable: **improve predictions of the
*uncertain* interface behavior and command response** (compliance, slip onset, controller realization)
that decide whether a candidate stays feasible *during execution* — the part `𝒦_E` with known limits
does not capture. Tier‑1 still guides *how* to use `W`: as a fitted feature it hurt (extrapolation);
the frozen **local compliance transferred** (white-box `R²=0.72`, though **median rel-err ≈55%** — an
aggregate fit, not accurate force control). Put `W` inside the real solve where it belongs, and only
in cases involving **acceleration, changing contacts, controller response, or coupled motion** — not
forced into a static selector to preserve the original thesis equation (`2207.05060`, `2103.15406`
back structure over black-box; the review backs not overclaiming `W` as capability).

### 3.2 The model (three legs)
For a candidate grasp `a = {contact patches, hand assignment, approach, closing, transport}` on
object estimate `b_t` with embodiment `E`:

1. **Port-conditioned interaction rollout.** `a` + object geometry → contact set → the analytical
   port `Y_G(a,E)` (per-embodiment, recomputed, no learning). Predicted contact wrench / object motion
   `= Solve(frozen C_θ(z_local), Y_G(a,E))` via the convex `(W+R)` solve. Rank grasps by a task cost
   `J` through this composed interaction — **mechanics, transferred, not geometry**.
2. **Certificate gate (accept / probe / reject), each with a physical reason.** Per candidate compute
   (i) **kinematic** margin `γ_kin` (reach/finger-span), (ii) **wrench/force-limit** margin — can
   `Y_G` + actuator limits deliver the task wrench (friction-cone `γ_fric`), (iii) the **transfer
   certificate** `‖ΔH‖ ≤ (ε_Y+ε_C)/(m(m−ε))`, `m=σ_min(Y_G+C⁻¹)` — does the frozen law transfer to
   this port within tolerance, (iv) the **observation/internal-force test** `ker O ⊆ ker P_τ` — can
   this body's sensing (camera-only ⇒ no squeeze) verify what the task needs; if not and the task is
   force-critical → **reject**. Output a calibrated decision naming the binding margin.
3. **Certified ranking reversal.** Because `Y_G` is recomputed analytically per body, the **same
   frozen model** yields a **different best grasp** when the embodiment changes — and the margins
   *name why* (reach vs force-limit vs balance vs unobservable). The change is **certified**, not
   merely learned.

### 3.3 Why this is novel (guardrails satisfied, per closest work)
- vs **GraspGen-X / D(R,O)/T(R,O)** — they condition on gripper **geometry/kinematics**; PC‑CGS
  conditions on the analytical **contact port** (a *physical capability*) and ranks by a *transferable
  contact-mechanics* outcome, not a learned geometric score.
- vs **MachaGrasp** — data-efficiency comes from the **analytical port swap** (zero new grasp data for
  the new body's capability), reported as a measured adaptation-data delta on a *physically distinct
  real arm* (Lynxmotion, no force sensing), not sim-family morphology fine-tuning.
- vs **Consensus-uncertainty / AnyGrasp** — refusal is **capability-and-sensing** driven (can't
  deliver the wrench / can't observe the needed internal force), from the internal-force nullspace +
  port, not pose-uncertainty robustness.
- vs **iTuP / force-closure** — uses a **transferable, embodiment-factored** contact model (frozen
  `C_θ` + per-embodiment port) with a **certificate**, not a single-embodiment mechanics net.
- vs **Neary–Topcu** — a **two-source (ε_Y+ε_C)** bound on the **Delassus contact operator** used for
  a **grasp decision**, not an energy-composition subsystem bound for prediction.
- vs **Knuth (Lipschitz-gated)** — the gate drives **grasp selection/refusal** on a **factorized**
  contact model, not a monolithic-model trust region for planning.
- vs **ContactNets** — splits `W=Y_obj+Y_robot` and **freezes the local law across embodiments**;
  ContactNets keeps robot dynamics fixed and models one system.

## 4. The decisive experiment — **certified grasp-ranking reversal**

Same object + task; change the embodiment; the correct grasp must change for a physically
identifiable reason, and the frozen model + certificate must predict the change **and its reason**.

**Bodies (increasing distance):** floating 2‑finger → Franka Panda → a Lynxmotion-like low-payload
3‑finger arm (camera-only). **Later:** a bimanual rig and a humanoid (shared base) — the port simply
becomes the multiport `Y_G = Bᵀ blkdiag(Y_1..Y_n,Y_obj) B`.

**Reversal scenarios (each isolates one physical cause):**
| Scenario | Correct behavior | Binding margin |
|---|---|---|
| Off-center CoM, heavy | shift grasp / share load; weak body must pick a deliverable grasp or refuse | wrench/force-limit |
| Object reachable by one body only | different approach/grasp | kinematic `γ_kin` |
| Slippery + heavy (force-critical) | force-sensored body accepts; **camera-only body REJECTS** | observation `ker O ⊆ ker P_τ` |
| Two inadequate single grasps | recruit 2nd hand — but do **not** assume a 2nd hand guarantees success | hand-participation + wrench |
| Denser contact sampling | **no** artificial increase in predicted holding capacity | certificate stability |

**Metrics (decisions, not force R²):** selected-grasp success & task completion in full sim;
**ranking-reversal accuracy** (does the frozen model's argmax match ground-truth feasibility across
bodies) and **reason accuracy** (does the binding margin match the true cause); **false-accept /
false-reject**; **unnecessary second-hand use**; and the **adaptation-data curve** — new-body data to
match a per-body-retrained selector (PC‑CGS target ≈ 0 for the capability channel).

**Baselines:** (i) geometry-conditioned selector (GraspGen-X-style proxy), (ii) per-body **retrained**
selector (reference), (iii) analytical-only (no learned law), (iv) single-embodiment physics selector
(iTuP-style). This separates *value of the factorization* from *value of learning* and from *value of
supplied physics* — the same three-way separation the 2nd reviewer demanded in Tier‑1.

## 5. Honest risks & limits
- **Grip-strategy / operating-point confound** (seen in Tier‑1, ~6× force scale) must be controlled by
  ranking on **grip-invariant** quantities (task feasibility, compliance) not absolute force.
- **`Y_G` is still the instantaneous/rigid port**; the closed-loop controller-shaped port (esp.
  Lynxmotion servo) is unidentified — the certificate's `ε_Y` must carry this, and reject when large.
- **Everything within-simulator** until a sim-to-real `ε_C` stress test exists (Tier‑0 open item).
- **Reduced-data claim is the risky one** — it must be a *measured* delta on a physically distinct
  body, not a sim-family number (the MachaGrasp trap).
- **No standardized benchmark** for capability-dependent ranking reversal across embodiments was found
  — we would define one (a contribution, but also a burden of proof: baselines must be strong).

## 6. First concrete build — built, then **corrected to a null result** (`mujoco/grasp_ranking_reversal.py`)
A MuJoCo micro-benchmark on the force/moment axis: off-center-CoM bar, candidate grasps, a strong vs a
weak gripper. The first version claimed a **certified ranking reversal** (strong→center grasp,
weak→CoM-ward) using a **0.12 s** hold. **A 3rd reviewer overturned it and this is confirmed:** at a
properly specified **2 s** hold the center grasp is slowly tipping for **both** bodies and fails, so
**both** prefer the CoM-ward grasp — **the reversal disappears.** At 2 s the two bodies have
**identical** feasible sets (the weak grip still holds the near-CoM grasps); the capability gap
survives only as a **transient tilt margin**, not a change in the sustained decision. A trivial
**CoM baseline** holds for both → **no task advantage** for the capability-aware ranker in this scene.
The benchmark now reports multi-horizon holds (0.12/0.5/2 s), separated failure modes (TIP vs DROP),
and the CoM baseline; the moment-rule lever is flagged as fit **in-sample** (not zero-shot). Figure +
log: `outputs/rankrev_reversal.{png,txt}`.

**Lessons folded into the plan (from the 3rd review):** specify the task horizon and score *sustained*
grasps; separate failure modes; add the CoM and calibrated-wrench-feasibility baselines and only claim
value if the learned method *beats* them on held-out cases; calibrate parameters on *separate* objects
and freeze before evaluation; and do not force a reversal — correctly *preserving* a ranking is a valid
outcome. **Next:** a scene where an off-CoM grasp is required for a genuine *task* reason (destination
clearance / mounting feature), a genuinely different embodiment (Panda **reach/port**, not a grip-force
knob), and an **independently calibrated** rule evaluated against the CoM + wrench-feasibility
baselines (priority tests in §5 of the review; see also the corrected §2–§3 below).

## 7. References (confidence-flagged)
Grasp synthesis: GraspGen-X ≈2606.00998 · D(R,O) ✓2410.01702 · T(R,O) ≈2510.12724 · MachaGrasp
✓2510.06068 · AdaGrasp ✓2011.14206 · UniGrasp ✓1910.10900 · RobotFingerPrint ✓2409.14519 ·
Contact-GraspNet ✓2103.14127 · AnyGrasp ✓2212.08333 · 6-DOF GraspNet ✓1905.10520 · GenDexGrasp
✓2210.00722 · DexGraspNet ✓2210.02697 · GraspGen ✓2507.13097.
Selection/uncertainty: iTuP/SDG-Net ✓2505.01399 · Consensus-uncertainty ✓2506.20045 · GRASPA
✓2002.05017 · reachability-ranking ✓2103.10562 · GRaCE ✓2309.08887.
Contact/certified: ContactNets ✓2009.11193 · Simultaneous-learning ✓2310.12054 · **Neary–Topcu
✓2212.00893** · Knuth Lipschitz-gate ✓2010.08993 · diff-sim-gradients ✓2207.05060 · stiff-contact
✓2103.15406 · SAP ✓2110.10107 · Nimble ✓2103.16021 · GNS ✓2002.09405 · TossingBot ✓1903.11239.
World models: HPT ✓2409.20537 · RoboCraft ✓2205.02909 · cross-embodiment WM (reviewer) 2511.01177 ·
XIRL/CrossFormer ⚠. From-memory (fetch before citing): HNN ⚠1906.01563 · DeLaN ⚠1907.04490 · Kloss ⚠1710.04102.

## 8. Package progress (3rd-review milestone)

RQ (§0a): *can a shared interaction model select grasp location + closing behavior on a new robot with
less robot-specific data than calibrated/trained alternatives, at comparable reliability?*

- **(2) command→contact calibration — DONE** (`mujoco/command_calibration.py`,
  `outputs/command_calibration.{png,txt}`). Swept 192 command/config combos (closing target, closing
  speed, actuator force limit, object width, interface compliance) recording synchronized command,
  finger pos/vel, actuator force, contact force, object motion, contact-formation time. **Delivered
  contact force is predictable from command+config by a parameter-free analytical command-response
  model `2*clip(kp*(target - x_contact), 0, F_limit)` — held-out R2=0.998, median rel-err 0.9%.** The
  naive "delivered = force cap" proxy (used in the overturned ranking benchmark) is badly wrong:
  held-out R2=-20, over-stating delivered force by ~2.5x on the 60% of combos where the actuator does
  NOT saturate (it saturates, delivered~cap, only for weak grips). **A learned correction adds nothing
  (-0.2 pts)** -> for the command->force map the calibrated analytical model wins; learning is
  unjustified here (a valid, reported outcome per the §7 decision table). This replaces the bad
  capability estimate and gives step (3) the correct deliverable-force model.
- **(3) complete grasp-and-place benchmark — TODO.** Three object families (uniform / off-center /
  placement-restricted); fixed task (transport+place, pose/slip tolerances, support duration, failure
  definitions, selector observations); task-set horizon (not a fixed 2 s); candidates + controller
  identical across methods.
- **(4) independent adaptation-data evaluation — TODO.** Shared model on source embodiments; hold out
  target embodiment+controller; evaluate at 0/5/20/50 target-episode budgets vs CoM + calibrated-
  wrench-feasibility + target-trained baselines + an oracle diagnostic; count ALL target-specific
  effort; report success + uncertainty across independent episodes. Deliverable: an adaptation curve
  (less new-robot work at equal reliability), or the honest negative.

`grasp_ranking_reversal.py` is kept as a **regression test** (the corrected null); reproducing a
reversal is not an objective.
